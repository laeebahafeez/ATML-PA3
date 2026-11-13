"""Task 2 data selection: COCO captions, annotation-derived grounding pairs, VQAv2 pairs.

    python -m task2.data --root data/task2 --out manifests/task2

Everything is chosen with seed 6304 from the annotation files only (no images needed);
the output manifests list every selected image, caption, question and pair ID.

Splits: train2014 -> training; val2014 -> development and final test. Each val2014
image is assigned to exactly one evaluation suite, so all dev/test suites are
image-disjoint from each other and from training.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from common.utils import SEED, write_json
from task2.vqa_norm import consensus, normalize

N_CAPTION_TRAIN, CAPS_PER_TRAIN_IMAGE, N_CAPTION_DEV = 6000, 3, 96
GROUND_DEV = {"presence": 12, "counting": 12}           # 24 pairs / 48 questions
GROUND_TEST = {"presence": 32, "counting": 32}          # 64 pairs / 128 questions
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}
MIN_AGREE = 8
POOL_PAIRS_PER_CATEGORY = 60

# reliability thresholds for annotation-derived labels (fraction of image area)
PRESENCE_MIN_AREA = 0.02     # "yes": at least one clearly visible instance
COUNT_MIN_AREA = 0.01        # counting: every instance at least this large
COUNT_RANGE = (1, 4)

PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "pairs of skis",
          "scissors": "pairs of scissors", "bus": "buses", "couch": "couches", "bench": "benches",
          "sandwich": "sandwiches", "wine glass": "wine glasses", "knife": "knives",
          "toothbrush": "toothbrushes", "hair drier": "hair driers", "skateboard": "skateboards",
          "dining table": "dining tables", "giraffe": "giraffes", "potted plant": "potted plants",
          "broccoli": "pieces of broccoli"}


def plural(name):
    return PLURAL.get(name, name + "s")


def article(name):
    if name in ("skis", "scissors"):
        return "a pair of " + name
    if name == "broccoli":
        return "broccoli"
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_q(name):
    return f"Does this image contain {article(name)}?"


def counting_q(name):
    return f"How many {plural(name)} can you see?"


def img_url(split, iid):
    return f"http://images.cocodataset.org/{split}/COCO_{split}_{iid:012d}.jpg"


# ---------------------------------------------------------------- grounding labels
def object_index(inst):
    """image -> category -> dict(n, n_crowd, min_frac, max_frac); plus image sizes."""
    imgs = {im["id"]: im for im in inst["images"]}
    idx = defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_crowd": 0, "min_frac": 1.0, "max_frac": 0.0}))
    for a in inst["annotations"]:
        im = imgs[a["image_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        d = idx[a["image_id"]][a["category_id"]]
        if a["iscrowd"]:
            d["n_crowd"] += 1
        else:
            d["n"] += 1
            d["min_frac"] = min(d["min_frac"], frac)
            d["max_frac"] = max(d["max_frac"], frac)
    return idx


def presence_yes(d):
    return d is not None and d["n_crowd"] == 0 and d["n"] >= 1 and d["max_frac"] >= PRESENCE_MIN_AREA


def count_label(d):
    """Reliable count in COUNT_RANGE or None."""
    if d is None or d["n_crowd"] or not (COUNT_RANGE[0] <= d["n"] <= COUNT_RANGE[1]):
        return None
    return d["n"] if d["min_frac"] >= COUNT_MIN_AREA else None


def grounding_pairs(image_ids, idx, cats, rng, n_per_cat=None, kinds=("presence", "counting")):
    """Candidate pairs from `image_ids`. Returns dict kind -> list of pairs."""
    image_ids = sorted(image_ids)
    out = {"presence": [], "counting": []}
    for cid in sorted(cats):
        name = cats[cid]
        yes = [i for i in image_ids if presence_yes(idx.get(i, {}).get(cid))]
        no = [i for i in image_ids if cid not in idx.get(i, {})]
        if "presence" in kinds and yes and no:
            ys = rng.permutation(yes)[: n_per_cat or len(yes)]
            ns = rng.choice(no, size=len(ys), replace=len(no) < len(ys))
            for y, n in zip(ys, ns):
                out["presence"].append({"kind": "presence", "category": name, "question": presence_q(name),
                                        "images": [int(y), int(n)], "answers": ["yes", "no"]})
        if "counting" in kinds:
            by_c = defaultdict(list)
            for i in image_ids:
                c = count_label(idx.get(i, {}).get(cid))
                if c is not None:
                    by_c[c].append(i)
            counts = sorted(by_c)
            cand = []
            for a_i, ca in enumerate(counts):
                for cb in counts[a_i + 1:]:
                    k = min(len(by_c[ca]), len(by_c[cb]))
                    A = rng.permutation(by_c[ca])[:k]
                    B = rng.permutation(by_c[cb])[:k]
                    cand += [(int(x), int(y), ca, cb) for x, y in zip(A, B)]
            if cand:
                sel = rng.permutation(len(cand))[: n_per_cat or len(cand)]
                for j in sel:
                    x, y, ca, cb = cand[j]
                    if rng.random() < 0.5:        # randomize which count comes first
                        x, y, ca, cb = y, x, cb, ca
                    out["counting"].append({"kind": "counting", "category": name, "question": counting_q(name),
                                            "images": [x, y], "answers": [str(ca), str(cb)]})
    return out


def take_disjoint(pairs, n, rng, used):
    """n pairs whose images are unused (and unique within the selection)."""
    out = []
    for j in rng.permutation(len(pairs)):
        p = pairs[j]
        if any(i in used for i in p["images"]) or p["images"][0] == p["images"][1]:
            continue
        out.append(p)
        used.update(p["images"])
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)}/{n} disjoint pairs")


# ---------------------------------------------------------------- VQAv2
def vqa_candidates(root, split):
    tag = "train2014" if split == "train" else "val2014"
    qs = {q["question_id"]: q for q in json.load(open(f"{root}/v2_OpenEnded_mscoco_{tag}_questions.json"))["questions"]}
    an = {a["question_id"]: a for a in json.load(open(f"{root}/v2_mscoco_{tag}_annotations.json"))["annotations"]}
    pairs = json.load(open(f"{root}/v2_mscoco_{tag}_complementary_pairs.json"))
    out = defaultdict(list)
    for q1, q2 in sorted(map(tuple, pairs)):
        a1, a2 = an[q1], an[q2]
        if a1["answer_type"] != a2["answer_type"] or a1["image_id"] == a2["image_id"]:
            continue
        refs = [[x["answer"] for x in a["answers"]] for a in (a1, a2)]
        cons = [consensus(r) for r in refs]
        if any(n < MIN_AGREE for _, n in cons) or cons[0][0] == cons[1][0]:
            continue
        targets = []
        for r, (c, _) in zip(refs, cons):     # an original string from the consensus group
            group = [x for x in r if normalize(x) == c]
            targets.append(max(group, key=lambda s: (group.count(s), -group.index(s))))
        out[a1["answer_type"]].append({
            "kind": a1["answer_type"], "question_ids": [q1, q2],
            "question": qs[q1]["question"], "question_2": qs[q2]["question"],
            "images": [a1["image_id"], a2["image_id"]], "answers": targets,
            "consensus": [c for c, _ in cons], "references": refs,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/task2")
    ap.add_argument("--out", default="manifests/task2")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    os.makedirs(a.out, exist_ok=True)
    R = a.root

    # ---------- captions ----------
    cap_tr = json.load(open(f"{R}/annotations/captions_train2014.json"))
    cap_va = json.load(open(f"{R}/annotations/captions_val2014.json"))
    caps_by = lambda d: {k: sorted(v, key=lambda x: x["id"]) for k, v in _group(d["annotations"]).items()}
    caps_tr, caps_va = caps_by(cap_tr), caps_by(cap_va)

    # ---------- val image assignment (dev pool / test pool) ----------
    val_ids = sorted(caps_va)
    perm = rng.permutation(len(val_ids))
    dev_pool = {val_ids[i] for i in perm[: len(val_ids) // 3]}
    test_pool = set(val_ids) - dev_pool
    used_eval = set()

    # ---------- VQAv2 ----------
    vtr, vva = vqa_candidates(R, "train"), vqa_candidates(R, "val")
    vqa = {"train": [], "dev": [], "test": []}
    for t, n in VQA_TRAIN.items():      # training images may recur, so no disjointness here
        vqa["train"] += [vtr[t][j] for j in rng.permutation(len(vtr[t]))[:n]]
    for split, quota, pool in (("dev", VQA_DEV, dev_pool), ("test", VQA_TEST, test_pool)):
        for t, n in quota.items():
            cands = [p for p in vva[t] if all(i in pool for i in p["images"])]
            vqa[split] += take_disjoint(cands, n, rng, used_eval)

    # ---------- caption training images (disjoint from nothing in particular; train only) ----------
    tr_ids = sorted(caps_tr)
    cap_train_imgs = [tr_ids[i] for i in rng.permutation(len(tr_ids))[:N_CAPTION_TRAIN]]
    captions = {"train": [], "dev": []}
    for iid in cap_train_imgs:
        cs = caps_tr[iid]
        for j in sorted(rng.choice(len(cs), CAPS_PER_TRAIN_IMAGE, replace=False)):
            captions["train"].append({"image": int(iid), "caption": cs[j]["caption"].strip(), "caption_id": cs[j]["id"]})
    dev_cap_cands = sorted(i for i in dev_pool if i not in used_eval)
    for j in rng.permutation(len(dev_cap_cands))[:N_CAPTION_DEV]:
        iid = dev_cap_cands[j]
        used_eval.add(iid)
        captions["dev"].append({"image": int(iid), "caption": caps_va[iid][0]["caption"].strip(),
                                "caption_id": caps_va[iid][0]["id"]})

    # ---------- grounding ----------
    ins_tr = json.load(open(f"{R}/annotations/instances_train2014.json"))
    ins_va = json.load(open(f"{R}/annotations/instances_val2014.json"))
    cats = {c["id"]: c["name"] for c in ins_tr["categories"]}
    idx_tr, idx_va = object_index(ins_tr), object_index(ins_va)
    # training pools are built on the caption-training images (their features are cached anyway)
    pools = grounding_pairs(cap_train_imgs, idx_tr, cats, rng, n_per_cat=POOL_PAIRS_PER_CATEGORY)
    grounding = {"train_pools": pools, "dev": [], "test": []}
    for split, quota, pool in (("dev", GROUND_DEV, dev_pool), ("test", GROUND_TEST, test_pool)):
        ids = sorted(i for i in pool if i not in used_eval)
        cands = grounding_pairs(ids, idx_va, cats, rng, n_per_cat=8)
        for kind, n in quota.items():
            grounding[split] += take_disjoint(cands[kind], n, rng, used_eval)

    # ---------- image list ----------
    images = {}
    def add(iid, split, role):
        e = images.setdefault(int(iid), {"split": split, "url": img_url(split, int(iid)), "roles": []})
        assert e["split"] == split
        if role not in e["roles"]:
            e["roles"].append(role)
    for c in captions["train"]:
        add(c["image"], "train2014", "caption_train")
    for kind in ("presence", "counting"):
        for p in pools[kind]:
            for i in p["images"]:
                add(i, "train2014", "grounding_train")
    for p in vqa["train"]:
        for i in p["images"]:
            add(i, "train2014", "vqa_train")
    for c in captions["dev"]:
        add(c["image"], "val2014", "caption_dev")
    for s in ("dev", "test"):
        for p in grounding[s]:
            for i in p["images"]:
                add(i, "val2014", f"grounding_{s}")
        for p in vqa[s]:
            for i in p["images"]:
                add(i, "val2014", f"vqa_{s}")
    # isolation checks
    roles = lambda r: {i for i, e in images.items() if r in e["roles"]}
    train_imgs = roles("caption_train") | roles("grounding_train") | roles("vqa_train")
    dev_imgs = roles("caption_dev") | roles("grounding_dev") | roles("vqa_dev")
    test_imgs = roles("grounding_test") | roles("vqa_test")
    assert not (train_imgs & dev_imgs) and not (train_imgs & test_imgs) and not (dev_imgs & test_imgs)

    write_json(captions, f"{a.out}/captions.json", indent=None)
    write_json(grounding, f"{a.out}/grounding.json", indent=None)
    write_json(vqa, f"{a.out}/vqa.json", indent=None)
    write_json({str(k): v for k, v in sorted(images.items())}, f"{a.out}/images.json", indent=None)
    stats = {
        "captions": {k: len(v) for k, v in captions.items()},
        "grounding_train_pool_pairs": {k: len(v) for k, v in pools.items()},
        "grounding_eval_pairs": {s: {k: sum(p["kind"] == k for p in grounding[s]) for k in ("presence", "counting")} for s in ("dev", "test")},
        "vqa_pairs": {s: {t: sum(p["kind"] == t for p in vqa[s]) for t in VQA_TRAIN} for s in vqa},
        "vqa_eligible_train": {t: len(v) for t, v in vtr.items()},
        "vqa_eligible_val": {t: len(v) for t, v in vva.items()},
        "images": {"train": len(train_imgs), "dev": len(dev_imgs), "test": len(test_imgs), "total": len(images)},
        "thresholds": {"presence_min_area": PRESENCE_MIN_AREA, "count_min_area": COUNT_MIN_AREA,
                       "count_range": COUNT_RANGE, "min_agree": MIN_AGREE,
                       "pool_pairs_per_category": POOL_PAIRS_PER_CATEGORY},
    }
    write_json(stats, f"{a.out}/stats.json")
    print(json.dumps(stats, indent=1))
    write_audit_html(grounding, vqa, f"{a.out}/label_audit.html")


def _group(anns):
    g = defaultdict(list)
    for x in anns:
        g[x["image_id"]].append(x)
    return g


def write_audit_html(grounding, vqa, path):
    """Contact sheet of all dev/test grounding pairs (COCO URLs) for label inspection
    independent of any model predictions."""
    rows = []
    for s in ("dev", "test"):
        for p in grounding[s]:
            cells = "".join(f'<td><img src="{img_url("val2014", i)}" height="180"><br>{a}</td>'
                            for i, a in zip(p["images"], p["answers"]))
            rows.append(f"<tr><td>{s}</td><td>{p['kind']}</td><td>{p['question']}</td>{cells}</tr>")
    html = ("<html><head><meta charset='utf-8'><title>Task 2 label audit</title></head><body>"
            "<table border=1 cellpadding=4>" + "".join(rows) + "</table></body></html>")
    open(path, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    main()
