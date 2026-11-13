"""Task 2 final evaluation on both test suites under four image conditions.

    python -m task2.evaluate --run outputs/task2/run196 --out results/task2/run196

Conditions: correct, complementary swap (partner image), mismatched (fixed unrelated image
from another pair of the same suite), no image (visual span removed). Questions and the
ORIGINAL references are kept fixed across conditions. Greedy decoding, 16 new tokens,
batch size 8, thinking disabled.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from common.utils import SEED, read_json, write_json
from task2 import vlm as V
from task2.train import Data
from task2.vqa_norm import normalize, soft_accuracy

CONDITIONS = ("correct", "swap", "mismatched", "no_image")
N_BOOT = 4000


def mismatch_map(pairs, seed=SEED):
    """For every (pair, k): an image from a different pair of the same suite."""
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(pairs):
        row = []
        for _ in range(2):
            j = int(rng.choice([x for x in range(len(pairs)) if x != i]))
            img = pairs[j]["images"][int(rng.integers(2))]
            assert img not in p["images"]
            row.append(img)
        out.append(row)
    return out


def load_model(run, device, n_vis, model_id, revision):
    model = V.VLM(device, n_vis, model_id, revision)
    model.enable_lora()
    w = load_file(os.path.join(run, "stage3_vqa.safetensors"))
    model.connector.load_state_dict({k[10:]: v for k, v in w.items() if k.startswith("connector.")})
    named = dict(model.lm.named_parameters())
    for k, v in w.items():
        if k.startswith("lm."):
            named[k[3:]].data.copy_(v.to(named[k[3:]].dtype))
    model.vis_scale.fill_(read_json(os.path.join(run, "calibration.json"))["scale"])
    return model.eval()


def score(suite, pair, k, pred):
    ref = pair["answers"][k]
    if suite == "grounding":
        acc = float(normalize(pred) == normalize(ref))
        canon_ok = acc == 1.0
    else:
        acc = soft_accuracy(pred, pair["references"][k])
        canon_ok = normalize(pred) == pair["consensus"][k]
    return acc, canon_ok


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    bs = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/task2/run196")
    ap.add_argument("--features", default="outputs/task2/features")
    ap.add_argument("--manifests", default="manifests/task2")
    ap.add_argument("--out", default="results/task2/run196")
    ap.add_argument("--n_vis", type=int, default=196)
    ap.add_argument("--model", default=V.QWEN)
    ap.add_argument("--revision", default=V.QWEN_REV)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    feats = torch.from_numpy(np.array(np.load(os.path.join(a.features, "features.npy"), mmap_mode="r")))
    fidx = read_json(os.path.join(a.features, "index.json"))
    tok = V.load_tokenizer(a.model, a.revision)
    fmt = V.Formatter(tok, a.n_vis)
    data = Data(a.manifests, fidx, fmt)
    model = load_model(a.run, device, a.n_vis, a.model, a.revision)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, mappings = [], {}
    for suite, pairs in data.suites[a.split].items():
        mm = mismatch_map(pairs)
        mappings[suite] = mm
        for cond in CONDITIONS:
            exs, meta = [], []
            for i, p in enumerate(pairs):
                if cond == "correct":
                    e = data.pair_examples(p)
                elif cond == "swap":
                    e = data.pair_examples(p, image_for=[p["images"][1], p["images"][0]])
                elif cond == "mismatched":
                    e = data.pair_examples(p, image_for=mm[i])
                else:
                    e = data.pair_examples(p, with_image=False)
                exs += e
                meta += [(i, 0), (i, 1)]
            preds, nlls, secs = [], [], []
            for s in range(0, len(exs), 8):
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = V.greedy_generate(model, fmt, exs[s:s + 8], feats, device, max_new=16)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    with torch.no_grad():
                        nll = V.per_example_nll(model, fmt, exs[s:s + 8], feats, tok.pad_token_id, device,
                                                exclude_eot=True).float().cpu().tolist()
                preds += out
                nlls += nll
                secs += [dt / len(out)] * len(out)
            for (i, k), pred, nll, sec in zip(meta, preds, nlls, secs):
                p = pairs[i]
                acc, canon = score(suite, p, k, pred)
                partner_acc, partner_canon = score(suite, p, 1 - k, pred)
                rows.append({"suite": suite, "condition": cond, "pair": i, "k": k, "kind": p["kind"],
                             "question": [p["question"], p.get("question_2", p["question"])][k],
                             "image_shown": None if cond == "no_image" else
                             (p["images"][k] if cond == "correct" else p["images"][1 - k] if cond == "swap" else mm[i][k]),
                             "reference": p["answers"][k], "partner_reference": p["answers"][1 - k],
                             "prediction": pred, "pred_norm": normalize(pred), "acc": acc, "canonical_ok": canon,
                             "matches_partner": partner_canon, "is_empty": pred.strip() == "",
                             "ref_nll": nll, "gen_seconds": sec})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, f"predictions_{a.split}.csv"), index=False)
    write_json(mappings, os.path.join(a.out, f"mismatch_map_{a.split}.json"))

    # ---------- aggregates ----------
    summ = {}
    rng = np.random.default_rng(SEED)
    for (suite, cond), d in df.groupby(["suite", "condition"]):
        pairs_df = d.groupby("pair")
        pair_acc = pairs_df.canonical_ok.all()
        ident = pairs_df.pred_norm.agg(lambda x: len(set(x)) == 1)
        summ.setdefault(suite, {})[cond] = {
            "accuracy": float(d.acc.mean()),
            "by_type": d.groupby("kind").acc.mean().to_dict(),
            "pair_accuracy": float(pair_acc.mean()), "identical_answer_rate": float(ident.mean()),
            "empty_rate": float(d.is_empty.mean()), "ref_nll": float(d.ref_nll.mean()),
            "gen_seconds_per_answer": float(d.gen_seconds.mean()),
            "follows_partner_rate": float(d.matches_partner.mean()),
            "n_questions": int(len(d)),
        }
    # correct-image advantage over swap, bootstrapping complete pairs (matched pair IDs)
    for suite in summ:
        c = df[(df.suite == suite) & (df.condition == "correct")].groupby("pair")
        for other in ("swap", "mismatched", "no_image"):
            o = df[(df.suite == suite) & (df.condition == other)].groupby("pair")
            acc_d = (c.acc.mean() - o.acc.mean()).sort_index()
            nll_d = (o.ref_nll.mean() - c.ref_nll.mean()).sort_index()
            pair_d = (c.canonical_ok.all().astype(float) - o.canonical_ok.all().astype(float)).sort_index()
            summ[suite][f"advantage_vs_{other}"] = {"accuracy": boot_ci(acc_d, rng), "nll": boot_ci(nll_d, rng),
                                                    "pair_accuracy": boot_ci(pair_d, rng)}
        for cond in CONDITIONS:
            d = df[(df.suite == suite) & (df.condition == cond)].groupby("pair")
            summ[suite][cond]["accuracy_ci"] = boot_ci(d.acc.mean(), rng)
            summ[suite][cond]["pair_accuracy_ci"] = boot_ci(d.canonical_ok.all().astype(float), rng)
    summ["peak_mem_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    write_json(summ, os.path.join(a.out, f"summary_{a.split}.json"))
    for suite in ("grounding", "vqa"):
        print(suite, {c: round(summ[suite][c]["accuracy"], 3) for c in CONDITIONS},
              "pair", {c: round(summ[suite][c]["pair_accuracy"], 3) for c in CONDITIONS})


if __name__ == "__main__":
    main()
