"""Task 2 analysis: learning curves, final-result tables, complementary-image examples, resources.

    python -m task2.analyze --run run196 outputs/task2/run196 outputs/task2/results/run196 \
                            --run run49 outputs/task2/run49 outputs/task2/results/run49 \
                            --features_info outputs/task2/features/info.json --out results/task2

Inputs are the files written by task2.train (train_log, dev_log, summary, calibration) and
task2.evaluate (predictions_test.csv, summary_test.json). Nothing is recomputed from the model.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common.utils import read_json, write_json

CONDS = ["correct", "swap", "mismatched", "no_image"]
COND_LABEL = {"correct": "Correct", "swap": "Complementary swap", "mismatched": "Mismatched", "no_image": "No image"}
SUITES = {"grounding": "Annotation-derived grounding", "vqa": "Natural VQAv2"}
STAGE_BOUNDS = (2250, 3750)
# reference categorical palette, slots 1-4 in fixed order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK = "#52514e"


def jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])


def ci(v, digits=1, scale=100):
    """'m [lo, hi]' from a [mean, lo, hi] triple."""
    m, lo, hi = (scale * x for x in v)
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------- learning curves
def learning_curves(name, train_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import plotstyle
    plotstyle.apply()
    tr, dev = jsonl(f"{train_dir}/train_log.jsonl"), jsonl(f"{train_dir}/dev_log.jsonl")
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.6))
    ax = ax.ravel()

    def bounds(a, label=False):
        for b in STAGE_BOUNDS:
            a.axvline(b, ls=":", c=plotstyle.NEUTRAL, lw=1)
        if label:
            for x, t in ((STAGE_BOUNDS[0] / 2, "Stage 1"), (sum(STAGE_BOUNDS) / 2, "Stage 2"),
                         ((STAGE_BOUNDS[1] + 4500) / 2, "Stage 3")):
                a.text(x, 0.97, t, transform=a.get_xaxis_transform(), ha="center", va="top",
                       fontsize=9.5, color=plotstyle.INK_2)

    ax[0].plot(tr["update"], tr["loss"].rolling(25, min_periods=1).mean(), c=C1, lw=1.5)
    ax[0].set(title="Training loss (25-update mean)", ylabel="mean token cross-entropy")
    ax[1].plot(dev["update"], dev["caption_nll"], c=C1, marker="o")
    ax[1].set(title="Dev caption NLL", ylabel="NLL per token")
    for a, suite in ((ax[2], "grounding"), (ax[3], "vqa")):
        for key, c, lab in (("nll_correct", C1, "correct image"), ("nll_swap", C2, "complementary swap"),
                            ("nll_noimage", C3, "no image")):
            a.plot(dev["update"], dev[suite].map(lambda d: d[key]), c=c, marker="o", label=lab)
        a.set(title=f"Dev answer NLL: {SUITES[suite]}", ylabel="NLL per answer token")
        a.legend(loc="upper right")
    for suite, c in (("grounding", C1), ("vqa", C2)):
        ax[4].plot(dev["update"], dev[suite].map(lambda d: d["swap_penalty"]), c=c, marker="o",
                   label=SUITES[suite])
    ax[4].axhline(0, c=plotstyle.NEUTRAL, lw=0.8)
    ax[4].set(title="Image-swap penalty ΔNLL (partner − correct)", ylabel="ΔNLL")
    ax[4].legend(loc="upper left")
    mix = pd.DataFrame(list(tr["mixture_counts"])).fillna(0)
    for k, c in zip(("caption", "presence", "counting", "vqa"), (C1, C2, C3, C4)):
        if k in mix:
            ax[5].plot(tr["update"], mix[k] / 1000, c=c, label=k)
    ax[5].set(title="Cumulative sampler draws", ylabel="draws (thousands)")
    ax[5].legend(loc="upper left")
    for i, a in enumerate(ax):
        bounds(a, label=(i == 0))
        a.set_xlabel("optimizer update")
    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    fig.savefig(f"{out}/learning_curves_{name}.png", dpi=200)
    plt.close(fig)
    dev.to_csv(f"{out}/dev_metrics_{name}.csv", index=False)
    return dev


# ---------------------------------------------------------------- final tables
def final_tables(name, eval_dir, out):
    S = read_json(f"{eval_dir}/summary_test.json")
    P = pd.read_csv(f"{eval_dir}/predictions_test.csv", keep_default_na=False)
    md = [f"# Final test results: {name}\n",
          "Accuracy: normalized exact match (grounding) or VQA soft accuracy (VQAv2), in %. "
          "Brackets: 95% interval from 4,000 bootstrap resamples of complete pairs. "
          "NLL: mean reference-answer token NLL (end-of-turn excluded) under each image condition, "
          "always against the original references.\n"]
    for suite, title in SUITES.items():
        if suite not in S:
            continue
        s = S[suite]
        md += [f"\n## {title} ({s['correct']['n_questions']} questions)\n",
               "| Image condition | Accuracy | By type | Pair accuracy | Identical answers | Empty | "
               "Follows partner answer | Ref. NLL | Gen. s/answer |",
               "|---|---|---|---|---|---|---|---|---|"]
        for c in CONDS:
            d = s[c]
            by = ", ".join(f"{k} {100 * v:.1f}" for k, v in sorted(d["by_type"].items()))
            md.append(f"| {COND_LABEL[c]} | {ci(d['accuracy_ci'])} | {by} | {ci(d['pair_accuracy_ci'])} | "
                      f"{100 * d['identical_answer_rate']:.1f} | {100 * d['empty_rate']:.1f} | "
                      f"{100 * d['follows_partner_rate']:.1f} | {d['ref_nll']:.3f} | {d['gen_seconds_per_answer']:.3f} |")
        md += ["", "Correct-image advantage (correct minus other condition; NLL: other minus correct, "
               "so positive means the correct image makes the reference more likely):", "",
               "| Versus | Δ accuracy (pp) | Δ NLL | Δ pair accuracy (pp) |", "|---|---|---|---|"]
        for o in ("swap", "mismatched", "no_image"):
            v = s[f"advantage_vs_{o}"]
            md.append(f"| {COND_LABEL[o]} | {ci(v['accuracy'])} | {ci(v['nll'], 3, 1)} | {ci(v['pair_accuracy'])} |")
        nk = P[P.suite == suite].pivot_table(index="kind", columns="condition", values="ref_nll", aggfunc="mean")
        nk = nk[[c for c in CONDS if c in nk]]
        md += ["", "Reference NLL by question type:", "",
               "| Type | " + " | ".join(COND_LABEL[c] for c in nk.columns) + " |",
               "|---|" + "---|" * len(nk.columns)]
        for kind, row in nk.iterrows():
            md.append(f"| {kind} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    open(f"{out}/final_results_{name}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    return S, P


# ---------------------------------------------------------------- qualitative
def outcome(pair_rows):
    """Category of a pair from its correct- and swap-condition predictions."""
    c = pair_rows[pair_rows.condition == "correct"].sort_values("k")
    w = pair_rows[pair_rows.condition == "swap"].sort_values("k")
    if len(c) < 2 or len(w) < 2:
        return None
    if c.canonical_ok.all() and w.matches_partner.all():
        return "grounded: right for both images, and the swap moves each answer to the partner's"
    if c.pred_norm.nunique() == 1:
        return "same answer for both images (image ignored)"
    if not c.canonical_ok.all():
        return "answers differ between the images, but at least one is wrong"
    return "right for both images, but the swap does not move both answers to the partner's"


def qualitative(name, P, manifests, out, per_cell=1):
    """Selection rule: within each suite and question type, sort pairs by pair index and keep the
    first `per_cell` pairs of each outcome category (no model-score ranking, no cherry-picking)."""
    urls = {}
    if manifests and os.path.exists(f"{manifests}/images.json"):
        urls = {int(k): v["url"] for k, v in read_json(f"{manifests}/images.json").items()}
    md = [f"# Complementary-image examples: {name}\n",
          f"Selection rule: for each suite and question type, pairs in pair-index order; the first {per_cell} "
          "pair(s) of each outcome category (grounded / image ignored / differ-but-wrong / right-but-swap-"
          "insensitive) are shown. References are the original answers; predictions are greedy, 16 tokens.\n"]
    rows = []
    for (suite, kind), d in P.groupby(["suite", "kind"]):
        taken = {}
        for pair, pr in d.groupby("pair"):
            cat = outcome(pr)
            if cat is None or taken.get(cat, 0) >= per_cell:
                continue
            taken[cat] = taken.get(cat, 0) + 1
            c = pr[pr.condition == "correct"].sort_values("k")
            md.append(f"\n### {SUITES[suite]} / {kind}: pair {pair} — {cat}\n")
            md.append(f"Question: {c.question.iloc[0]}\n")
            md.append("| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |")
            md.append("|---|---|---|---|---|---|---|")
            for k in (0, 1):
                ck = c[c.k == k].iloc[0]
                img = int(float(ck.image_shown)) if str(ck.image_shown) not in ("", "nan") else None
                cell = f'<img src="{urls[img]}" height="120">' if img in urls else str(img)
                pred = {cond: pr[(pr.condition == cond) & (pr.k == k)].prediction.iloc[0] for cond in CONDS}
                md.append(f"| image {k + 1} | {cell} | {ck.reference} | " +
                          " | ".join(f"`{pred[cond]}`" for cond in CONDS) + " |")
                rows.append({"suite": suite, "kind": kind, "pair": pair, "k": k, "category": cat,
                             "image": img, "reference": ck.reference, **{f"pred_{x}": pred[x] for x in CONDS}})
    open(f"{out}/qualitative_{name}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    cats = (P.groupby(["suite", "kind", "pair"]).apply(outcome, include_groups=False).rename("category")
            .reset_index().groupby(["suite", "kind", "category"]).size().rename("pairs").reset_index())
    cats.to_csv(f"{out}/pair_outcomes_{name}.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{out}/qualitative_{name}.csv", index=False)
    return cats


# ---------------------------------------------------------------- resources
def resources(name, train_dir, S, features_info):
    s = read_json(f"{train_dir}/summary.json")
    cal = read_json(f"{train_dir}/calibration.json")
    rows = []
    for st, d in s["stages"].items():
        rows.append({"run": name, "stage": st, "trainable_params": d.get("trainable_params"),
                     "connector_params": d.get("connector_params"), "lora_params": d.get("lora_params"),
                     "train_minutes": d.get("seconds", np.nan) / 60, "peak_mem_GB": d.get("peak_mem_bytes", np.nan) / 1e9,
                     "draws_at_stage_end": json.dumps(d.get("mixture_counts_end", {}))})
    gen = np.mean([S[su][c]["gen_seconds_per_answer"] for su in SUITES if su in S for c in CONDS])
    extra = {"run": name, "train_hours_total": s.get("train_seconds", np.nan) / 3600,
             "skipped_updates": len(s.get("skipped_updates", [])), "eval_peak_mem_GB": (S.get("peak_mem_bytes") or np.nan) / 1e9,
             "gen_seconds_per_answer_mean": gen, "connector_scale": cal["scale"], "scale_rule": cal["rule"]}
    if features_info and os.path.exists(features_info):
        f = read_json(features_info)
        extra.update(feature_download_min=f["download_seconds"] / 60, feature_encode_min=f["encode_seconds"] / 60,
                     feature_peak_mem_GB=(f.get("peak_mem_bytes") or np.nan) / 1e9, feature_cache_GB=f["bytes"] / 1e9,
                     n_images=f["n_images"])
    return pd.DataFrame(rows), extra


def run_config():
    """The training/evaluation configuration, read from the code that produced the runs."""
    from task2 import data as TD, train as TT, vlm as TV
    from task2.evaluate import CONDITIONS, N_BOOT
    return {"seed": 6304, "models": {"vision": "google/siglip-base-patch16-224", "language": TV.QWEN,
                                      "language_revision": TV.QWEN_REV},
            "connector": "Linear 768->1024, GELU, Linear 1024->1024; output scaled by calibration.json",
            "visual_tokens": {"main": TV.N_VIS, "optional": 49}, "caps": TV.CAPS,
            "prompts": {"caption": TV.CAPTION_PROMPT, "short_answer_suffix": TV.SHORT_SUFFIX},
            "micro_batch": TT.MICRO, "grad_accum": TT.ACCUM, "effective_batch": TT.EFF,
            "optimizer": "AdamW betas (0.9, 0.999) eps 1e-8, weight decay 0.01 (not on biases); rebuilt per stage",
            "schedule": "per stage: 5% warmup, cosine decay to 10% of peak", "grad_clip": 1.0,
            "precision": "FP16 autocast, dynamic loss scaling, frozen FP16 backbone, FP32 trainable params",
            "gradient_checkpointing": True, "stages": TT.STAGES,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "dev_every": TT.DEV_EVERY, "checkpoint_every": TT.CKPT_EVERY,
            "final_eval": {"conditions": list(CONDITIONS), "decoding": "greedy", "max_new_tokens": 16,
                           "batch_size": 8, "bootstrap_resamples": N_BOOT},
            "data": {"caption_train_images": TD.N_CAPTION_TRAIN, "captions_per_train_image": TD.CAPS_PER_TRAIN_IMAGE,
                     "caption_dev": TD.N_CAPTION_DEV, "grounding_dev": TD.GROUND_DEV, "grounding_test": TD.GROUND_TEST,
                     "vqa_train": TD.VQA_TRAIN, "vqa_dev": TD.VQA_DEV, "vqa_test": TD.VQA_TEST,
                     "vqa_min_agreement": TD.MIN_AGREE, "presence_min_area": TD.PRESENCE_MIN_AREA,
                     "count_min_area": TD.COUNT_MIN_AREA, "count_range": TD.COUNT_RANGE}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append", metavar=("NAME", "TRAIN_DIR", "EVAL_DIR"), required=True)
    ap.add_argument("--manifests", default="manifests/task2")
    ap.add_argument("--features_info", default=None)
    ap.add_argument("--out", default="results/task2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stage_rows, extras, compare = [], [], []
    for name, train_dir, eval_dir in a.run:
        dev = learning_curves(name, train_dir, a.out)
        S, P = final_tables(name, eval_dir, a.out)
        cats = qualitative(name, P, a.manifests, a.out)
        R, extra = resources(name, train_dir, S, a.features_info)
        stage_rows.append(R)
        extras.append(extra)
        row = {"run": name, "train_hours": extra["train_hours_total"],
               "peak_train_mem_GB": R.peak_mem_GB.max(), "gen_s_per_answer": extra["gen_seconds_per_answer_mean"]}
        for su in SUITES:
            if su in S:
                row[f"{su}_acc_correct"] = 100 * S[su]["correct"]["accuracy"]
                row[f"{su}_pair_acc_correct"] = 100 * S[su]["correct"]["pair_accuracy"]
                row[f"{su}_swap_adv_acc_pp"] = 100 * S[su]["advantage_vs_swap"]["accuracy"][0]
                row[f"{su}_swap_adv_nll"] = S[su]["advantage_vs_swap"]["nll"][0]
        row["final_dev"] = dev.iloc[-1][["caption_nll"]].to_dict()
        compare.append(row)
        print(name, "\n", cats.to_string())
    write_json(run_config(), f"{a.out}/config.json")
    pd.concat(stage_rows).to_csv(f"{a.out}/resources_by_stage.csv", index=False)
    write_json(extras, f"{a.out}/resources_summary.json")
    pd.DataFrame(compare).to_csv(f"{a.out}/run_comparison.csv", index=False)
    print(pd.concat(stage_rows).round(3).to_string())
    print(pd.DataFrame(compare).round(3).to_string())


if __name__ == "__main__":
    main()
