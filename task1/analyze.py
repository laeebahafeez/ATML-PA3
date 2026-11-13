"""Task 1 analysis: benchmark table, GSM8K extraction/truncation, learning curves,
L2-SP penalty, resources, qualitative sheets.

    python -m task1.analyze --runs outputs/task1 --eval outputs/results_task1 --out results/task1

All inputs are files written by task1.train / task1.evaluate (lm-eval --log_samples).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from common.utils import read_json, write_json

CONDS = ["baseline", "full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
RUNS = ["full", "l2sp", "lora4", "lora16", "oplora16"]
LABEL = {"baseline": "Baseline", "full_u625": "Full FT (mid)", "full": "Full FT", "l2sp": "Full FT + L2-SP",
         "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
METRICS = [("gsm8k_cot", "exact_match,flexible-extract", "GSM8K (flex EM)"),
           ("gsm8k_cot", "exact_match,strict-match", "GSM8K (strict EM)"),
           ("hellaswag", "acc_norm,none", "HellaSwag acc_norm"), ("hellaswag", "acc,none", "HellaSwag acc"),
           ("boolq", "acc,none", "BoolQ acc"),
           ("arc_easy", "acc_norm,none", "ARC-E acc_norm"), ("arc_easy", "acc,none", "ARC-E acc")]
GEN_CAP = 1024


def latest(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None


def load_results(eval_dir, cond):
    res = {}
    for sub in ("gsm8k", "likelihood"):
        f = latest(os.path.join(eval_dir, cond, sub, "**", "results_*.json"))
        if f:
            for task, vals in json.load(open(f))["results"].items():
                res.setdefault(task, {}).update(vals)
    return res


def gsm8k_samples(eval_dir, cond, tok):
    f = latest(os.path.join(eval_dir, cond, "gsm8k", "**", "samples_gsm8k_cot_*.jsonl"))
    if not f:
        return None
    rows = {}
    for line in open(f, encoding="utf-8"):
        s = json.loads(line)
        if s.get("filter") != "flexible-extract":
            continue
        resp = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        ext = s["filtered_resps"][0] if isinstance(s["filtered_resps"], list) else s["filtered_resps"]
        n = len(tok(resp, add_special_tokens=False)["input_ids"]) if tok else len(resp.split())
        # no number extracted: "[invalid]", or a digitless match such as the LaTeX "$$" delimiter
        rows[s["doc_id"]] = {"doc_id": s["doc_id"], "correct": float(s["exact_match"]), "extracted": ext,
                             "unextractable": not any(ch.isdigit() for ch in str(ext)), "n_tokens": n,
                             "truncated": n >= GEN_CAP - 1, "target": s["target"], "response": resp}
    return pd.DataFrame(rows.values()).sort_values("doc_id")


def benchmark_table(eval_dir, out, tok):
    rows, gsm = [], {}
    for c in CONDS:
        r = load_results(eval_dir, c)
        if not r:
            continue
        row = {"condition": c, "label": LABEL[c]}
        for task, key, name in METRICS:
            row[name] = 100 * r.get(task, {}).get(key, np.nan)
            se = r.get(task, {}).get(key.replace(",", "_stderr,"), np.nan)
            row[name + " se"] = 100 * se if isinstance(se, float) else np.nan
        g = gsm8k_samples(eval_dir, c, tok)
        if g is not None:
            gsm[c] = g
            g.to_csv(os.path.join(out, f"gsm8k_per_example_{c}.csv"), index=False)
            row.update({"GSM8K unextractable %": 100 * g.unextractable.mean(),
                        "GSM8K truncated %": 100 * g.truncated.mean(),
                        "GSM8K mean resp tokens": g.n_tokens.mean(),
                        "GSM8K acc | not truncated": 100 * g[~g.truncated].correct.mean()})
        rows.append(row)
    T = pd.DataFrame(rows)
    if "baseline" in set(T.condition):
        base = T[T.condition == "baseline"].iloc[0]
        for _, _, name in METRICS:
            T[name + " Δpp"] = T[name] - base[name]
    T.to_csv(os.path.join(out, "benchmarks.csv"), index=False)
    # compact markdown table: primary metrics with deltas, raw accuracy beside normalized
    prim = ["GSM8K (flex EM)", "HellaSwag acc_norm", "HellaSwag acc", "BoolQ acc", "ARC-E acc_norm", "ARC-E acc"]
    lines = ["| Condition | " + " | ".join(prim) + " | GSM8K unextr. % | GSM8K trunc. % |",
             "|---|" + "---|" * (len(prim) + 2)]
    for _, r in T.iterrows():
        cells = [f"{r[p]:.1f} ({r.get(p + ' Δpp', 0):+.1f})" for p in prim]
        lines.append(f"| {r.label} | " + " | ".join(cells) +
                     f" | {r.get('GSM8K unextractable %', np.nan):.1f} | {r.get('GSM8K truncated %', np.nan):.1f} |")
    open(os.path.join(out, "benchmarks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # GSM8K transitions vs baseline (which items flip)
    if "baseline" in gsm:
        b = gsm["baseline"].set_index("doc_id")
        tr = {}
        for c, g in gsm.items():
            g = g.set_index("doc_id")
            tr[c] = {"gained": int(((b.correct == 0) & (g.correct == 1)).sum()),
                     "lost": int(((b.correct == 1) & (g.correct == 0)).sum())}
        write_json(tr, os.path.join(out, "gsm8k_flips_vs_baseline.json"))
    print("\n".join(lines))
    return T


def learning_curves(runs_dir, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import plotstyle
    plotstyle.apply()
    P = plotstyle.PALETTE
    color = {r: P[i] for i, r in enumerate(RUNS)}  # fixed colour per run, in run order
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    logs = {}
    # Full FT and L2-SP losses nearly coincide, so L2-SP is dashed to keep Full FT visible
    style = {"l2sp": (0, (4, 2))}
    for r in RUNS:
        f = os.path.join(runs_dir, r, "train_log.jsonl")
        if not os.path.exists(f):
            continue
        d = pd.DataFrame([json.loads(l) for l in open(f)])
        logs[r] = d
        ax[0].plot(d["update"], d.sft_loss.rolling(25, min_periods=1).mean(), label=LABEL[r],
                   ls=style.get(r, "-"), c=color[r], lw=1.8)
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (25-update mean)", title="Training SFT loss",
              ylim=(0.15, 0.5))
    ax[0].axvline(625, ls=":", c=plotstyle.NEUTRAL, lw=1)
    ax[0].text(640, 0.158, "midpoint", fontsize=9, color=plotstyle.INK_2, va="bottom")
    ax[0].legend(ncol=2, loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    # learning rate: the five schedules collapse onto two curves (full-model vs adapter peak)
    groups = [("Full FT, L2-SP (peak 2e-5)", ["full", "l2sp"], P[0]),
              ("LoRA-4, LoRA-16, OPLoRA (peak 2e-4)", ["lora4", "lora16", "oplora16"], P[2])]
    for lab, members, c in groups:
        m = next((x for x in members if x in logs), None)
        if m is not None:
            ax[2].plot(logs[m]["update"], logs[m].lr, c=c, label=lab)
    # the schedule starts at 0 and decays to ~1e-10, which would stretch a log axis
    ax[2].set(xlabel="optimizer update", ylabel="learning rate (log scale)", yscale="log", ylim=(1e-7, 1e-3),
              title="Learning rate (38 warm-up updates, cosine)")
    ax[2].legend(loc="upper right")
    if "l2sp" in logs:
        d = logs["l2sp"]
        ax[1].plot(d["update"], d.sft_loss.rolling(25, min_periods=1).mean(), c=color["l2sp"], label="SFT loss")
        ax[1].plot(d["update"], d.total_loss.rolling(25, min_periods=1).mean(), c=plotstyle.INK, lw=1.2,
                   ls=(0, (4, 2)), label="total = SFT + λΩ")
        ax[1].plot(d["update"], d.lam_omega.rolling(25, min_periods=1).mean(), c=P[6], label="λΩ")
        ax2 = ax[1].twinx()
        ax2.plot(d["update"], d.omega, c=plotstyle.NEUTRAL, alpha=0.55, lw=1.6, label="Ω (right axis)")
        ax2.set_ylabel("Ω = ½‖θ−θ₀‖²")
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax[1].set(xlabel="optimizer update", ylabel="loss", title="L2-SP objective (λ = 1e-3)", ylim=(-0.02, 0.6))
        h1, l1 = ax[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[1].legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(os.path.join(out, "learning_curves.png"), dpi=200)
    lr = {r: {"final_sft_loss_last50": float(d.sft_loss.tail(50).mean()),
              "skipped_updates": int(d.skipped.sum())} for r, d in logs.items()}
    if "l2sp" in logs:
        d = logs["l2sp"]
        lr["l2sp"].update(final_omega=float(d.omega.iloc[-1]), final_lam_omega=float(d.lam_omega.iloc[-1]),
                          mean_penalty_seconds=float(d.penalty_seconds.mean()))
    write_json(lr, os.path.join(out, "learning_summary.json"))


def resources(runs_dir, out):
    rows = []
    for r in RUNS:
        cfg_f, sum_f = os.path.join(runs_dir, r, "config.json"), os.path.join(runs_dir, r, "summary.json")
        if not os.path.exists(sum_f):
            continue
        cfg, s = read_json(cfg_f), read_json(sum_f)
        exp = s["exports"].get("1250", {})
        rows.append({"run": LABEL[r], "trainable_params": s["n_trainable"],
                     "train_hours": s["train_seconds"] / 3600,
                     "peak_alloc_GB": s["peak_mem_bytes"] / 1e9,
                     "peak_reserved_GB": s.get("peak_mem_reserved_bytes", 0) / 1e9,
                     "resume_ckpt_GB": max(s["checkpoint_bytes"].values()) / 1e9 if s["checkpoint_bytes"] else np.nan,
                     "final_export_MB": exp.get("export_bytes", np.nan) / 1e6,
                     "displacement_final": exp.get("displacement_l2"),
                     "displacement_mid": s["exports"].get("625", {}).get("displacement_l2"),
                     "svd_seconds": cfg.get("svd_seconds"), "basis_MB": (cfg.get("basis_bytes") or 0) / 1e6,
                     "reference_MB": (cfg.get("reference_bytes") or 0) / 1e6,
                     "export_seconds": exp.get("export_seconds"),
                     "oplora_leak": exp.get("oplora_constraint_check")})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, "resources.csv"), index=False)
    print(R.round(3).to_string())
    return R


# ---------------------------------------------------------------- qualitative
MATH_PAT = re.compile(r"\\boxed|\\frac|\$[^$]+\$|\bstep \d|\bequation\b|=\s*\d|\bcalculate\b|\bsolve\b", re.I)


def repetition(text, n=4):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def format_ok(item):
    r = item["response"].strip()
    if item["format"] == "json":
        s = re.sub(r"^```(?:json)?|```$", "", r.strip()).strip()
        try:
            json.loads(s)
            return True
        except Exception:
            return False
    if item["format"] == "template":
        keys = re.findall(r"^(\w+) = #", item["messages"][-1]["content"], re.M)
        return all(re.search(rf"^\s*{k}\s*=", r, re.M) for k in keys)
    return None


def qualitative(eval_dir, out):
    allq = {}
    for c in CONDS:
        f = os.path.join(eval_dir, c, "qualitative.json")
        if os.path.exists(f):
            allq[c] = read_json(f)["items"]
    if not allq:
        return
    rows = []
    for c, items in allq.items():
        for it in items:
            rows.append({"condition": c, "id": it["id"], "question": it["question"], "format": it["format"],
                         "n_tokens": it["n_new_tokens"], "truncated": it["truncated"],
                         "format_ok_auto": format_ok(it), "math_framing_auto": bool(MATH_PAT.search(it["response"])),
                         "repetition_4gram": round(repetition(it["response"]), 3), "response": it["response"]})
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(out, "qualitative_all_168.csv"), index=False)
    # 42-response annotation subset: first and last plain-order questions, all formats and conditions
    order = list(dict.fromkeys(it["question"] for it in next(iter(allq.values()))))
    subset = Q[Q.question.isin([order[0], order[-1]])].copy()
    for col in ("format_compliance", "coherence_relevance", "repetition", "unnecessary_math", "inappropriate", "notes"):
        subset[col] = ""
    subset.sort_values(["question", "format", "condition"]).to_csv(os.path.join(out, "annotation_subset_42.csv"), index=False)
    # readable grouped markdown of all 168
    md = []
    for q in order:
        for fmt in ("plain", "json", "template"):
            md.append(f"\n## {q} — {fmt}\n")
            for c in CONDS:
                r = Q[(Q.question == q) & (Q.format == fmt) & (Q.condition == c)]
                if len(r):
                    r = r.iloc[0]
                    flag = " **[TRUNCATED]**" if r.truncated else ""
                    md.append(f"**{LABEL[c]}**{flag} ({r.n_tokens} tok)\n\n```\n{r.response}\n```\n")
    open(os.path.join(out, "qualitative_grouped.md"), "w", encoding="utf-8").write("\n".join(md))
    auto = Q.groupby("condition")[["truncated", "math_framing_auto", "repetition_4gram", "n_tokens"]].mean()
    auto["format_ok_json"] = Q[Q.format == "json"].groupby("condition").format_ok_auto.mean()
    auto["format_ok_template"] = Q[Q.format == "template"].groupby("condition").format_ok_auto.mean()
    auto.to_csv(os.path.join(out, "qualitative_auto_flags.csv"))
    print(auto.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/task1")
    ap.add_argument("--eval", default="outputs/results_task1")
    ap.add_argument("--out", default="results/task1")
    ap.add_argument("--no_tokenizer", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = None
    if not a.no_tokenizer:
        from task1.data import load_tokenizer
        tok = load_tokenizer()
    if os.path.isdir(a.eval):
        benchmark_table(a.eval, a.out, tok)
        qualitative(a.eval, a.out)
    learning_curves(a.runs, a.out)
    resources(a.runs, a.out)


if __name__ == "__main__":
    main()
