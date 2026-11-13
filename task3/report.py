"""Task 3 figures, tables and rule-selected examples from the saved results (Steps 1-6).

    python -m task3.report          # needs results/task3/* and outputs/task3/{math,mathvista}.pkl

Writes results/task3/figures/*.png, results/task3/report_tables.md and
results/task3/examples.md. Gold answers and later candidates are used here only for
retrospective analysis, never for selection or stopping.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

from common.utils import read_json
from task3 import methods as M

R = "results/task3"
FIG = f"{R}/figures"
BUDGETS = (1, 2, 4, 8, 16, 32)
NAMES = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
RWC = {"math": "selfcert_rwc", "mathvista": "visconf_rwc"}
LABEL = {"first_ans": "FirstAns", "self_consistency": "Self-consistency", "best_logp": "Highest mean log-prob",
         "best_selfcert": "Highest self-certainty", "selfcert_rwc": "Self-certainty + RWC",
         "shortest": "Shortest rollout", "best_visconf": "Highest VisConf", "visconf_rwc": "VisConf + RWC",
         "best_vision": "Highest vision", "vision_rwc": "Vision + RWC",
         "best_visconf_token": "Highest token-fused VisConf", "visconf_token_rwc": "Token-fused VisConf + RWC"}
# reference categorical palette in fixed slot order; a method keeps its colour in every figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PLOTTED = ["self_consistency", "selfcert_rwc", "best_selfcert", "best_logp", "first_ans", "shortest",
           "visconf_rwc", "best_visconf"]
COLOR = dict(zip(PLOTTED, PALETTE))
INK = "#5f5f5a"  # neutral reference ink (oracle line, adaptive markers)


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import plotstyle
    plotstyle.apply()
    return plt


def load_pkl(name):
    with open(f"outputs/task3/{name}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- figures
def fig_curves(summ):
    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, name in zip(ax, NAMES):
        e = summ[name][summ[name].split == "eval"]
        for m in PLOTTED:
            d = e[e.method == m]
            if len(d):
                a.plot(d.N, 100 * d.acc, c=COLOR[m], marker="o", ms=4, lw=1.8, label=LABEL[m])
        o = e[e.method == "self_consistency"]
        a.plot(o.N, 100 * o.oracle, c=INK, ls=(0, (4, 2)), lw=1.8, label="Oracle coverage")
        a.set(xscale="log", xticks=BUDGETS, xticklabels=[str(b) for b in BUDGETS], xlabel="candidates N",
              ylabel="% of held-out questions", title=NAMES[name])
        a.minorticks_off()
    # one shared legend below the panels (MathVista has every series), so no line is covered
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=3)
    fig.savefig(f"{FIG}/accuracy_coverage.png", dpi=200)
    plt.close(fig)


def fig_cost(summ, adaptive):
    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, name in zip(ax, NAMES):
        e = summ[name][summ[name].split == "eval"]
        for m in ("self_consistency", RWC[name]):
            d = e[e.method == m]
            a.plot(d.cost, 100 * d.acc, c=COLOR[m], marker="o", ms=4, label=f"{LABEL[m]} (fixed N)")
        f = e[(e.method == "first_ans") & (e.N == 1)]
        a.scatter(f.cost, 100 * f.acc, c=COLOR["first_ans"], marker="D", s=40, zorder=3, label="FirstAns")
        ev = adaptive[name]["eval"]
        for fam, mk in (("agreement", "^"), ("quality", "s")):
            sel = adaptive[name]["selected"][fam]
            lab = f"Adaptive {fam} (τ={sel['tau']}" + (f", γ={sel['gamma']})" if fam == "quality" else ")")
            a.scatter(ev[fam]["cost"], 100 * ev[fam]["acc"], c=INK, marker=mk, s=60, zorder=4, label=lab)
        a.set(xscale="log", xlabel="mean generated tokens per question (offline cost proxy)",
              ylabel="held-out accuracy (%)", title=NAMES[name])
        a.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/accuracy_vs_cost.png", dpi=200)
    plt.close(fig)


def fig_decomposition(dec, summ):
    plt = plt_setup()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    parts = (("correct", "correct", PALETTE[0]), ("selection_failure", "correct candidate available but missed",
                                                    PALETTE[1]), ("sampling_failure", "no correct candidate", PALETTE[2]))
    for a, name in zip(ax, NAMES):
        x = np.arange(len(BUDGETS))
        for j, m in enumerate(("self_consistency", RWC[name])):
            d = dec[name][(dec[name].split == "eval") & (dec[name].method == m)].set_index("N").loc[list(BUDGETS)]
            bottom = np.zeros(len(BUDGETS))
            for col, lab, c in parts:
                a.bar(x + (j - 0.5) * 0.4, 100 * d[col], 0.36, bottom=bottom, color=c, edgecolor="white",
                      linewidth=1, label=lab if j == 0 else None)
                bottom += 100 * d[col].to_numpy()
        a.set_xticks(x, [f"{b}\nSC | RWC" for b in BUDGETS], fontsize=8)
        a.set(xlabel="candidates N (left bar: self-consistency, right bar: " + LABEL[RWC[name]] + ")",
              ylabel="% of held-out questions", title=NAMES[name])
        a.grid(False)
        g = summ[name][(summ[name].split == "eval") & (summ[name].method == "self_consistency")]
        for xi, n in zip(x, g.n_groups):
            a.text(xi, 101, f"{n:.1f} groups", ha="center", fontsize=8.5, color=INK)
        a.set_ylim(0, 108)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    fig.savefig(f"{FIG}/failure_decomposition.png", dpi=200)
    plt.close(fig)


def trajectory_examples(D):
    """Rule: the first two held-out MathVista questions (split order) with mixed correctness;
    in each, the earliest-id correct and the earliest-id incorrect rollout."""
    out = []
    for qid in D["split"]["eval"]:
        q = D["questions"][qid]
        if 0 < q["correct"].mean() < 1:
            out.append((qid, int(np.flatnonzero(q["correct"])[0]), int(np.flatnonzero(~q["correct"])[0])))
        if len(out) == 2:
            return out
    return out


def fig_trajectories(D):
    plt = plt_setup()
    ex = trajectory_examples(D)
    # c_t (0-1) and v_t (a few %) differ in scale, so each gets its own panel on a shared token axis
    fig, ax = plt.subplots(2 * len(ex), 2, figsize=(13, 2.6 * 2 * len(ex)), squeeze=False)
    idx = np.arange(32)
    rows = []
    for r, (qid, ok, bad) in enumerate(ex):
        q = D["questions"][qid]
        c = M.calibrated_conf(q, idx)          # calibrated on all 32 candidates (N = 32)
        for col, rid in enumerate((ok, bad)):
            top, bot = ax[2 * r][col], ax[2 * r + 1][col]
            ct, vt = pd.Series(c[rid]), pd.Series(q["v_tokens"][rid])
            top.plot(ct.rolling(15, min_periods=1).mean(), c=PALETTE[0], lw=1.4)
            bot.plot(vt.rolling(15, min_periods=1).mean(), c=PALETTE[1], lw=1.4)
            rho = np.corrcoef(ct, vt)[0, 1] if ct.std() > 0 and vt.std() > 0 else np.nan
            zt, z = (ct * vt).mean(), ct.mean() * vt.mean()
            top.set(ylim=(0, 1), ylabel="calibrated\nconfidence c_t",
                    title=f"{qid} rollout {rid} ({'correct' if q['correct'][rid] else 'incorrect'}): "
                          f"corr(c, v) {rho:.2f}, Z {z:.3f}, Z_token {zt:.3f}")
            bot.set(ylim=(0, None), ylabel="visual\nengagement v_t", xlabel="generated token (15-token mean)")
            for a in (top, bot):              # each rollout on its own length
                a.set_xlim(0, max(len(ct) - 1, 1))
            rows.append({"sample_id": qid, "rollout": rid, "correct": bool(q["correct"][rid]), "tokens": len(ct),
                         "corr_c_v": rho, "Z": z, "Z_token": zt, "C": ct.mean(), "V": vt.mean()})
    fig.tight_layout()
    fig.savefig(f"{FIG}/mathvista_trajectories.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables
def pct(x):
    return f"{100 * x:.1f}"


def boot_cell(b):
    return f"{100 * b['diff']:+.1f} [{100 * b['lo']:+.1f}, {100 * b['hi']:+.1f}]"


def stopping_vs_pool(name, D, adaptive):
    """Retrospective: stopping and accuracy by the fraction of correct candidates in the full pool."""
    rows = []
    for fam in ("agreement", "quality"):
        df = pd.read_csv(f"{R}/adaptive_{name}_{fam}.csv", dtype={"sample_id": str})
        frac = df.sample_id.map(lambda s: D["questions"][s]["correct"].mean())
        top = df.sample_id.map(lambda s: np.bincount(D["questions"][s]["gid"]).max() / 32)
        df["pool"] = pd.cut(frac, [-0.01, 0.0, 0.5, 0.999, 1.0], labels=["0", "(0, 0.5]", "(0.5, 1)", "1"])
        df["agree"] = pd.cut(top, [0, 0.5, 0.75, 1.0], labels=["≤ 0.5", "(0.5, 0.75]", "(0.75, 1]"])
        for by in ("pool", "agree"):
            g = df.groupby(by, observed=True).agg(runs=("stop_n", "size"), mean_rollouts=("stop_n", "mean"),
                                                  stop_at_4=("stop_n", lambda x: (x == 4).mean()),
                                                  acc=("correct", "mean"), fixed32_acc=("fixed32_correct", "mean"))
            for k, v in g.iterrows():
                rows.append({"policy": fam, "by": by, "bin": k, **v.to_dict()})
    return pd.DataFrame(rows)


def tables(summ, dec, boot, sig, adaptive, live, pools, traj):
    md = ["# Task 3 tables (held-out split unless stated)\n",
          "Accuracy, oracle coverage and token cost are averaged over the 10 saved orders within a question, "
          "then over questions. Bootstrap intervals: 1,000 paired question-level resamples.\n"]
    for name in NAMES:
        e = summ[name][summ[name].split == "eval"]
        piv = e.pivot(index="method", columns="N", values="acc")
        md += [f"\n## {NAMES[name]}: accuracy (%) by candidate budget\n",
               "| Method | " + " | ".join(f"N={b}" for b in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
        for m in piv.index:
            md.append(f"| {LABEL.get(m, m)} | " + " | ".join(pct(piv.loc[m, b]) for b in BUDGETS) + " |")
        sc = e[e.method == "self_consistency"].set_index("N")
        md.append("| *Oracle coverage* | " + " | ".join(pct(sc.loc[b, "oracle"]) for b in BUDGETS) + " |")
        md.append("| *Distinct answer groups* | " + " | ".join(f"{sc.loc[b, 'n_groups']:.2f}" for b in BUDGETS) + " |")
        md.append("| *Tokens (all candidates)* | " + " | ".join(f"{sc.loc[b, 'cost']:.0f}" for b in BUDGETS) + " |")
        md += ["", "Paired differences in accuracy (pp, 95% CI):", "", "| Comparison | N=8 | N=32 |", "|---|---|---|"]
        for k in sorted({k.split(":", 1)[1] for k in boot[name]}):
            a, b = k.split("-")
            md.append(f"| {LABEL.get(a, a)} − {LABEL.get(b, b)} | {boot_cell(boot[name][f'N8:{k}'])} | "
                      f"{boot_cell(boot[name][f'N32:{k}'])} |")
        d = dec[name][(dec[name].split == "eval") & (dec[name].N == 32)].set_index("method")
        md += ["", "Failure decomposition at N=32 (% of question-orders):", "",
               "| Method | Correct | Selection failure | Sampling failure |", "|---|---|---|---|"]
        for m in d.index:
            md.append(f"| {LABEL.get(m, m)} | {pct(d.loc[m, 'correct'])} | {pct(d.loc[m, 'selection_failure'])} | "
                      f"{pct(d.loc[m, 'sampling_failure'])} |")
        s = sig[name]
        md += ["", "Signals at N=32 (within-question; 'mixed' = questions with some correct and some incorrect "
               "candidates):", "", "| Signal | Top−bottom quartile correct rate (all) | (mixed) | "
               "Spearman with correctness (mixed) | Spearman with length (mixed) |", "|---|---|---|---|---|"]
        for k, v in s.items():
            if "top_minus_bottom_quartile_all" in v:
                md.append(f"| {k} | {100 * v['top_minus_bottom_quartile_all']:+.1f} | "
                          f"{100 * v['top_minus_bottom_quartile_mixed']:+.1f} | "
                          f"{v['mean_spearman_with_correct_mixed']:+.3f} | {v['mean_spearman_with_length_mixed']:+.3f} |")
        p = s["pool_stats"]
        md.append(f"\nPool: mean candidate accuracy {pct(p['mean_rollout_acc'])}%, all-correct {pct(p['frac_all_correct'])}%, "
                  f"none-correct {pct(p['frac_none_correct'])}%, mean length correct {p['mean_len_correct']:.0f} vs "
                  f"incorrect {p['mean_len_incorrect']:.0f} tokens, capped {pct(p['capped_rate'])}%, "
                  f"NO_ANSWER {pct(p['no_answer_rate'])}%.")
        if "confidence_vision_corr" in s:
            c = s["confidence_vision_corr"]
            md.append(f"\nConfidence–vision correlation within rollouts: mean over questions {c['mean_over_questions']:+.3f} "
                      f"(median {c['median']:+.3f}, IQR {c['q25']:+.3f} to {c['q75']:+.3f}; "
                      f"{pct(c['frac_questions_positive'])}% of questions positive). Mean Z_token − Z = "
                      f"{c['mean_Ztoken_minus_Z']:+.5f} (= mean within-rollout covariance; identity error "
                      f"{c['max_cov_identity_error']:.1e}).")
        a = adaptive[name]
        md += ["", f"Adaptive stopping (quality signal: {a['quality_signal']}); settings chosen on the dev split:", "",
               "| Policy | τ | γ | Dev acc | Held-out acc | Δ vs fixed-32 (pp, 95% CI) | Mean tokens | Mean rollouts | "
               "Stops at 4 / 8 / 16 / 32 (%) | Wrong early stops (fixable by fixed-32) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in ("agreement", "quality", "fixed32"):
            ev = a["eval"][fam]
            fixed_dev = next(r for r in a["dev_grid"]["agreement"] if r["policy"] == "fixed32")
            sel = a["selected"].get(fam, {"tau": None, "gamma": None, "dev": fixed_dev})
            dist = " / ".join(pct(ev["stop_distribution"].get(str(n), ev["stop_distribution"].get(n, 0)))
                              for n in (4, 8, 16, 32))
            dci = ev.get("acc_diff_vs_fixed32_ci")
            dcell = f"{100 * dci[0]:+.2f} [{100 * dci[1]:+.2f}, {100 * dci[2]:+.2f}]" if dci else "—"
            md.append(f"| {fam} | {sel['tau'] if sel['tau'] is not None else '—'} | "
                      f"{sel['gamma'] if sel['gamma'] is not None else '—'} | {pct(sel['dev']['acc'])} | {pct(ev['acc'])} | "
                      f"{dcell} | {ev['cost']:.0f} | {ev['rollouts']:.2f} | {dist} | "
                      f"{ev['early_incorrect']} ({ev['early_incorrect_recoverable_by_fixed32']}) |")
        sc = e[e.method == "self_consistency"]
        md.append("\nFixed-N self-consistency at comparable cost: " + ", ".join(
            f"N={int(r.N)}: {pct(r.acc)}% at {r.cost:.0f} tokens" for r in sc.itertuples() if r.N in (4, 8, 16)) + ".")
        pt = pools[name]
        md += ["", "Stopping by full-pool composition (retrospective; question-orders):", "",
               "| Policy | Grouped by | Bin | Runs | Mean rollouts | Stop at 4 | Accuracy | Fixed-32 accuracy |",
               "|---|---|---|---|---|---|---|---|"]
        for r in pt.itertuples():
            by = "fraction of correct candidates" if r.by == "pool" else "largest answer-group share"
            md.append(f"| {r.policy} | {by} | {r.bin} | {int(r.runs)} | {r.mean_rollouts:.1f} | {pct(r.stop_at_4)} | "
                      f"{pct(r.acc)} | {pct(r.fixed32_acc)} |")
    if live:
        md += ["\n## Live MATH strategies (50 held-out questions, 3,072-token output allowance)\n",
               "| Strategy | Accuracy | Mean input tokens | Mean output tokens | Mean seconds | Truncated (any call) | "
               "Answer extracted |", "|---|---|---|---|---|---|---|"]
        for r in live["summary"]:
            md.append(f"| {r['strategy']} | {pct(r['accuracy'])} | {r['mean_input_tokens']:.0f} | "
                      f"{r['mean_output_tokens']:.0f} | {r['mean_seconds']:.1f} | {pct(r['truncation_rate'])} | "
                      f"{pct(r['extraction_rate'])} |")
        md += ["", "Per call: " + "; ".join(f"{c['kind']}: truncated {pct(c['call_truncation'])}%, extracted "
                                             f"{pct(c['call_extraction'])}%, {c['call_output_tokens']:.0f} output tokens"
                                             for c in live["per_call"]) + ".",
               f"\nRefinement transitions vs the shared initial answer: {live['transitions']}; answer changed in "
               f"{pct(live['refine_answer_changed_rate'])}%; initial accuracy {pct(live['initial_accuracy'])}%.",
               "\nPaired accuracy differences (pp, 95% CI): " + "; ".join(
                   f"{k}: {100 * v[0]:+.0f} [{100 * v[1]:+.0f}, {100 * v[2]:+.0f}]" for k, v in live["bootstrap_acc_diff"].items()) + "."]
    md += ["\n## Illustrative MathVista trajectories (figure mathvista_trajectories.png)\n",
           "| Question | Rollout | Correct | Tokens | corr(c, v) | C | V | Z = V·C | Z_token |", "|---|---|---|---|---|---|---|---|---|"]
    for r in traj.itertuples():
        md.append(f"| {r.sample_id} | {r.rollout} | {r.correct} | {r.tokens} | {r.corr_c_v:+.2f} | {r.C:.3f} | "
                  f"{r.V:.3f} | {r.Z:.4f} | {r.Z_token:.4f} |")
    open(f"{R}/report_tables.md", "w", encoding="utf-8").write("\n".join(md) + "\n")


# ---------------------------------------------------------------- examples
def groups_line(q, idx):
    g = q["gid"][idx]
    vals, cnt = np.unique(g, return_counts=True)
    order = np.argsort(-cnt, kind="stable")
    return ", ".join(f"`{q['reps'][vals[i]]}` ×{cnt[i]}{' ✓' if q['group_correct'][vals[i]] else ''}" for i in order[:5])


def examples(D_all, texts, live_calls, live_pq):
    md = ["# Task 3 examples (rule-selected)\n",
          "Rules: held-out questions in split order, saved order 0. *Disagreement*: the first question where "
          "self-consistency and the RWC method differ at N=32 with SC wrong and RWC right, and the first with the "
          "reverse. *Premature stop*: the first question where the quality-aware controller stopped early with a "
          "wrong answer that fixed-32 self-consistency got right. *Revision*: the first live question of each "
          "transition type, plus the first wrong-to-wrong case whose answer changed. ✓ marks the group graded "
          "correct (retrospective only).\n"]
    for name in NAMES:
        D, sel = D_all[name], pd.read_parquet(f"{R}/selections_{name}.parquet")
        s = sel[(sel.order == 0) & (sel.N == 32) & sel.method.isin(["self_consistency", RWC[name]])]
        piv = s.pivot(index="sample_id", columns="method", values="correct")
        md.append(f"\n## {NAMES[name]}\n")
        for want, title in (((False, True), "self-consistency wrong, RWC right"), ((True, False), "self-consistency right, RWC wrong")):
            for qid in D["split"]["eval"]:
                if qid in piv.index and tuple(piv.loc[qid, ["self_consistency", RWC[name]]]) == want:
                    q, idx = D["questions"][qid], D["orders"][qid][0][:32]
                    ans = s[s.sample_id == qid].set_index("method").answer
                    md += [f"\n### Selection disagreement ({title}): {qid}\n", f"> {texts[name][qid][:400]}\n",
                           f"Gold `{q['gold']}`. Self-consistency → `{ans['self_consistency']}`; {LABEL[RWC[name]]} → "
                           f"`{ans[RWC[name]]}`. Answer groups among the 32: {groups_line(q, idx)}."]
                    break
        ad = pd.read_csv(f"{R}/adaptive_{name}_quality.csv", dtype={"sample_id": str})
        ad = ad[ad.order == 0].set_index("sample_id")
        for qid in D["split"]["eval"]:
            r = ad.loc[qid]
            if r.stop_n < 32 and not r.correct and r.fixed32_correct:
                q, order = D["questions"][qid], D["orders"][qid][0]
                md += [f"\n### Premature stop: {qid} (stopped after {int(r.stop_n)} candidates)\n",
                       f"> {texts[name][qid][:400]}\n", f"Gold `{q['gold']}`. Revealed groups at the stop: "
                       f"{groups_line(q, order[:int(r.stop_n)])}. All 32: {groups_line(q, order[:32])}."]
                break
    if live_calls is not None:
        md.append("\n## Live sequential refinement (MATH)\n")
        gold = {q["sample_id"]: q["gold_answer"] for q in read_json("manifests/task3_live_questions.json")}
        pq = live_pq[live_pq.strategy == "refine"]
        picks = []
        for t in ("W->C", "C->W"):
            hit = pq[pq.transition == t]
            if len(hit):
                picks.append((hit.sample_id.iloc[0], t))
        ww = pq[(pq.transition == "W->W") & pq.answer_changed]
        if len(ww):
            picks.append((ww.sample_id.iloc[0], "W->W, answer changed"))
        for qid, t in picks:
            c = live_calls[live_calls.sample_id == qid].set_index("call")
            md += [f"\n### {t}: {qid}\n", f"> {texts['math'].get(qid, '')[:400]}\n",
                   f"Initial answer `{c.loc['parallel_0', 'extracted']}` → revised `{c.loc['refine', 'extracted']}` "
                   f"(gold `{gold[qid]}`).",
                   "\nEnd of initial response:\n", "```", str(c.loc["parallel_0", "response"])[-300:], "```",
                   "\nEnd of revision:\n", "```", str(c.loc["refine", "response"])[-300:], "```"]
    open(f"{R}/examples.md", "w", encoding="utf-8").write("\n".join(md) + "\n")


def main():
    os.makedirs(FIG, exist_ok=True)
    summ = {n: pd.read_csv(f"{R}/summary_{n}.csv") for n in NAMES}
    dec = {n: pd.read_csv(f"{R}/decomposition_{n}.csv") for n in NAMES}
    boot = {n: read_json(f"{R}/bootstrap_{n}.json") for n in NAMES}
    sig = {n: read_json(f"{R}/signals_{n}.json") for n in NAMES}
    adaptive = read_json(f"{R}/adaptive.json")
    live = read_json(f"{R}/live/summary.json") if os.path.exists(f"{R}/live/summary.json") else None
    D_all = {n: load_pkl(n) for n in NAMES}
    fig_curves(summ)
    fig_cost(summ, adaptive)
    fig_decomposition(dec, summ)
    traj = fig_trajectories(D_all["mathvista"])
    pools = {n: stopping_vs_pool(n, D_all[n], adaptive) for n in NAMES}
    tables(summ, dec, boot, sig, adaptive, live, pools, traj)
    texts = {}
    for n, folder in (("math", "qwen25-math"), ("mathvista", "qwen25vl-mathvista")):
        g = pd.read_parquet(f"data/task3/{folder}/generations.parquet", columns=["sample_id", "rollout_id", "question_text"])
        texts[n] = dict(zip(g[g.rollout_id == 0].sample_id, g[g.rollout_id == 0].question_text))
    calls = pq = None
    if os.path.exists("outputs/task3_live/calls.jsonl"):
        calls = pd.DataFrame([json.loads(l) for l in open("outputs/task3_live/calls.jsonl", encoding="utf-8")])
        pq = pd.read_csv(f"{R}/live/per_question.csv", dtype={"sample_id": str})
    examples(D_all, texts, calls, pq)
    print("wrote", FIG, f"{R}/report_tables.md", f"{R}/examples.md")


if __name__ == "__main__":
    main()
