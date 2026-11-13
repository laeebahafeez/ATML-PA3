"""Task 3 Steps 1-3: repeated sampling, selection/aggregation, signal analysis.

    python -m task3.offline        # needs outputs/task3/{math,mathvista}.pkl (task3.data)

Writes results/task3/:
  selections_<ds>.parquet   one row per (question, order, N, method): selected rollout, answer, correct, cost
  per_question_<ds>.csv     accuracy/coverage/cost averaged over the 10 orders
  summary_<ds>.csv          accuracy, oracle coverage, cost per method and N (dev / eval)
  decomposition_<ds>.csv    sampling vs selection failures
  bootstrap_<ds>.json       paired question-level bootstrap CIs (eval split)
  signals_<ds>.json         Step 3 associations
"""
import os
import pickle

import numpy as np
import pandas as pd

from common.utils import SEED, write_json
from task3 import methods as M

BUDGETS = (1, 2, 4, 8, 16, 32)
OUT = "results/task3"
N_BOOT = 1000


def load(name):
    with open(f"outputs/task3/{name}.pkl", "rb") as f:
        return pickle.load(f)


def run_selection(name, D):
    meths = M.methods_for(name)
    rows = []
    for qid, q in D["questions"].items():
        for o, order in enumerate(D["orders"][qid]):
            for N in BUDGETS:
                idx = order[:N]
                oracle = bool(q["correct"][idx].any())
                n_groups = len(set(q["gid"][idx]))
                cost_all = int(q["T"][idx].sum())
                for mname, fn in meths.items():
                    pos = fn(q, idx)
                    rid = int(idx[pos])
                    rows.append((qid, o, N, mname, rid, q["reps"][q["gid"][rid]],
                                 bool(q["correct"][rid]),
                                 int(q["T"][idx[0]]) if mname == "first_ans" else cost_all,
                                 oracle, n_groups))
    return pd.DataFrame(rows, columns=["sample_id", "order", "N", "method", "selected_rollout",
                                       "answer", "correct", "token_cost", "oracle", "n_groups"])


def paired_boot(a, b, rng):
    """95% CI of mean(a - b) with question-level paired resampling."""
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "n_questions": int(len(d))}


BOOT_PAIRS = {
    "math": [("self_consistency", "first_ans"), ("selfcert_rwc", "self_consistency"),
             ("best_selfcert", "self_consistency"), ("best_selfcert", "best_logp"),
             ("selfcert_rwc", "best_selfcert"), ("best_logp", "first_ans")],
    "mathvista": [("self_consistency", "first_ans"), ("selfcert_rwc", "self_consistency"),
                  ("best_selfcert", "self_consistency"), ("visconf_rwc", "self_consistency"),
                  ("visconf_rwc", "best_visconf"), ("best_visconf", "best_selfcert"),
                  ("vision_rwc", "self_consistency"), ("visconf_rwc", "visconf_token_rwc"),
                  ("best_visconf", "best_visconf_token")],
}


def summarize(name, D, sel):
    split = {q: s for s in ("dev", "eval") for q in D["split"][s]}
    sel["split"] = sel.sample_id.map(split)
    perq = sel.groupby(["split", "sample_id", "method", "N"], as_index=False).agg(
        acc=("correct", "mean"), oracle=("oracle", "mean"), cost=("token_cost", "mean"),
        n_groups=("n_groups", "mean"))
    perq.to_csv(f"{OUT}/per_question_{name}.csv", index=False)
    summ = perq.groupby(["split", "method", "N"], as_index=False)[["acc", "oracle", "cost", "n_groups"]].mean()
    summ.to_csv(f"{OUT}/summary_{name}.csv", index=False)

    # sampling vs selection failures (per order, then averaged)
    sel["outcome"] = np.where(sel.correct, "correct", np.where(sel.oracle, "selection_failure",
                                                               "sampling_failure"))
    dec = (sel.groupby(["split", "method", "N"]).outcome.value_counts(normalize=True)
           .unstack(fill_value=0).reset_index())
    dec.to_csv(f"{OUT}/decomposition_{name}.csv", index=False)

    rng = np.random.default_rng(SEED)
    ev = perq[perq.split == "eval"]
    boot = {}
    for N in BUDGETS:
        piv = ev[ev.N == N].pivot(index="sample_id", columns="method", values="acc")
        for a, b in BOOT_PAIRS[name]:
            boot[f"N{N}:{a}-{b}"] = paired_boot(piv[a], piv[b], rng)
    write_json(boot, f"{OUT}/bootstrap_{name}.json")
    return summ


# ---------------------------------------------------------------- Step 3
def quartile_gap(score, correct):
    """Correct rate among the top 8 minus bottom 8 by score (stable, earliest id first)."""
    order = np.lexsort((np.arange(len(score)), -np.asarray(score)))
    k = len(score) // 4
    return correct[order[:k]].mean() - correct[order[-k:]].mean()


def spearman(x, y):
    from scipy.stats import spearmanr
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y).statistic


def signals(name, D):
    idx = np.arange(32)
    res = {}
    qs = {q: D["questions"][q] for q in D["split"]["eval"]}
    scorers = {"logp": M.score_L, "selfcert": M.score_K, "length": lambda q, i: q["T"][i].astype(float)}
    if name == "mathvista":
        scorers.update(vision=M.score_V, visconf=M.score_visconf, visconf_token=M.score_visconf_token)
    for sname, fn in scorers.items():
        gaps, gaps_mixed, rho, rho_len = [], [], [], []
        for q in qs.values():
            s = fn(q, idx)
            c = q["correct"].astype(float)
            g = quartile_gap(s, c)
            gaps.append(g)
            if 0 < c.mean() < 1:
                gaps_mixed.append(g)
                rho.append(spearman(s, c))
                rho_len.append(spearman(s, q["T"].astype(float)))
        res[sname] = {"top_minus_bottom_quartile_all": float(np.mean(gaps)),
                      "top_minus_bottom_quartile_mixed": float(np.mean(gaps_mixed)),
                      "n_mixed_questions": len(gaps_mixed),
                      "mean_spearman_with_correct_mixed": float(np.nanmean(rho)),
                      "mean_spearman_with_length_mixed": float(np.nanmean(rho_len))}
    if name == "mathvista":
        per_q, cov_check = [], []
        for q in qs.values():
            c = M.calibrated_conf(q, idx)
            r = []
            for ci, vi in zip(c, q["v_tokens"]):
                if len(ci) > 2 and ci.std() > 0 and vi.std() > 0:
                    r.append(np.corrcoef(ci, vi)[0, 1])
                if len(ci):
                    cov_check.append(abs((ci * vi).mean() - ci.mean() * vi.mean()
                                         - np.cov(ci, vi, bias=True)[0, 1] if len(ci) > 1 else 0))
            per_q.append(np.mean(r) if r else np.nan)
        per_q = np.array(per_q)
        zt_minus_z = [(M.score_visconf_token(q, idx) - M.score_visconf(q, idx)).mean() for q in qs.values()]
        res["confidence_vision_corr"] = {
            "mean_over_questions": float(np.nanmean(per_q)), "median": float(np.nanmedian(per_q)),
            "frac_questions_positive": float(np.nanmean(per_q > 0)),
            "q25": float(np.nanpercentile(per_q, 25)), "q75": float(np.nanpercentile(per_q, 75)),
            "mean_Ztoken_minus_Z": float(np.mean(zt_minus_z)),
            "max_cov_identity_error": float(np.max(cov_check)),
        }
    res["pool_stats"] = {
        "mean_rollout_acc": float(np.mean([q["correct"].mean() for q in qs.values()])),
        "frac_all_correct": float(np.mean([q["correct"].all() for q in qs.values()])),
        "frac_none_correct": float(np.mean([not q["correct"].any() for q in qs.values()])),
        "mean_len_correct": float(np.mean(np.concatenate([q["T"][q["correct"]] for q in qs.values()]))),
        "mean_len_incorrect": float(np.mean(np.concatenate([q["T"][~q["correct"]] for q in qs.values()]))),
        "capped_rate": float(np.mean(np.concatenate([q["capped"] for q in qs.values()]))),
        "no_answer_rate": float(np.mean(np.concatenate([q["no_answer"] for q in qs.values()]))),
    }
    write_json(res, f"{OUT}/signals_{name}.json")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in ("math", "mathvista"):
        D = load(name)
        sel = run_selection(name, D)
        sel.to_parquet(f"{OUT}/selections_{name}.parquet", index=False)
        summ = summarize(name, D, sel)
        ev = summ[summ.split == "eval"].pivot(index="method", columns="N", values="acc")
        print(f"\n== {name} eval accuracy\n", ev.round(4).to_string())
        print(summ[(summ.split == "eval") & (summ.method == "self_consistency")][["N", "oracle", "cost", "n_groups"]].round(3).to_string())
        signals(name, D)


if __name__ == "__main__":
    main()
