"""Task 3 Step 5: adaptive sampling (offline simulation over the saved orders).

    python -m task3.adaptive

Rollouts are revealed in each saved order; stopping is checked after n in {4, 8, 16}
candidates, otherwise 32 are consumed. Both controllers return the self-consistency
answer of the revealed candidates. Only revealed candidates are used for votes, scores,
VisConf calibration and ranks. Settings are chosen on the dev split, then frozen.
"""
import itertools
import os
import pickle

import numpy as np
import pandas as pd

from common.utils import SEED, write_json
from task3 import methods as M
from task3.data import NO_ANSWER

CHECKS = (4, 8, 16)
TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMAS = (0.0, 0.1, 0.25, 0.5)
OUT = "results/task3"


def prefix_stats(q, idx, quality):
    """Everything the controllers need at one checkpoint (revealed candidates only)."""
    gids = q["gid"][idx]
    vals, counts = np.unique(gids, return_counts=True)
    top = counts.max()
    leaders = vals[counts == top]
    lead_unique = len(leaders) == 1
    lead = leaders[0] if lead_unique else None
    sc_pos = M.self_consistency(q, idx)
    sc_gid = gids[sc_pos]
    score = M.score_K(q, idx) if quality == "selfcert" else M.score_visconf(q, idx)
    rwc_gid = gids[M.rwc(q, idx, score)]
    return {
        "lead_frac": top / len(idx),
        "lead_ok": lead_unique and q["reps"][lead] != NO_ANSWER,
        "sc_gid": sc_gid, "sc_correct": bool(q["group_correct"][sc_gid]),
        "agree": sc_gid == rwc_gid, "margin": M.rwc_margin(q, idx, score),
        "cost": int(q["T"][idx].sum()),
    }


def precompute(D, qids, quality):
    """stats[qid][order][n] for n in CHECKS + (32,)."""
    out = {}
    for qid in qids:
        q = D["questions"][qid]
        out[qid] = [{n: prefix_stats(q, order[:n], quality) for n in CHECKS + (32,)}
                    for order in D["orders"][qid]]
    return out


def simulate(stats, policy, tau=None, gamma=None):
    """Returns DataFrame (qid, order, stop_n, correct, cost)."""
    rows = []
    for qid, per_order in stats.items():
        for o, st in enumerate(per_order):
            stop = 32
            if policy != "fixed32":
                for n in CHECKS:
                    s = st[n]
                    ok = s["lead_ok"] and s["lead_frac"] >= tau - 1e-12
                    if policy == "quality":
                        ok = ok and s["agree"] and s["margin"] >= gamma - 1e-12
                    if ok:
                        stop = n
                        break
            s = st[stop]
            # premature stop that fixed-32 SC would have fixed (retrospective only)
            rows.append((qid, o, stop, s["sc_correct"], s["cost"], st[32]["sc_correct"]))
    return pd.DataFrame(rows, columns=["sample_id", "order", "stop_n", "correct", "cost", "fixed32_correct"])


def agg(df):
    perq = df.groupby("sample_id")[["correct", "cost", "stop_n"]].mean()
    return {"acc": float(perq.correct.mean()), "cost": float(perq.cost.mean()),
            "rollouts": float(perq.stop_n.mean())}


def select(dev_stats, family):
    base = agg(simulate(dev_stats, "fixed32"))
    cands = [("fixed32", None, None, base)]
    grid = [(t, None) for t in TAUS] if family == "agreement" else list(itertools.product(TAUS, GAMMAS))
    for t, g in grid:
        cands.append((family, t, g, agg(simulate(dev_stats, family, t, g))))
    ok = [c for c in cands if c[3]["acc"] >= base["acc"] - 0.01]
    best = min(ok, key=lambda c: (c[3]["cost"], -c[3]["acc"], c[1] if c[1] is not None else 9,
                                  c[2] if c[2] is not None else 9))
    table = [{"policy": c[0], "tau": c[1], "gamma": c[2], **c[3]} for c in cands]
    return best, table


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}
    rng = np.random.default_rng(SEED)
    for name, quality in (("math", "selfcert"), ("mathvista", "visconf")):
        with open(f"outputs/task3/{name}.pkl", "rb") as f:
            D = pickle.load(f)
        dev = precompute(D, D["split"]["dev"], quality)
        ev = precompute(D, D["split"]["eval"], quality)
        res = {"quality_signal": quality, "dev_grid": {}, "selected": {}, "eval": {}}
        sims = {}
        for fam in ("agreement", "quality"):
            best, table = select(dev, fam)
            res["dev_grid"][fam] = table
            res["selected"][fam] = {"policy": best[0], "tau": best[1], "gamma": best[2], "dev": best[3]}
            sims[fam] = simulate(ev, best[0], best[1], best[2])
        sims["fixed32"] = simulate(ev, "fixed32")
        for fam, df in sims.items():
            r = agg(df)
            r["stop_distribution"] = {int(k): float(v) for k, v in df.stop_n.value_counts(normalize=True).sort_index().items()}
            early = df[df.stop_n < 32]
            r["early_stops"] = int(len(early))
            r["early_incorrect"] = int((~early.correct).sum())
            r["early_incorrect_recoverable_by_fixed32"] = int((~early.correct & early.fixed32_correct).sum())
            r["early_correct_but_fixed32_wrong"] = int((early.correct & ~early.fixed32_correct).sum())
            res["eval"][fam] = r
            df.to_csv(f"{OUT}/adaptive_{name}_{fam}.csv", index=False)
        # bootstrap: adaptive minus fixed-32 accuracy (paired by question)
        base = sims["fixed32"].groupby("sample_id").correct.mean()
        for fam in ("agreement", "quality"):
            d = (sims[fam].groupby("sample_id").correct.mean() - base).to_numpy()
            bs = d[rng.integers(0, len(d), (1000, len(d)))].mean(1)
            res["eval"][fam]["acc_diff_vs_fixed32_ci"] = [float(d.mean()), float(np.percentile(bs, 2.5)),
                                                          float(np.percentile(bs, 97.5))]
        results[name] = res
        print(name, {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                         if kk in ("acc", "cost", "rollouts")} for k, v in res["eval"].items()},
              res["selected"])
    write_json(results, f"{OUT}/adaptive.json")


if __name__ == "__main__":
    main()
