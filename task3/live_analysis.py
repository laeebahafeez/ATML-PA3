"""Score the Step 4 live-generation run (math-verify) and compare the three strategies.

    python -m task3.live_analysis --calls outputs/task3_live/calls.jsonl
"""
import argparse
import json

import numpy as np
import pandas as pd

from common.utils import SEED, read_json, write_json
from task3.data import NO_ANSWER, math_equiv


def grade(ans, gold):
    return ans is not None and math_equiv(str(gold), ans)


def self_consistency(answers):
    """Majority over math-verify-equivalent groups; NO_ANSWER is a group; ties -> earliest."""
    reps, counts, first = [], [], []
    for i, a in enumerate(answers):
        key = NO_ANSWER if a is None else a
        for j, r in enumerate(reps):
            if (r == NO_ANSWER) == (key == NO_ANSWER) and (r == key or (r != NO_ANSWER and math_equiv(r, key))):
                counts[j] += 1
                break
        else:
            reps.append(key)
            counts.append(1)
            first.append(i)
    best = max(range(len(reps)), key=lambda j: (counts[j], -first[j]))
    return None if reps[best] == NO_ANSWER else reps[best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="outputs/task3_live/calls.jsonl")
    ap.add_argument("--questions", default="manifests/task3_live_questions.json")
    ap.add_argument("--out", default="results/task3/live")
    a = ap.parse_args()
    gold = {q["sample_id"]: q["gold_answer"] for q in read_json(a.questions)}
    calls = pd.DataFrame([json.loads(l) for l in open(a.calls, encoding="utf-8")])
    # pandas turns missing extractions into NaN; keep them as None (the NO_ANSWER group)
    calls["extracted"] = calls["extracted"].astype(object).where(calls["extracted"].notna(), None)
    rows = []
    for qid, d in calls.groupby("sample_id"):
        c = {r.call: r for r in d.itertuples()}
        par = [c[f"parallel_{k}"] for k in range(3)]
        strategies = {
            "single": ([c["single"]], c["single"].extracted),
            "parallel": (par, self_consistency([p.extracted for p in par])),
            "refine": ([c["parallel_0"], c["refine"]], c["refine"].extracted),
        }
        for s, (used, ans) in strategies.items():
            rows.append({
                "sample_id": qid, "strategy": s, "answer": ans, "correct": grade(ans, gold[qid]),
                "input_tokens": sum(u.input_tokens for u in used),
                "output_tokens": sum(u.output_tokens for u in used),
                "seconds": sum(u.seconds for u in used),
                "any_truncated": any(u.hit_cap for u in used),
                "final_truncated": used[-1].hit_cap if s != "parallel" else np.mean([u.hit_cap for u in used]),
                "extracted": ans is not None,
            })
        init_ok = grade(c["parallel_0"].extracted, gold[qid])
        ref_ok = grade(c["refine"].extracted, gold[qid])
        rows[-1]["initial_correct"] = init_ok
        rows[-1]["transition"] = {(0, 0): "W->W", (0, 1): "W->C", (1, 0): "C->W", (1, 1): "C->C"}[(init_ok, ref_ok)]
        rows[-1]["answer_changed"] = not (
            (c["parallel_0"].extracted is None and c["refine"].extracted is None) or
            (c["parallel_0"].extracted is not None and c["refine"].extracted is not None
             and math_equiv(c["parallel_0"].extracted, c["refine"].extracted)))
    df = pd.DataFrame(rows)
    import os
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/per_question.csv", index=False)
    summ = df.groupby("strategy").agg(
        accuracy=("correct", "mean"), mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"), mean_seconds=("seconds", "mean"),
        truncation_rate=("any_truncated", "mean"), extraction_rate=("extracted", "mean")).reset_index()
    # per-call rates (parallel strategy has 3 calls)
    per_call = calls.assign(kind=calls.call.str.replace(r"_\d$", "", regex=True)).groupby("kind").agg(
        call_truncation=("hit_cap", "mean"), call_extraction=("extracted", lambda x: x.notna().mean()),
        call_output_tokens=("output_tokens", "mean"))
    rng = np.random.default_rng(SEED)
    piv = df.pivot(index="sample_id", columns="strategy", values="correct").astype(float)
    boot = {}
    for x, y in (("parallel", "single"), ("refine", "single"), ("refine", "parallel")):
        dd = (piv[x] - piv[y]).to_numpy()
        bs = dd[rng.integers(0, len(dd), (1000, len(dd)))].mean(1)
        boot[f"{x}-{y}"] = [float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ref = df[df.strategy == "refine"]
    out = {"summary": summ.to_dict(orient="records"), "per_call": per_call.reset_index().to_dict(orient="records"),
           "bootstrap_acc_diff": boot,
           "transitions": ref.transition.value_counts().to_dict(),
           "refine_answer_changed_rate": float(ref.answer_changed.mean()),
           "initial_accuracy": float(ref.initial_correct.mean())}
    write_json(out, f"{a.out}/summary.json")
    print(summ.round(3).to_string())
    print(out["transitions"], out["bootstrap_acc_diff"])


if __name__ == "__main__":
    main()
