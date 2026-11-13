"""Parallel MATH answer grouping with math-verify, with a per-question timeout.

    python -m task3.math_groups      # -> outputs/task3/math_groups.json

math-verify's own timeouts rely on SIGALRM (unavailable on Windows), so each question
runs in a worker process; a question exceeding TIMEOUT falls back to whitespace-stripped
exact-match grouping and is recorded in the output.
"""
import json
import logging
import multiprocessing as mp
import os
import re
import time

import numpy as np
import pandas as pd

TIMEOUT = 120
OUT = "outputs/task3/math_groups.json"


def _quiet():
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    logging.disable(logging.WARNING)


def group_one(args):
    _quiet()
    from task3.data import group_math, math_equiv, NO_ANSWER
    qid, ext, gold = args
    gid, reps = group_math(ext)
    ok = [r != NO_ANSWER and math_equiv(str(gold), r) for r in reps]
    return qid, {"gid": gid.tolist(), "reps": reps, "rep_ok": ok, "fallback": False}


def fallback_one(qid, ext, gold):
    from task3.data import group_keys, NO_ANSWER
    key = lambda s: NO_ANSWER if s is None or not str(s).strip() else re.sub(r"\s+", "", str(s))
    gid, reps = group_keys([key(e) for e in ext])
    return {"gid": gid.tolist(), "reps": reps, "rep_ok": [r != NO_ANSWER and r == key(gold) for r in reps],
            "fallback": True}


def main():
    g = pd.read_parquet("data/task3/qwen25-math/generations.parquet",
                        columns=["sample_id", "rollout_id", "extracted_answer", "gold_answer"])
    g = g.sort_values(["sample_id", "rollout_id"])
    jobs = []
    for qid, d in g.groupby("sample_id", sort=True):
        ext = [None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
               for x in d.extracted_answer]
        jobs.append((qid, ext, d.gold_answer.iloc[0]))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    while todo:
        pool = mp.Pool(max(os.cpu_count() - 1, 1), initializer=_quiet)
        futs = [(j, pool.apply_async(group_one, (j,))) for j in todo]
        restart = False
        for j, f in futs:
            try:
                qid, res = f.get(timeout=TIMEOUT)
                done[qid] = res
            except mp.TimeoutError:
                print("timeout -> fallback:", j[0], flush=True)
                done[j[0]] = fallback_one(*j)
                restart = True
                break
            if len(done) % 100 == 0:
                print(f"{len(done)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
                json.dump(done, open(OUT, "w"))
        pool.terminate()
        todo = [j for j in jobs if j[0] not in done]
        if not restart:
            break
    json.dump(done, open(OUT, "w"))
    print("done", len(done), "fallbacks:", sum(v["fallback"] for v in done.values()))


if __name__ == "__main__":
    main()
