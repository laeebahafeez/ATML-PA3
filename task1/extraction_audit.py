"""Count GSM8K stock flexible-extract answers that contain no digit (e.g. the LaTeX "$$"
delimiter of a display-math block), per condition, from analyze.py's per-example files.

    python -m task1.extraction_audit --dir results/task1
"""
import argparse
import glob
import os

import pandas as pd

from common.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/task1")
    a = ap.parse_args()
    out = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "gsm8k_per_example_*.csv"))):
        c = os.path.basename(f)[len("gsm8k_per_example_"):-len(".csv")]
        g = pd.read_csv(f, dtype={"extracted": str, "target": str}, keep_default_na=False)
        no_digit = g.extracted.map(lambda e: not any(ch.isdigit() for ch in e))
        out[c] = {"n": len(g), "correct": int(g.correct.sum()), "no_digit_extractions": int(no_digit.sum()),
                  "no_digit_%": 100 * float(no_digit.mean()),
                  "no_digit_values": g.extracted[no_digit].value_counts().to_dict()}
    write_json(out, os.path.join(a.dir, "gsm8k_extraction_audit.json"))
    for c, v in out.items():
        print(c, {k: v[k] for k in ("correct", "no_digit_extractions", "no_digit_values")})


if __name__ == "__main__":
    main()
