"""Change from the baseline on each benchmark's primary metric (report Figure B.1).

    python -m task1.benchmark_figure --bench results/task1/benchmarks.csv --out report/figures/t1_benchmark_changes.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import plotstyle

ORDER = ["full_u625", "full", "l2sp", "lora4", "lora16", "oplora16"]
PANELS = [("GSM8K (flex EM)", "GSM8K"), ("HellaSwag acc_norm", "HellaSwag (norm.)"),
          ("BoolQ acc", "BoolQ"), ("ARC-E acc_norm", "ARC-Easy (norm.)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results/task1/benchmarks.csv")
    ap.add_argument("--out", default="report/figures/t1_benchmark_changes.png")
    a = ap.parse_args()
    plotstyle.apply()
    gain, loss = plotstyle.PALETTE[0], plotstyle.PALETTE[1]  # polarity: gains blue, losses orange
    T = pd.read_csv(a.bench).set_index("condition")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True, sharex=True)
    for ax, (col, title) in zip(axes, PANELS):
        v = T.loc[ORDER, col + " Δpp"]
        ax.barh(range(len(ORDER)), v, height=0.6, color=[gain if x >= 0 else loss for x in v])
        for i, x in enumerate(v):
            ax.text(x + (0.8 if x >= 0 else -0.8), i, f"{x:+.1f}", va="center",
                    ha="left" if x >= 0 else "right", fontsize=9.5, color=plotstyle.INK)
        ax.axvline(0, color=plotstyle.NEUTRAL, lw=0.9)
        ax.set_xlim(-12, 30)
        ax.set_title(title)
        ax.set_xlabel("change from baseline (points)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(ORDER)), [T.loc[c, "label"] for c in ORDER])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.5)
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
