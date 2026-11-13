"""Figure of rule-selected complementary pairs (Task 2, Step 6) for the report.

    python -m task2.example_figure --qual results/task2/qualitative_run196.csv --images DIR --out report/figures/t2_examples.png

DIR holds the COCO val2014 images named <image_id>.jpg (download them from images.cocodataset.org).
Pairs shown: the first grounded presence pair, the first image-ignored counting pair, and the first grounded
and first image-ignored VQAv2 "other" pairs from the rule-selected table (first pair of each outcome category).
"""
import argparse
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from common import plotstyle

PAIRS = [("grounding", "0", "Does this image contain a cell phone?", "Grounded"),
         ("grounding", "32", "How many cars can you see?", "Image ignored"),
         ("vqa", "66", "What is the gender of the primary individual in the picture?", "Grounded"),
         ("vqa", "60", "What color is her dress?", "Image ignored")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qual", default="results/task2/qualitative_run196.csv")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="report/figures/t2_examples.png")
    a = ap.parse_args()
    Q = pd.read_csv(a.qual, dtype=str)
    plotstyle.apply()
    fig, axes = plt.subplots(2, 4, figsize=(11, 7.4))
    axes = axes.ravel()
    for p, (suite, pair, question, outcome) in enumerate(PAIRS):
        rows = Q[(Q.suite == suite) & (Q.pair == pair)].sort_values("k")
        for j, r in enumerate(rows.itertuples()):
            ax = axes[2 * p + j]
            im = Image.open(os.path.join(a.images, f"{int(r.image)}.jpg")).convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ok = r.pred_correct == r.reference
            ax.set_xlabel(f"reference: {r.reference}\nanswer: {r.pred_correct} {'(right)' if ok else '(wrong)'}\n"
                          f"with partner image: {r.pred_swap}", fontsize=10.5)
        title = "\n".join(textwrap.wrap(f"{outcome}: {question}", 50))
        axes[2 * p].set_title(title, fontsize=11, loc="left", x=0.0, ha="left")
    plt.tight_layout(w_pad=0.6)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=160, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
