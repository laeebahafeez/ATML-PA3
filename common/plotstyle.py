"""Shared matplotlib style for every report figure.

Call apply() once after importing pyplot. Colours are a fixed categorical order (assigned per
entity, never cycled past eight); grids and axes are light so the data stands out.
"""
import matplotlib as mpl

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#1f1f1d"        # titles and values
INK_2 = "#55554f"      # axis labels, ticks, legend text
GRID = "#e6e5e0"       # recessive gridlines
NEUTRAL = "#5f5f5a"    # reference lines (oracle, totals)


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8,
        "axes.titlecolor": INK,
        "axes.labelsize": 11,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": "#c9c8c1",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.labelcolor": INK_2,
        "legend.handlelength": 1.8,
        "patch.linewidth": 0,
        "savefig.dpi": 200,
    })
