"""Shared style for the thesis figure set.

Import and call apply() BEFORE creating any figure; use the helpers so every
figure carries identical spines, grids, and panel lettering.
"""
import matplotlib

FIGW = 6.9
DPI = 400
C50 = "#2C6FB5"       # 50-cluster group
C40 = "#C0392B"       # 40-cluster group
TP, FP, FN = "#1a9850", "#d7301f", "#f0932b"


def apply():
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })


def lineax(ax):
    """Line-plot treatment: no top/right spines, light grid behind."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.4, color="0.9")
    ax.set_axisbelow(True)


def letter(ax, ch, dx=-0.14, dy=1.08):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")
