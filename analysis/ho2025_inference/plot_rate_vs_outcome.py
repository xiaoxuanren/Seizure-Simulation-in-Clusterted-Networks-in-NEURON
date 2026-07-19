#!/usr/bin/env python3
"""Figure: per-neuron firing rate vs TP/FP (presyn & postsyn) and vs total predicted edges.
Reads results/per_neuron_outcomes.csv (produced by analyze.py). Run analyze.py first.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "per_neuron_outcomes.csv"))))


def col(cond, key, cast=float):
    return np.array([cast(r[key]) for r in rows if r["condition"] == cond])
data = {c: dict(rate=col(c, "rate_hz"), pops=col(c, "population", str),
                TPpre=col(c, "tp_pre"), FPpre=col(c, "fp_pre"),
                TPpost=col(c, "tp_post"), FPpost=col(c, "fp_post"),
                predpre=col(c, "predicted_out"))
        for c in ("normal", "mauve")}

TPc, FPc = "#0ca30c", "#d03b3b"                          # good / critical, redundant markers
PC = {"E2": "#2a78d6", "E5": "#eb6834", "I2": "#4a3aa7"}
PM = {"E2": "o", "E5": "s", "I2": "^"}
INK, SEC, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8})

fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
fig.patch.set_facecolor(SURF)
col_ymax = {0: max(max(data[c]["TPpre"].max(), data[c]["FPpre"].max()) for c in data),
            1: max(max(data[c]["TPpost"].max(), data[c]["FPpost"].max()) for c in data),
            2: max(data[c]["predpre"].max() for c in data)}

for row, cond in enumerate(("normal", "mauve")):
    dd = data[cond]
    ax = axes[row, 0]
    ax.scatter(dd["rate"], dd["TPpre"], s=22, c=TPc, marker="o", alpha=0.6, linewidths=0, label="TP")
    ax.scatter(dd["rate"], dd["FPpre"], s=30, c=FPc, marker="x", alpha=0.75, linewidths=1.3, label="FP")
    ax.set_ylabel(f"{cond}\n\nedges (count)", fontsize=10, color=INK)
    if row == 0:
        ax.set_title("Presynaptic rate  vs  TP / FP", fontsize=11.5, pad=8)
        ax.legend(frameon=False, loc="upper left", markerscale=1.3)
    ax.set_ylim(-col_ymax[0] * 0.04, col_ymax[0] * 1.08)
    ax = axes[row, 1]
    ax.scatter(dd["rate"], dd["TPpost"], s=22, c=TPc, marker="o", alpha=0.6, linewidths=0, label="TP")
    ax.scatter(dd["rate"], dd["FPpost"], s=30, c=FPc, marker="x", alpha=0.75, linewidths=1.3, label="FP")
    if row == 0:
        ax.set_title("Postsynaptic rate  vs  TP / FP", fontsize=11.5, pad=8)
        ax.legend(frameon=False, loc="upper left", markerscale=1.3)
    ax.set_ylim(-col_ymax[1] * 0.04, col_ymax[1] * 1.08)
    ax = axes[row, 2]
    for g in ("E2", "E5", "I2"):
        m = dd["pops"] == g
        ax.scatter(dd["rate"][m], dd["predpre"][m], s=24, c=PC[g], marker=PM[g],
                   alpha=0.65, linewidths=0, label=g)
    if row == 0:
        ax.set_title("Rate  vs  total predicted edges (out)", fontsize=11.5, pad=8)
        ax.legend(frameon=False, loc="upper left", title="population", title_fontsize=9)
    ax.set_ylim(-col_ymax[2] * 0.04, col_ymax[2] * 1.08)
    for c in range(3):
        axes[row, c].grid(True, color=GRID, linewidth=0.7); axes[row, c].set_axisbelow(True)
        for s in ("top", "right"):
            axes[row, c].spines[s].set_visible(False)
for c in range(3):
    axes[1, c].set_xlabel("firing rate (Hz)")

fig.text(0.5, 0.008,
         "Two rate clusters per panel: pyramidal E (~1.5–2.2 Hz, left) and FS I2 (~6.6–7.3 Hz, right). "
         "FP does NOT scale with firing rate; in mauve the I2 cluster's TP (as presynaptic source) drops to ~0 "
         "despite unchanged high rate — the inhibitory collapse is a timing effect, not a rate effect.",
         ha="center", fontsize=9, color=SEC)
fig.suptitle("Per-neuron firing rate vs prediction outcome — normal (top) vs mauve (bottom)",
             fontsize=13, fontweight="bold", color=INK, y=0.985)
fig.tight_layout(rect=[0, 0.028, 1, 0.955])
out = os.path.join(HERE, "results", "ho_rate_vs_outcome.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print("saved", out)
