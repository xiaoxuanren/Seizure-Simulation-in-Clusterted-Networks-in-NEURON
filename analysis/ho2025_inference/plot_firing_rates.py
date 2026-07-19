#!/usr/bin/env python3
"""Figure: per-group firing rates + pre/post rate by connection class.
Reads results/per_neuron_outcomes.csv (produced by analyze.py). Run analyze.py first.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results", "per_neuron_outcomes.csv"))))
data = {}
for c in ("normal", "mauve"):
    sub = [r for r in rows if r["condition"] == c]
    data[c] = (np.array([r["population"] for r in sub]),
               np.array([float(r["rate_hz"]) for r in sub]))

C = {"normal": "#2a78d6", "mauve": "#eb6834"}                 # CVD-safe blue / orange
INK, SEC, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
                     "font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.05, 1.25]})
fig.patch.set_facecolor(SURF)

groups = ["E2", "E5", "I2"]
x = np.arange(len(groups)); w = 0.38
for k, cond in enumerate(("normal", "mauve")):
    pops, rate = data[cond]
    means = [rate[pops == g].mean() for g in groups]
    sds = [rate[pops == g].std() for g in groups]
    xs = x + (k - 0.5) * w
    axA.bar(xs, means, w * 0.92, color=C[cond], label=cond, zorder=2,
            yerr=sds, error_kw=dict(ecolor=SEC, elinewidth=1, capsize=3, zorder=4))
    for xi, g in zip(xs, groups):
        r = rate[pops == g]
        jit = (np.random.RandomState(1).rand(len(r)) - 0.5) * w * 0.55
        axA.scatter(np.full(len(r), xi) + jit, r, s=7, color="white",
                    edgecolor=INK, linewidth=0.35, alpha=0.55, zorder=3)
    for xi, m, sd in zip(xs, means, sds):
        axA.text(xi, m + sd + 0.25, f"{m:.2f}", ha="center", va="bottom",
                 fontsize=9, color=INK, fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels([f"{g}\n({'FS interneuron' if g=='I2' else 'pyramidal'})" for g in groups])
axA.set_ylabel("Firing rate (Hz)")
axA.set_title("Per-neuron firing rate by population", fontsize=12, color=INK, pad=8)
axA.legend(frameon=False, loc="upper left")
axA.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
axA.set_ylim(0, 9)


def grate(cond, sel):
    pops, rate = data[cond]
    return rate[np.isin(pops, sel)].mean()
E, I = ["E2", "E5"], ["I2"]
classes = [("E→E", E, E), ("E→I", E, I), ("I→E", I, E), ("I→I", I, I)]
xb = np.arange(len(classes)); wb = 0.2
for cond, role, off in [("normal", "pre", -1.5), ("normal", "post", -0.5),
                        ("mauve", "pre", 0.5), ("mauve", "post", 1.5)]:
    vals = [grate(cond, pre if role == "pre" else post) for _, pre, post in classes]
    axB.bar(xb + off * wb, vals, wb * 0.9, color=C[cond], zorder=2,
            hatch="" if role == "pre" else "////", edgecolor="white", linewidth=0.6)
    for xi, v in zip(xb + off * wb, vals):
        axB.text(xi, v + 0.22, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=SEC)
axB.set_xticks(xb); axB.set_xticklabels([c[0] for c in classes], fontsize=13)
axB.set_ylabel("Mean firing rate (Hz)")
axB.set_title("Pre- vs post-synaptic rate by connection class", fontsize=12, color=INK, pad=8)
axB.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0); axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.set_ylim(0, 9)
leg = [Patch(facecolor=C["normal"], label="normal"), Patch(facecolor=C["mauve"], label="mauve"),
       Patch(facecolor="#bbb", label="presynaptic (solid)"),
       Patch(facecolor="#bbb", hatch="////", edgecolor="white", label="postsynaptic (hatched)")]
axB.legend(handles=leg, frameon=False, loc="upper left", fontsize=9)

fig.suptitle("Ho2025 firing rates — normal vs mauve (900 s, N=303: E2=65, E5=206, I2=32)",
             fontsize=13, fontweight="bold", color=INK, y=0.99)
fig.tight_layout(rect=[0, 0.0, 1, 0.955])
out = os.path.join(HERE, "results", "ho_firing_rates.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print("saved", out)
