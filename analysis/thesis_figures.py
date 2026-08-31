"""Assemble the three thesis composite figures from existing panels + data.

Figure 1  Model and states: topology, FOV rate maps (normal/seizure),
          burst-marked rasters (normal/seizure). Example: sweep_c50_seed09.
Figure 2  Phenotype across all 20 networks: paired slopes for firing rate,
          burst rate, participation, duration (clean 1x4).
Figure 3  Inference: correctness map, scaling by group, precision vs
          burstiness, exclusion arrows for BOTH states, detection vs weight.

Outputs sweep_summary/thesis_fig{1,2,3}.png at 200 dpi.

    python analysis/thesis_figures.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA  # noqa: E402

OUT = os.path.join(DATA, "sweep_summary")
C50, C40, GREEN = "#1f5fd0", "#c0392b", "#2e8b57"
EX = "sweep_c50_seed09"


def _img(ax, path, title):
    ax.imshow(mpimg.imread(path))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, loc="left")


def fig1():
    topo = os.path.join(DATA, EX, "results", "normal", "figures", "topology_true.png")
    fovN = os.path.join(OUT, "fov_rate_maps", "%s_normal.png" % EX)
    fovS = os.path.join(OUT, "fov_rate_maps", "%s_seizure.png" % EX)
    rasN = os.path.join(DATA, EX, "results", "normal", "figures",
                        "recording000_raster_shuffled.png")
    rasS = os.path.join(DATA, EX, "results", "seizure", "figures",
                        "recording000_raster_shuffled.png")
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1.0, 1.0],
                          hspace=0.10, wspace=0.03)
    _img(fig.add_subplot(gs[0, 0]), topo, "a  network structure (%s)" % EX)
    _img(fig.add_subplot(gs[0, 1]), fovN, "b  firing-rate map, normal")
    _img(fig.add_subplot(gs[0, 2]), fovS, "c  firing-rate map, seizure")
    _img(fig.add_subplot(gs[1, :]), rasN, "d  raster + detected bursts, normal")
    _img(fig.add_subplot(gs[2, :]), rasS,
         "e  raster + detected bursts, seizure (same noise stream as d)")
    p = os.path.join(OUT, "thesis_fig1.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(p)


def fig2():
    r = list(csv.DictReader(open(os.path.join(OUT, "firing_rate_change.csv"),
                                 encoding="utf-8")))
    b = list(csv.DictReader(open(os.path.join(OUT, "burst_change.csv"),
                                 encoding="utf-8")))
    bidx = {x["session"]: x for x in b}

    def bk(d, key):
        row = bidx[d["session"]]
        return float(row["normal_" + key]), float(row["seizure_" + key])

    panels = [
        ("firing rate (Hz)", lambda d: (float(d["normal"]), float(d["seizure"]))),
        ("network bursts / 60 s", lambda d: bk(d, "bursts_per_rec")),
        ("burst participation", lambda d: bk(d, "participation")),
        ("burst duration (ms)", lambda d: bk(d, "duration_ms")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))
    for ax, (label, get) in zip(axes, panels):
        pts = {"c50": [], "c40": []}
        for d in r:
            n_, s_ = get(d)
            col = C50 if d["group"] == "c50" else C40
            ax.plot([0, 1], [n_, s_], "-o", color=col, ms=4.5, lw=1.1, alpha=0.7)
            pts[d["group"]].append((n_, s_))
        for grp, col in (("c50", C50), ("c40", C40)):
            arr = np.array(pts[grp])
            ax.plot([0, 1], arr.mean(0), "-", color=col, lw=3.2,
                    label="%s mean" % grp)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["normal", "seizure"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, loc="upper left")
    for ax, t in zip(axes, "abcd"):
        ax.set_title(t, fontsize=12, loc="left", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(OUT, "thesis_fig2.png")
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print(p)


def _arms(state):
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, "sweep_*", "results", state,
                                           "glm", "burstexcl_arms.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[d["session"]] = d
    return out


def fig3():
    s = list(csv.DictReader(open(os.path.join(OUT, "sweep_summary.csv"),
                                 encoding="utf-8")))
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.15, 0.62],
                          hspace=0.22, wspace=0.30)

    # a: cluster-sorted correctness matrix (right half of the predicted-
    # topology figure; the full-network edge drawing is illegible at ~10k edges)
    from PIL import Image
    src = Image.open(os.path.join(DATA, EX, "results", "normal", "figures",
                                  "glm_predicted_topology.png"))
    w, h = src.size
    mat = src.crop((int(0.50 * w), int(0.06 * h), w, h))
    axa = fig.add_subplot(gs[0, 0])
    axa.imshow(np.asarray(mat))
    axa.set_axis_off()
    axa.set_title("a  predicted vs true adjacency (%s)" % EX,
                  fontsize=10, loc="left")

    axc = fig.add_subplot(gs[0, 1])
    for row in s:
        if row.get("typed_precision") in ("", "nan", None):
            continue
        col = C50 if row["group"] == "c50" else C40
        axc.plot(float(row["bursts_per_rec"]), float(row["typed_precision"]),
                 "o" if row["group"] == "c50" else "s", color=col, ms=8)
    axc.set_xlabel("full-recruitment bursts per recording")
    axc.set_ylabel("edge precision (typed, FDR 0.70)")
    axc.set_title("b  precision vs burstiness, normal state",
                  fontsize=11, loc="left")
    axc.plot([], [], "o", color=C50, label="c50")
    axc.plot([], [], "s", color=C40, label="c40")
    axc.legend(fontsize=8)
    axc.grid(alpha=0.3)

    axd = fig.add_subplot(gs[0, 2])
    for state, marker in (("normal", "o"), ("seizure", "^")):
        for sess, d in _arms(state).items():
            bx = 100.0 * d["dropped_bins"] / d["total_bins"]
            col = C50 if "_c50_" in sess else C40
            axd.annotate("", xy=(bx, d["excl"]["precision"]),
                         xytext=(bx, d["full"]["precision"]),
                         arrowprops=dict(arrowstyle="-|>", color=col,
                                         lw=1.2, alpha=0.75))
            axd.plot(bx, d["full"]["precision"], marker, color=col,
                     ms=5, alpha=0.8)
            axd.plot(bx, d["excl"]["precision"], marker, color=GREEN,
                     ms=5, alpha=0.8)
    axd.plot([], [], "o", color="#555555", label="normal (start)")
    axd.plot([], [], "^", color="#555555", label="seizure (start)")
    axd.plot([], [], "o", color=GREEN, label="after burst exclusion")
    axd.set_xlabel("recording time inside burst windows (%)")
    axd.set_ylabel("edge precision (|W|, FDR 0.70)")
    axd.set_title("c  burst-window exclusion, both states (40 datasets)",
                  fontsize=11, loc="left")
    axd.grid(alpha=0.3)
    axd.legend(fontsize=7, loc="lower right")

    axb = fig.add_subplot(gs[1, :])
    _img(axb, os.path.join(OUT, "scaling_by_group.png"),
         "d  performance vs recording duration (color = burstiness)")

    axe = fig.add_subplot(gs[2, :])
    _img(axe, os.path.join(OUT, "weight_vs_detection.png"),
         "e  detection probability vs synaptic weight (per E/I class)")

    p = os.path.join(OUT, "thesis_fig3.png")
    fig.savefig(p, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
