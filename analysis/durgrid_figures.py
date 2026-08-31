"""Render the deck-style duration-grid figures from durgrid_*.json points.

Inputs: sweep_summary/durgrid/durgrid_<session>_n<NNN>.json, one per
(session, n_recordings) point, produced by chtc/analysis_one.py.

Per-session figures (into <session>/results/normal/figures/):
  durgrid_scaling.png       AUC / AP / typed P+R vs recording count (slide 18)
  durgrid_calibration.png   nominal FDR target vs realized FDR, per duration
                            (slide 24)
  durgrid_oracle.png        oracle ceilings vs achieved operating point vs
                            duration (slide 26)

All-dataset figures (into sweep_summary/):
  durgrid_all_scaling.png     20 thin session curves + group means, 4 metrics
  durgrid_all_calibration.png calibration at full duration, all sessions
  durgrid_all_oracle.png      achieved-vs-oracle gap across sessions

Only points present on disk are drawn; partial grids render fine.

    python analysis/durgrid_figures.py             # everything
    python analysis/durgrid_figures.py --all-only  # skip per-session figures
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA  # noqa: E402

GRID = os.path.join(DATA, "sweep_summary", "durgrid")
OUT_ALL = os.path.join(DATA, "sweep_summary")
C50, C40, GREEN, GRAY = "#1f5fd0", "#c0392b", "#2e8b57", "#888888"
FNAME = re.compile(r"durgrid_(?P<session>sweep_c\d+_seed\d+)_n(?P<n>\d+)\.json$")


def load_grid(grid_dir=GRID):
    """{session: {n_recordings: point_dict}}, sorted keys."""
    out = {}
    for p in sorted(glob.glob(os.path.join(grid_dir, "durgrid_*.json"))):
        m = FNAME.search(os.path.basename(p))
        if not m:
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        out.setdefault(m.group("session"), {})[int(m.group("n"))] = d
    return {s: dict(sorted(v.items())) for s, v in sorted(out.items())}


def _sess_color(session):
    return C50 if "_c50_" in session else C40


def _series(pts, getter):
    ns, vals = [], []
    for n, d in pts.items():
        try:
            v = getter(d)
        except (KeyError, IndexError, TypeError):
            continue
        if v is not None and np.isfinite(v):
            ns.append(n)
            vals.append(v)
    return np.array(ns), np.array(vals)


METRICS = [
    ("excitatory AUC", lambda d: d["layers"]["exc"]["auc"]),
    ("excitatory AP", lambda d: d["layers"]["exc"]["ap"]),
    ("typed precision (FDR 0.70)", lambda d: d["typed_at_070"]["precision"]),
    ("typed recall (FDR 0.70)", lambda d: d["typed_at_070"]["recall"]),
]


def fig_scaling(session, pts, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.1))
    col = _sess_color(session)
    for ax, (label, get) in zip(axes, METRICS):
        ns, vals = _series(pts, get)
        ax.plot(ns, vals, "-o", color=col, ms=5)
        ax.set_xlabel("recordings (x 60 s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle("%s: edge recovery vs recording duration" % session, y=1.02)
    fig.tight_layout()
    p = os.path.join(out_dir, "durgrid_scaling.png")
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_calibration(session, pts, out_dir, layer="exc"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    cmap = plt.cm.viridis
    ns = list(pts)
    for i, n in enumerate(ns):
        # n_pred == 0 rows carry realized_fdr = 1.0 by construction (nothing
        # was predicted) -- drawing them would fake catastrophic miscalibration
        rows = [r for r in pts[n]["layers"][layer]["targets"]
                if r["n_pred"] > 0]
        tgt = [r["target"] for r in rows]
        rea = [r["realized_fdr"] for r in rows]
        est = [r["est_fdr"] for r in rows if np.isfinite(r["est_fdr"])]
        rea_est = [r["realized_fdr"] for r in rows if np.isfinite(r["est_fdr"])]
        c = cmap(i / max(len(ns) - 1, 1))
        axes[0].plot(tgt, rea, "-o", color=c, ms=4, label="n=%d" % n)
        axes[1].plot(est, rea_est, "o", color=c, ms=4)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("nominal FDR target")
    axes[0].set_ylabel("realized FDR (1 - precision)")
    axes[1].set_xlabel("jitter-null estimated FDR at threshold")
    axes[1].set_ylabel("realized FDR")
    axes[0].legend(fontsize=6, ncol=3, loc="upper left")
    fig.suptitle("%s: FDR calibration, %s layer (dashed = perfect)"
                 % (session, layer))
    fig.tight_layout()
    p = os.path.join(out_dir, "durgrid_calibration.png")
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_oracle(session, pts, out_dir, layer="exc"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    col = _sess_color(session)
    # one pass so all four series stay aligned on the same n-vector
    ns, bf1, af1, r10, rach = [], [], [], [], []
    for n, d in pts.items():
        try:
            vals = (d["layers"][layer]["oracle"]["best_f1"],
                    d["typed_at_070"]["f1"],
                    d["layers"][layer]["oracle"]["recall_at_true_fdr10"],
                    d["typed_at_070"]["recall"])
        except (KeyError, TypeError):
            continue
        if not all(np.isfinite(v) for v in vals):
            continue
        ns.append(n)
        bf1.append(vals[0])
        af1.append(vals[1])
        r10.append(vals[2])
        rach.append(vals[3])
    axes[0].plot(ns, bf1, "-o", color=GRAY, ms=5, label="oracle best F1")
    axes[0].plot(ns, af1, "-o", color=col, ms=5,
                 label="achieved (typed, FDR 0.70)")
    axes[0].set_ylabel("F1")
    axes[1].plot(ns, r10, "-o", color=GRAY, ms=5,
                 label="oracle recall @ true FDR 0.10")
    axes[1].plot(ns, rach, "-o", color=col, ms=5, label="achieved recall")
    axes[1].set_ylabel("recall")
    for ax in axes:
        ax.set_xlabel("recordings (x 60 s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("%s: label-free operating point vs oracle ceiling (%s)"
                 % (session, layer))
    fig.tight_layout()
    p = os.path.join(out_dir, "durgrid_oracle.png")
    fig.savefig(p, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_all_scaling(grid):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (label, get) in zip(axes, METRICS):
        curves = {"c50": [], "c40": []}
        for session, pts in grid.items():
            ns, vals = _series(pts, get)
            if not len(ns):
                continue
            col = _sess_color(session)
            ax.plot(ns, vals, "-", color=col, lw=0.9, alpha=0.45)
            grp = "c50" if "_c50_" in session else "c40"
            curves[grp].append(dict(zip(ns.tolist(), vals.tolist())))
        for grp, col in (("c50", C50), ("c40", C40)):
            if not curves[grp]:
                continue
            # mean only over n present in EVERY session of the group, so the
            # bold line never kinks from composition changes on partial grids
            all_ns = sorted(set.intersection(*[set(c) for c in curves[grp]]))
            mean = [np.mean([c[n] for c in curves[grp]]) for n in all_ns]
            ax.plot(all_ns, mean, "-", color=col, lw=3.0,
                    label="%s mean" % grp)
        ax.set_xlabel("recordings (x 60 s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("edge recovery vs duration, all %d networks" % len(grid),
                 y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT_ALL, "durgrid_all_scaling.png")
    fig.savefig(p, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_all_calibration(grid, layer="exc"):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for session, pts in grid.items():
        n = max(pts)                       # fullest duration available
        rows = [r for r in pts[n]["layers"][layer]["targets"]
                if r["n_pred"] > 0]
        ax.plot([r["target"] for r in rows],
                [r["realized_fdr"] for r in rows],
                "-o", color=_sess_color(session), ms=3.5, lw=1.0, alpha=0.6)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2,
            label="perfect calibration")
    ax.plot([], [], "-", color=C50, label="c50 sessions")
    ax.plot([], [], "-", color=C40, label="c40 sessions")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal FDR target")
    ax.set_ylabel("realized FDR (1 - precision)")
    ax.set_title("FDR calibration at full duration, %s layer" % layer)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_ALL, "durgrid_all_calibration.png")
    fig.savefig(p, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_all_oracle(grid, layer="exc"):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for session, pts in grid.items():
        n = max(pts)
        d = pts[n]
        orc = d["layers"][layer]["oracle"]["best_f1"]
        ach = d["typed_at_070"]["f1"]
        ax.plot(orc, ach, "o" if "_c50_" in session else "s",
                color=_sess_color(session), ms=8, alpha=0.85)
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], "--", color=GRAY, lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("oracle best F1 (ground-truth threshold)")
    ax.set_ylabel("achieved F1 (typed, label-free, FDR 0.70)")
    ax.set_title("achieved vs oracle at full duration, all networks")
    ax.plot([], [], "o", color=C50, label="c50")
    ax.plot([], [], "s", color=C40, label="c40")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT_ALL, "durgrid_all_oracle.png")
    fig.savefig(p, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--all-only", action="store_true")
    a = ap.parse_args()
    grid = load_grid(a.grid)
    if not grid:
        raise SystemExit("no durgrid_*.json under %s" % a.grid)
    n_pts = sum(len(v) for v in grid.values())
    print("loaded %d points across %d sessions" % (n_pts, len(grid)))
    if not a.all_only:
        for session, pts in grid.items():
            out_dir = os.path.join(DATA, session, "results", "normal",
                                   "figures")
            os.makedirs(out_dir, exist_ok=True)
            for fn in (fig_scaling, fig_calibration, fig_oracle):
                print(fn(session, pts, out_dir))
    for fn in (fig_all_scaling, fig_all_calibration, fig_all_oracle):
        print(fn(grid))


if __name__ == "__main__":
    main()
