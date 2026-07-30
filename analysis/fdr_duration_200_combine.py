"""Merge the FDR-target x duration shards and draw the two requested views.

    (1) realized-vs-nominal, one curve per duration  -- calibration
    (2) performance-vs-duration, one curve per target -- operating-point choice

Writes ``fdrdur10to200_metrics.json``, ``fdrdur10to200_calibration.png`` and
``fdrdur10to200_performance.png`` into the session directory. The existing
5-100 min sweep outputs are untouched.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burstexcl_glm_arm import SESSION  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARTS = os.path.join(SESSION, "fdrdur10to200_parts")


def main():
    files = sorted(glob.glob(os.path.join(PARTS, "fdrdur_n*.json")))
    parts = [json.load(open(f)) for f in files]
    parts.sort(key=lambda p: p["n_rec"])
    sizes = [p["n_rec"] for p in parts]
    targets = [r["target"] for r in parts[0]["rows"]]
    n_true = parts[0]["n_true_exc"]
    print("durations: %s" % sizes, flush=True)
    print("targets:   %s" % targets, flush=True)

    keys = ("realized_fdr", "estimated_fdr", "precision", "recall", "f1",
            "TP", "n_pred", "thr")
    G = {k: np.full((len(targets), len(sizes)), np.nan) for k in keys}
    for si, p in enumerate(parts):
        for ti, r in enumerate(p["rows"]):
            for k in keys:
                G[k][ti, si] = r[k]

    json.dump(dict(sizes=sizes, targets=targets, n_true_exc=n_true,
                   grids={k: v.tolist() for k, v in G.items()}),
              open(os.path.join(SESSION, "fdrdur10to200_metrics.json"), "w"),
              indent=2)

    x = np.array(sizes, float)
    tg = np.array(targets, float)
    dcol = plt.cm.plasma(np.linspace(0, 0.9, len(sizes)))
    tcol = plt.cm.viridis(np.linspace(0, 1, len(targets)))

    # ---------- view 1: realized vs nominal, per duration ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for si, n in enumerate(sizes):
        ax[0].plot(tg, G["realized_fdr"][:, si], "-o", ms=3.5, lw=1.3,
                   color=dcol[si], label="%d rec" % n)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax[0].set_xlabel("nominal target FDR")
    ax[0].set_ylabel("realized FDR")
    ax[0].set_title("(a) realized vs nominal FDR, per duration")
    ax[0].legend(fontsize=7, ncol=2)

    for si, n in enumerate(sizes):
        ax[1].plot(tg, G["realized_fdr"][:, si] - tg, "-o", ms=3.5, lw=1.3,
                   color=dcol[si])
    ax[1].axhline(0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("nominal target FDR")
    ax[1].set_ylabel("realized - nominal")
    ax[1].set_title("(b) calibration error (negative = conservative)")

    pm = ax[2].pcolormesh(x, tg, G["realized_fdr"], cmap="RdYlGn_r",
                          shading="nearest")
    fig.colorbar(pm, ax=ax[2], fraction=0.046, pad=0.04, label="realized FDR")
    cs = ax[2].contour(x, tg, G["realized_fdr"], levels=[0.05, 0.10, 0.20],
                       colors="k", linewidths=1.2)
    ax[2].clabel(cs, fontsize=8, fmt="%.2f")
    ax[2].set_xlabel("recordings (= minutes)")
    ax[2].set_ylabel("nominal target FDR")
    ax[2].set_title("(c) realized FDR over (target x duration)")

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("Label-free jitter-FDR calibration - sum4, 10-200 recordings x "
                 "nominal target 0.1-1.0\nthe null is conservative: realized FDR "
                 "sits far below nominal at every duration", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(SESSION, "fdrdur10to200_calibration.png")
    fig.savefig(p1, dpi=140, facecolor="white", bbox_inches="tight")

    # ---------- view 2: performance vs duration, per target ----------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("f1", "F1 (excitatory)"), ("recall", "recall"),
              ("precision", "precision"), ("realized_fdr", "realized FDR"),
              ("TP", "true positives"), ("n_pred", "predicted edges")]
    for a, (k, title) in zip(ax.ravel(), panels):
        for ti, t in enumerate(targets):
            a.plot(x, G[k][ti], "-o", ms=3, lw=1.3, color=tcol[ti])
        a.set_xlabel("recordings (= minutes)")
        a.set_ylabel(title)
        a.set_title(title + " vs duration")
        a.grid(alpha=0.25)
    ax.ravel()[4].axhline(n_true, color="gray", ls=":", lw=1)
    ax.ravel()[4].text(x[-1], n_true, " all true exc (%d)" % n_true,
                       fontsize=8, va="bottom", ha="right", color="gray")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(min(targets), max(targets)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("nominal target FDR")
    fig.suptitle("Label-free recovery vs duration, one curve per nominal FDR "
                 "target (10-200 recordings)", fontsize=13, fontweight="bold")
    p2 = os.path.join(SESSION, "fdrdur10to200_performance.png")
    fig.savefig(p2, dpi=140, facecolor="white", bbox_inches="tight")

    # ---------- text summaries ----------
    print("\n--- realized FDR (rows = nominal target, cols = recordings) ---",
          flush=True)
    print("target " + "".join("%8d" % n for n in sizes), flush=True)
    for ti, t in enumerate(targets):
        print("%6.1f " % t + "".join("%8.4f" % v for v in G["realized_fdr"][ti]),
              flush=True)

    print("\n--- F1 (rows = nominal target, cols = recordings) ---", flush=True)
    print("target " + "".join("%8d" % n for n in sizes), flush=True)
    for ti, t in enumerate(targets):
        print("%6.1f " % t + "".join("%8.4f" % v for v in G["f1"][ti]), flush=True)

    print("\n--- per duration: best-F1 target, and target nearest realized 0.10 ---",
          flush=True)
    for si, n in enumerate(sizes):
        bf = int(np.nanargmax(G["f1"][:, si]))
        j = int(np.nanargmin(np.abs(G["realized_fdr"][:, si] - 0.10)))
        print("  %3d rec: best-F1 target %.1f (F1 %.4f, realFDR %.4f, R %.4f) | "
              "realized~0.10 at target %.1f (realFDR %.4f)"
              % (n, targets[bf], G["f1"][bf, si], G["realized_fdr"][bf, si],
                 G["recall"][bf, si], targets[j], G["realized_fdr"][j, si]),
              flush=True)

    print("\n--- calibration gap (realized - nominal), mean over durations ---",
          flush=True)
    for ti, t in enumerate(targets):
        row = G["realized_fdr"][ti]
        print("  target %.1f: realized %.4f +/- %.4f  (gap %+.4f)"
              % (t, np.nanmean(row), np.nanstd(row), np.nanmean(row) - t),
              flush=True)

    print("\nsaved -> fdrdur10to200_metrics.json", flush=True)
    print("         %s" % os.path.basename(p1), flush=True)
    print("         %s" % os.path.basename(p2), flush=True)


if __name__ == "__main__":
    main()
