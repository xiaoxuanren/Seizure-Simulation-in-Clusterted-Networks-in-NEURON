"""Cross-network distance-resolved recovery, from the 40 per-session JSONs.

Reads results/<state>/glm/distance_recovery.json for every sweep session and
both states (written by glm_distance_recovery.py) and renders
sweep_summary/distance_recovery.png:

  (a) recall and precision vs distance, mean across networks (normal state),
      per-network curves faint behind
  (b) recall within- vs between-cluster (normal)
  (c) mean |W| for true edges and for non-edges vs distance (normal) -- the
      short-range rise of the NON-edge curve is the common-input signature
  (d) the non-edge curve, normal vs seizure, same axes

Sessions have session-specific bin grids; curves are interpolated onto a
common distance grid and averaged where a session has support. The mean
curves are persisted to sweep_summary/distance_recovery.json.

    python analysis/sweep_distance_recovery.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA  # noqa: E402
import thesis_style as st  # noqa: E402

st.apply()
OUT = os.path.join(DATA, "sweep_summary")
GRIDX = np.linspace(0.0, 16.0, 33)
MIN_SESSIONS = 10          # a mean point needs this many sessions' support


def load(state):
    out = {}
    for p in sorted(glob.glob(os.path.join(
            DATA, "sweep_*", "results", state, "glm",
            "distance_recovery.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out[d["session"]] = d
    return out


def onto_grid(d, key):
    """Interpolate one session's binned curve onto GRIDX; NaN out of range."""
    xc = np.asarray(d["bin_centers"], float)
    y = np.asarray(d[key], float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(GRIDX, np.nan)
    yi = np.interp(GRIDX, xc[ok], y[ok])
    yi[(GRIDX < xc[ok][0]) | (GRIDX > xc[ok][-1])] = np.nan
    return yi


def stack(sessions, key):
    return np.array([onto_grid(d, key) for d in sessions.values()])


def meanline(arr):
    n = np.isfinite(arr).sum(0)
    m = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), 0)
    m[n < MIN_SESSIONS] = np.nan
    return m


def main():
    normal = load("normal")
    seizure = load("seizure")
    print("normal: %d sessions | seizure: %d" % (len(normal), len(seizure)))

    fig, axes = plt.subplots(2, 2, figsize=(st.FIGW, 4.6),
                             gridspec_kw=dict(wspace=0.32, hspace=0.52,
                                              left=0.10, right=0.975,
                                              top=0.94, bottom=0.10))
    (axa, axb), (axc, axd) = axes

    rec = stack(normal, "recall")
    prc = stack(normal, "precision")
    for row in rec:
        axa.plot(GRIDX, row, "-", color=st.TP, lw=0.5, alpha=0.22)
    for row in prc:
        axa.plot(GRIDX, row, "-", color=st.C50, lw=0.5, alpha=0.22)
    axa.plot(GRIDX, meanline(rec), "-", color=st.TP, lw=1.6, label="recall")
    axa.plot(GRIDX, meanline(prc), "-", color=st.C50, lw=1.6,
             label="precision")
    axa.set_xlabel("inter-neuron distance (space units)")
    axa.set_ylabel("rate")
    axa.set_ylim(0, 1.02)
    axa.legend(loc="lower left")
    st.lineax(axa)
    st.letter(axa, "a", dx=-0.16)

    rw = stack(normal, "recall_within")
    rb = stack(normal, "recall_between")
    for row in rw:
        axb.plot(GRIDX, row, "-", color=st.C40, lw=0.5, alpha=0.22)
    for row in rb:
        axb.plot(GRIDX, row, "-", color=st.C50, lw=0.5, alpha=0.22)
    axb.plot(GRIDX, meanline(rw), "-", color=st.C40, lw=1.6,
             label="within-cluster")
    axb.plot(GRIDX, meanline(rb), "-", color=st.C50, lw=1.6,
             label="between-cluster")
    axb.set_xlabel("inter-neuron distance (space units)")
    axb.set_ylabel("recall")
    axb.set_ylim(0, 1.02)
    axb.legend(loc="lower left")
    st.lineax(axb)
    st.letter(axb, "b", dx=-0.16)

    wt = stack(normal, "mean_absW_true")
    wn = stack(normal, "mean_absW_nonedge")
    for row in wt:
        axc.plot(GRIDX, row, "-", color=st.TP, lw=0.5, alpha=0.22)
    for row in wn:
        axc.plot(GRIDX, row, "-", color="0.45", lw=0.5, alpha=0.22)
    axc.plot(GRIDX, meanline(wt), "-", color=st.TP, lw=1.6,
             label="true edges")
    axc.plot(GRIDX, meanline(wn), "-", color="0.3", lw=1.6,
             label="non-edges (background)")
    axc.set_xlabel("inter-neuron distance (space units)")
    axc.set_ylabel("mean |W|")
    axc.set_yscale("log")
    axc.legend(loc="upper right")
    st.lineax(axc)
    st.letter(axc, "c", dx=-0.16)

    wns = stack(seizure, "mean_absW_nonedge")
    for row in wn:
        axd.plot(GRIDX, row, "-", color="0.45", lw=0.5, alpha=0.18)
    for row in wns:
        axd.plot(GRIDX, row, "-", color=st.C40, lw=0.5, alpha=0.18)
    axd.plot(GRIDX, meanline(wn), "-", color="0.3", lw=1.6, label="normal")
    axd.plot(GRIDX, meanline(wns), "-", color=st.C40, lw=1.6,
             label="seizure")
    axd.set_xlabel("inter-neuron distance (space units)")
    axd.set_ylabel("non-edge mean |W|")
    axd.set_yscale("log")
    axd.legend(loc="upper right")
    st.lineax(axd)
    st.letter(axd, "d", dx=-0.16)

    p = os.path.join(OUT, "distance_recovery.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)

    out = dict(
        grid=GRIDX.tolist(), min_sessions=MIN_SESSIONS,
        normal=dict(
            recall=meanline(rec).tolist(),
            precision=meanline(prc).tolist(),
            recall_within=meanline(rw).tolist(),
            recall_between=meanline(rb).tolist(),
            mean_absW_true=meanline(wt).tolist(),
            mean_absW_nonedge=meanline(wn).tolist()),
        seizure=dict(mean_absW_nonedge=meanline(wns).tolist()),
        per_session={s: dict(near_half_recall=d["near_half_recall"],
                             far_half_recall=d["far_half_recall"],
                             median_true_distance=d["median_true_distance"])
                     for s, d in normal.items()})
    q = os.path.join(OUT, "distance_recovery.json")
    with open(q, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(q)
    nh = [d["near_half_recall"] for d in normal.values()]
    fh_ = [d["far_half_recall"] for d in normal.values()]
    print("normal near-half recall mean %.3f | far-half %.3f"
          % (np.mean(nh), np.mean(fh_)))


if __name__ == "__main__":
    main()
