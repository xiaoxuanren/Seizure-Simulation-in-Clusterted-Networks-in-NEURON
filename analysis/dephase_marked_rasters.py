"""Dephased-IC pilot rasters with bursts marked, plus the settling diagnostic.

Three figures:
  A  all 5 dephased recordings, full 60 s, bursts marked (0.35 gate)
  B  zoomed to scale: the t~0 event in rec003 next to an ordinary spontaneous burst
  C  the diagnostic -- SIM-time raster over the first 2 s for the snapshot that
     bursts (3) and one that does not (0), with the summed synaptic conductances
     overlaid and the 1 s discard boundary marked

Figure C is the evidence for what causes the t~0 event. Note the x-axis of C is
SIMULATION time; recordings discard the first 1000 ms, so sim t = 1020 ms is
file-clock t = 20 ms.
"""

import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from neuron_simulation.analysis import detect_network_bursts  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEPHASED = os.path.join(REPO, "notebooks", "NEURON data parallel", "dephased_ic")
DIAG = os.path.join(HERE, "dephase_settling_diagnostic.npz")
GATE = 0.35
FLAG_IC_BAND = (4.60, 5.34)


def load_all():
    fs = sorted(glob.glob(os.path.join(DEPHASED, "recording*.npz")))
    recs = []
    for p in fs:
        d = np.load(p, allow_pickle=True)
        st = [np.atleast_1d(np.asarray(t, float)) for t in d["spike_times"]]
        n = len(st)
        b = detect_network_bursts({j: st[j] for j in range(n)}, n,
                                 float(d["duration"]),
                                 participation_threshold=GATE, burn_in_ms=0.0)
        recs.append(dict(name=os.path.basename(p), st=st, n=n, bursts=b,
                         snap=int(d["snapshot_index"]),
                         dur=float(d["duration"])))
    return recs


def xy(st, t0=None, t1=None):
    xs, ys = [], []
    for gid, t in enumerate(st):
        if t.size == 0:
            continue
        if t0 is not None:
            t = t[(t >= t0) & (t <= t1)]
        if t.size:
            xs.append(t)
            ys.append(np.full(t.size, gid))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def fig_a(recs):
    fig, axes = plt.subplots(len(recs), 1, figsize=(15, 2.6 * len(recs)),
                            sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, recs):
        x, y = xy(r["st"])
        ax.scatter(x / 1000.0, y, s=0.45, c="k", marker=".", linewidths=0,
                   rasterized=True)
        ax.axvspan(*FLAG_IC_BAND, color="#c0392b", alpha=0.10, lw=0)
        for b in r["bursts"]:
            tc = 0.5 * (b["start_ms"] + b["end_ms"]) / 1000.0
            early = b["start_ms"] < 200.0
            c = "#e67e22" if early else "#1f5fd0"
            ax.axvline(tc, color=c, lw=1.0, alpha=0.7, zorder=0)
            ax.plot([tc], [r["n"] * 1.04], marker="v", ms=8, color=c,
                    clip_on=False)
            ax.annotate("%.0f%% / %.0fms%s" % (100 * b["participation"],
                                               b["duration_ms"],
                                               "  <- t~0 EVENT" if early else ""),
                        xy=(tc, r["n"] * 1.07), ha="center", va="bottom",
                        fontsize=7.5, color=c, rotation=90,
                        annotation_clip=False)
        ax.set_ylim(0, r["n"] * 1.45)
        ax.set_xlim(0, r["dur"] / 1000.0)
        ax.set_ylabel("%s\nsnap %d" % (r["name"].replace(".npz", ""), r["snap"]),
                      fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time (s, file clock — 1 s already discarded)")
    fig.suptitle("Dephased-IC pilot: 5 recordings, bursts marked at the 0.35 gate\n"
                 "shaded band = where the flagship's initialization burst sat "
                 "(4.60–5.34 s) — now empty. Orange = the residual t~0 event.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = os.path.join(HERE, "dephase_marked_rasters_A_overview.png")
    fig.savefig(p, dpi=145, facecolor="white")
    print("saved ->", os.path.basename(p), flush=True)


def fig_b(recs):
    picks = []
    for r in recs:
        for b in r["bursts"]:
            picks.append((r, b, b["start_ms"] < 200.0))
    picks = ([p for p in picks if p[2]] + [p for p in picks if not p[2]])[:4]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.2),
                            squeeze=False)
    for ax, (r, b, early) in zip(axes[0], picks):
        pad = 250.0
        x, y = xy(r["st"], b["start_ms"] - pad, b["end_ms"] + pad)
        ax.scatter(x - b["start_ms"], y, s=1.4, c="k", marker=".", linewidths=0)
        c = "#e67e22" if early else "#1f5fd0"
        ax.axvspan(0, b["end_ms"] - b["start_ms"], color=c, alpha=0.16, lw=0)
        ax.set_title("%s (snap %d) @ %.2f s\n%s  part %.0f%%  dur %.0f ms"
                     % (r["name"].replace(".npz", ""), r["snap"],
                        b["start_ms"] / 1000.0,
                        "t~0 EVENT" if early else "spontaneous",
                        100 * b["participation"], b["duration_ms"]),
                     fontsize=8.5, color=c)
        ax.set_xlabel("ms from window start", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same events zoomed to scale — the t~0 event is "
                 "indistinguishable in shape from a spontaneous burst",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(HERE, "dephase_marked_rasters_B_zoom.png")
    fig.savefig(p, dpi=145, facecolor="white")
    print("saved ->", os.path.basename(p), flush=True)


def fig_c():
    d = np.load(DIAG)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7),
                             gridspec_kw={"height_ratios": [2, 1]})
    for col, (lab, title) in enumerate((
            ("snap3_burst", "snapshot 3 — DOES burst"),
            ("snap0_control", "snapshot 0 — does not"))):
        t = d[lab + "_spike_t"]
        i = d[lab + "_spike_i"]
        ax = axes[0, col]
        ax.scatter(t, i, s=1.6, c="k", marker=".", linewidths=0)
        ax.axvline(1000, color="#c0392b", lw=1.6, ls="--",
                   label="discard boundary (file t = 0)")
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 926)
        ax.set_ylabel("neuron")
        ax.set_title("%s\n(x-axis is SIMULATION time; recordings keep t > 1000 ms)"
                     % title, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

        ax2 = axes[1, col]
        tt = d[lab + "__t"]
        ax2.plot(tt, d[lab + "__g_exc"], color="#2e8b57", lw=1.4,
                 label="summed g_exc (AMPA+NMDA)")
        ax2.set_ylabel("summed g_exc (uS)", color="#2e8b57")
        ax2.tick_params(axis="y", labelcolor="#2e8b57")
        ax3 = ax2.twinx()
        ax3.plot(tt, d[lab + "__g_inh"], color="#8e44ad", lw=1.4,
                 label="summed g_inh")
        ax3.set_ylabel("summed g_inh (uS)", color="#8e44ad")
        ax3.tick_params(axis="y", labelcolor="#8e44ad")
        ax2.axvline(1000, color="#c0392b", lw=1.6, ls="--")
        ax2.set_xlim(0, 2000)
        ax2.set_xlabel("simulation time (ms)")
        ax2.spines[["top"]].set_visible(False)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.suptitle("What causes the t~0 event: synaptic state is NOT restored, so "
                 "every recording rebuilds recurrent conductance from zero.\n"
                 "The burst lands at sim ~1030 ms — right at the discard boundary "
                 "— while g_exc is still climbing and depression resource R = 1.",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = os.path.join(HERE, "dephase_marked_rasters_C_diagnostic.png")
    fig.savefig(p, dpi=145, facecolor="white")
    print("saved ->", os.path.basename(p), flush=True)


if __name__ == "__main__":
    recs = load_all()
    for r in recs:
        print("%s snap=%d: %d bursts" % (r["name"], r["snap"], len(r["bursts"])),
              flush=True)
    fig_a(recs)
    fig_b(recs)
    fig_c()
