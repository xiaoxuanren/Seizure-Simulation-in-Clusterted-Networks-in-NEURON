"""Rasters for a few flagship recordings with network bursts marked.

Marks bursts from the 0.35-gate recompute (``burstwindows_p035.npz``), coloured
by class -- IC-locked (start 4.5-5.5 s, an initialization artifact) vs
spontaneous -- and outlines which of them the shipped 0.8-gate detector actually
stored. The bottom strip of each panel is the detector's OWN input signal: the
fraction of distinct neurons firing per 5 ms bin, with the 0.05 onset line that
brackets candidate windows.

Bursts are only ~56-73 ms wide in a 60 s record, so at full scale they are marked
with a line + triangle rather than a shaded span; the second figure zooms each
burst to scale so the window edges are visible.

    python burstmarked_rasters.py --recordings 0 26 35 37 53 114

Writes ``burstmarked_rasters_overview.png`` and ``burstmarked_rasters_zoom.png``.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron_simulation.analysis import population_activity  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
IC_COLOR, SPO_COLOR = "#c0392b", "#1f5fd0"
ONSET_FRAC = 0.05          # detector's onset_active_frac
GATE = 0.35


def load_rec(idx):
    p = os.path.join(SESSION, "recording%03d.npz" % idx)
    d = np.load(p, allow_pickle=True)
    st = [np.atleast_1d(np.asarray(t, float)) for t in d["spike_times"]]
    stored = np.asarray(d["burst_windows"])
    if stored.dtype == object:
        stored = np.asarray(stored.tolist(), dtype=float)
    stored = stored.reshape(-1, stored.shape[-1]) if stored.size else np.zeros((0, 2))
    clusters = np.asarray(d["resampled_cluster_assignments"])
    return st, float(d["duration"]), stored, clusters


def bursts_for(idx, B):
    m = B["recording"] == idx
    return dict(start=B["start_ms"][m], end=B["end_ms"][m],
                part=B["participation"][m], is_ic=B["is_ic"][m],
                dur=B["duration_ms"][m])


def raster_xy(st, clusters, t0=None, t1=None):
    order = np.argsort(clusters, kind="stable")
    row = np.empty(len(order), int)
    row[order] = np.arange(len(order))
    xs, ys = [], []
    for gid, t in enumerate(st):
        if t.size == 0:
            continue
        if t0 is not None:
            t = t[(t >= t0) & (t <= t1)]
        if t.size:
            xs.append(t / 1000.0)
            ys.append(np.full(t.size, row[gid]))
    if not xs:
        return np.array([]), np.array([])
    return np.concatenate(xs), np.concatenate(ys)


def overview(recs, B, out):
    fig, axes = plt.subplots(len(recs), 1, figsize=(15, 2.9 * len(recs)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, recs):
        st, dur, stored, clusters = load_rec(idx)
        bu = bursts_for(idx, B)
        n = len(st)

        x, y = raster_xy(st, clusters)
        ax.scatter(x, y, s=0.45, c="k", marker=".", linewidths=0, rasterized=True)
        ax.set_ylim(-0.03 * n, n * 1.55)
        ax.set_ylabel("rec %03d\nneuron" % idx, fontsize=9)

        # detector's own signal, drawn into the top 18% of the panel
        _, af, _ = population_activity({i: st[i] for i in range(n)}, n, dur,
                                       bin_ms=5.0)
        tb = (np.arange(len(af)) + 0.5) * 5.0 / 1000.0
        base, height = n * 1.005, n * 0.185
        scale = height / max(af.max(), ONSET_FRAC * 2)
        ax.fill_between(tb, base, base + af * scale, color="#777", lw=0)
        ax.axhline(base + ONSET_FRAC * scale, color="#e08020", lw=0.8, ls="--")

        for s, e, p, ic, d in zip(bu["start"], bu["end"], bu["part"],
                                  bu["is_ic"], bu["dur"]):
            c = IC_COLOR if ic else SPO_COLOR
            tc = 0.5 * (s + e) / 1000.0
            ax.axvline(tc, color=c, lw=0.9, alpha=0.55, zorder=0)
            ax.plot([tc], [n * 1.155], marker="v", ms=7, color=c, clip_on=False)
            ax.annotate("%.0f%% / %.0fms" % (100 * p, d), xy=(tc, n * 1.175),
                        ha="center", va="bottom", fontsize=6.8, color=c,
                        rotation=90, annotation_clip=False)
        for s, e in stored:
            ax.plot([0.5 * (s + e) / 1000.0], [-0.02 * n], marker="^", ms=8,
                    mfc="none", mec="k", mew=1.3, clip_on=False)

        ax.set_xlim(0, dur / 1000.0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("time (s, file clock — 1 s transient already removed)")
    handles = [
        Line2D([], [], marker="v", ls="", color=IC_COLOR,
               label="IC-locked burst (4.5–5.5 s, initialization artifact)"),
        Line2D([], [], marker="v", ls="", color=SPO_COLOR,
               label="spontaneous burst"),
        Line2D([], [], marker="^", ls="", mfc="none", mec="k",
               label="stored 0.8-gate window (what shipped)"),
        Line2D([], [], color="#777", lw=4, label="active fraction / 5 ms bin"),
        Line2D([], [], color="#e08020", ls="--", label="onset threshold 0.05"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2,
                   framealpha=0.9)
    fig.suptitle("Flagship recordings with network bursts marked — 0.35 gate "
                 "(participation > 35%% of 926 distinct neurons over the event "
                 "window)\nlabels show participation and window duration",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out, dpi=145, facecolor="white", bbox_inches="tight")
    print("saved ->", out, flush=True)


def zoom(recs, B, out, pad_ms=250.0):
    rows = []
    for idx in recs:
        bu = bursts_for(idx, B)
        for k in range(len(bu["start"])):
            rows.append((idx, bu["start"][k], bu["end"][k], bu["part"][k],
                         bu["is_ic"][k], bu["dur"][k]))
    ncol = 4
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                             squeeze=False)
    cache = {}
    for ax, (idx, s, e, p, ic, d) in zip(axes.ravel(), rows):
        if idx not in cache:
            cache[idx] = load_rec(idx)
        st, dur, stored, clusters = cache[idx]
        n = len(st)
        x, y = raster_xy(st, clusters, s - pad_ms, e + pad_ms)
        ax.scatter(x * 1000.0 - s, y, s=1.2, c="k", marker=".", linewidths=0)
        c = IC_COLOR if ic else SPO_COLOR
        ax.axvspan(0, e - s, color=c, alpha=0.16, lw=0)
        ax.axvline(0, color=c, lw=1.0)
        ax.axvline(e - s, color=c, lw=1.0)
        was_stored = any(abs(0.5 * (bs + be) - 0.5 * (s + e)) < 100.0
                         for bs, be in stored)
        ax.set_title("rec %03d @ %.2f s  %s\npart %.0f%%  dur %.0f ms%s"
                     % (idx, s / 1000.0, "IC-locked" if ic else "spontaneous",
                        100 * p, d, "  [stored@0.8]" if was_stored else ""),
                     fontsize=8.5, color=c)
        ax.set_xlim(-pad_ms, (e - s) + pad_ms)
        ax.set_ylim(0, n)
        ax.set_xlabel("ms from window start", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Same bursts, zoomed to scale — shaded band is the detector's "
                 "window (data-defined: contiguous 5 ms bins over 5%% active, "
                 "merged across <50 ms, padded ±10 ms)", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=145, facecolor="white", bbox_inches="tight")
    print("saved ->", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", type=int, nargs="+",
                    default=[0, 26, 35, 37, 53, 114])
    a = ap.parse_args()
    B = np.load(os.path.join(SESSION, "burstwindows_p035.npz"), allow_pickle=True)
    here = os.path.dirname(os.path.abspath(__file__))
    overview(a.recordings, B, os.path.join(here, "burstmarked_rasters_overview.png"))
    zoom(a.recordings[:3], B, os.path.join(here, "burstmarked_rasters_zoom.png"))


if __name__ == "__main__":
    main()
