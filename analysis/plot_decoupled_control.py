"""Side-by-side figure for the zero-recurrence control.

Left column: flagship coupled network. Right column: identical network with
exc_weight_scale = inh_weight_scale = 0. Same topology, same per-cell Poisson
streams, same initial conditions, transient KEPT.

Three rows, shared time axis:
    raster (cluster-sorted)  -- is there a burst?
    participation (% of cells spiking per 50 ms) -- how many cells join it?
    population Vm (mean, p10-p90 band) -- the slow sAHP recovery ramp

Run ``decoupled_control.py --coupled 0`` and ``--coupled 1`` first.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    return np.load(os.path.join(HERE, "decoupled_control_%s.npz" % tag),
                   allow_pickle=True)


def panel(axes, d, title, tmax_s, color):
    ax_r, ax_p, ax_v = axes

    spikes = d["spike_times"]
    clusters = d["cluster_assignments"]
    order = np.argsort(clusters, kind="stable")
    row_of = np.empty(len(order), dtype=int)
    row_of[order] = np.arange(len(order))

    xs, ys = [], []
    for gid, times in enumerate(spikes):
        t = np.asarray(times, dtype=float) / 1000.0
        t = t[t <= tmax_s]
        if t.size:
            xs.append(t)
            ys.append(np.full(t.size, row_of[gid]))
    if xs:
        ax_r.scatter(np.concatenate(xs), np.concatenate(ys), s=0.6,
                     c="k", marker=".", linewidths=0, rasterized=True)
    ax_r.set_ylim(0, len(spikes))
    ax_r.set_ylabel("neuron\n(cluster-sorted)")
    ax_r.set_title(title, fontsize=11)

    edges = d["part_edges"] / 1000.0
    part = d["part"] * 100.0
    keep = edges <= tmax_s
    ax_p.fill_between(edges[keep], 0, part[keep], color=color, lw=0)
    ax_p.set_ylabel("participation\n(% cells / 50 ms)")
    peak = part[keep].max()
    ax_p.set_ylim(0, max(5.0, peak * 1.15))
    j = int(np.argmax(part[keep]))
    ax_p.annotate("peak %.0f%% @ %.2f s" % (peak, edges[keep][j]),
                  xy=(edges[keep][j], peak), xytext=(6, -2),
                  textcoords="offset points", fontsize=8, color=color)

    vt = d["vm_times"] / 1000.0
    keep = vt <= tmax_s
    ax_v.fill_between(vt[keep], d["vm_p10"][keep], d["vm_p90"][keep],
                      color=color, alpha=0.25, lw=0)
    ax_v.plot(vt[keep], d["vm_mean"][keep], color=color, lw=0.9)
    ax_v.set_ylabel("population Vm (mV)")
    ax_v.set_xlabel("time (s)")

    for ax in axes:
        ax.set_xlim(0, tmax_s)
        ax.spines[["top", "right"]].set_visible(False)


def make(tmax_s, fname):
    co, de = load("coupled"), load("decoupled")
    fig, ax = plt.subplots(3, 2, figsize=(13, 8.5), sharex="col",
                           gridspec_kw={"height_ratios": [2.2, 1, 1]})
    panel(ax[:, 0], co,
          "coupled (flagship: exc x2.0, inh x2.5)   rate %.3f Hz" % co["rate"],
          tmax_s, "#1f4e79")
    panel(ax[:, 1], de,
          "decoupled (exc x0, inh x0)   rate %.3f Hz" % de["rate"],
          tmax_s, "#b03030")

    # Match Vm axes so the ramp amplitudes are visually comparable.
    lo = min(a.get_ylim()[0] for a in ax[2, :])
    hi = max(a.get_ylim()[1] for a in ax[2, :])
    for a in ax[2, :]:
        a.set_ylim(lo, hi)

    fig.suptitle("Zero-recurrence control -- same topology, same noise seeds, "
                 "same initial conditions, transient kept", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=160)
    print("saved ->", out)


if __name__ == "__main__":
    make(20.0, "decoupled_control_first20s.png")
    make(60.0, "decoupled_control_full60s.png")
