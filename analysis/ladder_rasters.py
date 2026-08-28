"""Render a burst-marked raster for every mechanism-ladder point.

Reads the spike npz files produced by chtc/ladder_one.py --save-spikes and
renders, per (network, ladder point): randomized-row raster, burst duration
bars above the plot (red = full, green = partial) labelled with participation,
and a stats box in the LOWER RIGHT reporting firing rate, mean burst
participation, and burst rate.

    python analysis/ladder_rasters.py --src <dir of *_spikes.npz> \
        [--out <dir>]     # default: sweep_summary/ladder_rasters/
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from session_paths import DATA  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)

PRETTY = {"fourap": "4-AP (A-current block)", "sahp": "sAHP deficit",
          "kclear": "impaired K+ clearance", "both": "K+ clearance + sAHP seizure"}


def render(npz_path, out_dir):
    d = np.load(npz_path, allow_pickle=True)
    st = d["spike_times"]
    n = int(d["n_neurons"])
    duration = float(d["duration"])
    session = str(d["session"])
    label = str(d["label"])
    overrides = json.loads(str(d["overrides"]))
    spike_data = {i: np.asarray(st[i], float) for i in range(n)}

    bursts, _meta = _an.detect_network_bursts_all(spike_data, n, duration)
    parts = [b["participation"] for b in bursts]
    n_spikes = sum(len(s) for s in spike_data.values())
    rate_hz = n_spikes / n / (duration / 1000.0)
    burst_rate = len(bursts) / (duration / 60000.0)
    n_full = sum(1 for b in bursts if b["burst_class"] == "full")

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in range(n):
        s = spike_data[i]
        if s.size:
            ax.plot(s / 1000.0, np.full(s.size, order[i]), ".", ms=1.0,
                    color="#1f5fd0", alpha=0.55)
    y_bar, y_txt = n * 1.03, n * 1.055
    for b in bursts:
        col = "#c0392b" if b["burst_class"] == "full" else "#2e8b57"
        ax.plot([b["start_ms"] / 1000.0, b["end_ms"] / 1000.0], [y_bar, y_bar],
                "-", color=col, lw=4, solid_capstyle="butt", clip_on=False)
        if len(bursts) <= 40:
            ax.text(0.5 * (b["start_ms"] + b["end_ms"]) / 1000.0, y_txt,
                    "%.2f" % b["participation"], ha="center", va="bottom",
                    fontsize=6, color=col, clip_on=False)

    stats = ("firing rate       %.2f Hz\n"
             "burst rate        %.1f /min\n"
             "participation     %.2f\n"
             "burst duration    %.0f ms\n"
             "bursts            %d full / %d partial"
             % (rate_hz, burst_rate,
                float(np.mean(parts)) if parts else float("nan"),
                float(np.mean([b["duration_ms"] for b in bursts])) if bursts else float("nan"),
                n_full, len(bursts) - n_full))
    ax.text(0.995, 0.02, stats, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#888888", alpha=0.92))

    fam = label.split("_")[0]
    ov = ", ".join("%s=%g" % (k, v) for k, v in sorted(overrides.items()))
    ax.set_xlim(0, duration / 1000.0)
    ax.set_ylim(0, n * 1.10)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([t for t in ax.get_yticks() if t <= n])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron (randomized)")
    ax.set_title("%s -- %s  [%s]\n%s" % (session, PRETTY.get(fam, fam), label, ov),
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, "%s__%s.png" % (session, label))
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=os.path.join(DATA, "sweep_summary", "ladder_rasters"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.src, "**", "*_spikes.npz"), recursive=True))
    if not paths:
        raise SystemExit("no *_spikes.npz under %s" % a.src)
    for i, p in enumerate(paths, 1):
        out = render(p, a.out)
        if i % 20 == 0 or i == len(paths):
            print("  %d/%d -> %s" % (i, len(paths), os.path.basename(out)), flush=True)
    print("%d rasters -> %s" % (len(paths), a.out))


if __name__ == "__main__":
    main()
