"""Re-detect network bursts (full AND partial) from saved recordings and save
per-burst statistics.

Uses neuron_simulation.analysis.detect_network_bursts_all: the legacy Stage-1
bracketing with a STATISTICAL acceptance gate (background median + k*MAD, or
the self-tuning "valley" mode) instead of the fixed 0.35 participation cut,
so low-participation ("partial") bursts are kept and classified. The legacy
detector and every stat computed from it remain untouched.

Writes results/<state>/bursts/burst_stats.json per session: per-recording
burst tables (start/end/duration/participation/peak fraction/spike count/
class) plus session aggregates. --mark-raster renders a validation raster
with detected windows shaded (red = full, green = partial).

    python analysis/burst_stats.py                          # all sweep_* sessions
    python analysis/burst_stats.py sweep_c40_seed20 --mark-raster 3
    DATASET_STATE=seizure python analysis/burst_stats.py    # seizure state
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from session_paths import resolve, results_dir, list_sessions  # noqa: E402

# load analysis.py directly by path: the package __init__ imports NEURON,
# which this numpy-only script does not need
_spec = importlib.util.spec_from_file_location(
    "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)

STATE = os.environ.get("DATASET_STATE", "normal")


def recording_paths(sd):
    return [p for p in sorted(glob.glob(os.path.join(sd, "recording*.npz")))
            if "raster" not in os.path.basename(p) and "voltage" not in os.path.basename(p)]


def detect_one(path, args):
    d = np.load(path, allow_pickle=True)
    st = d["spike_times"]
    spike_data = {i: np.asarray(st[i], float) for i in range(len(st))}
    duration = float(d["duration"])
    bursts, meta = _an.detect_network_bursts_all(
        spike_data, len(st), duration,
        min_participation=args.min_participation,
        significance_k=args.k, threshold_mode=args.mode,
        onset_active_frac=args.onset, merge_gap_ms=args.merge_gap)
    return spike_data, len(st), duration, bursts, meta


def mark_raster(spike_data, n, duration, bursts, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in range(n):
        s = spike_data[i]
        if s.size:
            ax.plot(s / 1000.0, np.full(s.size, order[i]), ".", ms=1.2,
                    color="#1f5fd0", alpha=0.6)
    # burst duration bars in a strip above the raster (red = full,
    # green = partial), each labeled with its participation rate
    y_bar, y_txt = n * 1.03, n * 1.055
    for b in bursts:
        col = "#c0392b" if b["burst_class"] == "full" else "#2e8b57"
        ax.plot([b["start_ms"] / 1000.0, b["end_ms"] / 1000.0], [y_bar, y_bar],
                "-", color=col, lw=5, solid_capstyle="butt", clip_on=False)
        ax.text(0.5 * (b["start_ms"] + b["end_ms"]) / 1000.0, y_txt,
                "%.2f" % b["participation"], ha="center", va="bottom",
                fontsize=7, color=col, clip_on=False)
    ax.set_xlim(0, duration / 1000.0)
    ax.set_ylim(0, n * 1.10)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([t for t in ax.get_yticks() if t <= n])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron (randomized)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor="white")
    plt.close(fig)


def run_session(session, args):
    sd = resolve(session, STATE)
    out_dir = results_dir(session, STATE, "bursts")
    per_rec, all_bursts = [], []
    for path in recording_paths(sd):
        idx = int(os.path.basename(path)[9:12])
        spike_data, n, duration, bursts, meta = detect_one(path, args)
        per_rec.append(dict(
            index=idx,
            n_bursts=len(bursts),
            n_full=sum(1 for b in bursts if b["burst_class"] == "full"),
            n_partial=sum(1 for b in bursts if b["burst_class"] == "partial"),
            threshold=meta.get("threshold_used"),
            threshold_mode=meta.get("threshold_mode"),
            bursts=bursts))
        all_bursts.extend(bursts)
        if args.mark_raster is not None and idx == args.mark_raster:
            figdir = results_dir(session, STATE, "figures")
            out_png = os.path.join(figdir, "burstmarked_all_rec%03d.png" % idx)
            mark_raster(
                spike_data, n, duration, bursts, out_png,
                "%s %s rec %03d -- detect_all (%s, k=%g): %d full (red) + %d partial (green)"
                % (session, STATE, idx, meta["threshold_mode"], args.k,
                   per_rec[-1]["n_full"], per_rec[-1]["n_partial"]))
            print("  marked raster -> %s" % out_png)

    n_rec = len(per_rec)
    durations = np.array([b["duration_ms"] for b in all_bursts])
    parts = np.array([b["participation"] for b in all_bursts])
    summary = dict(
        session=session, state=STATE, n_recordings=n_rec,
        detector="detect_network_bursts_all",
        params=dict(mode=args.mode, significance_k=args.k,
                    min_participation=args.min_participation),
        n_bursts_total=len(all_bursts),
        n_full=int(sum(1 for b in all_bursts if b["burst_class"] == "full")),
        n_partial=int(sum(1 for b in all_bursts if b["burst_class"] == "partial")),
        bursts_per_rec=len(all_bursts) / max(1, n_rec),
        mean_participation=float(parts.mean()) if parts.size else None,
        median_participation=float(np.median(parts)) if parts.size else None,
        mean_duration_ms=float(durations.mean()) if durations.size else None,
        median_duration_ms=float(np.median(durations)) if durations.size else None,
        per_recording=per_rec)
    out = os.path.join(out_dir, "burst_stats.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("%s/%s: %d bursts (%d full + %d partial) in %d recs | "
          "%.2f/rec | participation %.2f | duration %.0f ms -> %s"
          % (session, STATE, summary["n_bursts_total"], summary["n_full"],
             summary["n_partial"], n_rec, summary["bursts_per_rec"],
             summary["mean_participation"] or float("nan"),
             summary["mean_duration_ms"] or float("nan"), out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", default=None)
    ap.add_argument("--mode", choices=["mad", "valley"], default="mad")
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--min-participation", type=float, default=0.10)
    ap.add_argument("--onset", type=float, default=0.01,
                    help="Stage-1 bracketing active fraction per 5 ms bin")
    ap.add_argument("--merge-gap", type=float, default=100.0)
    ap.add_argument("--mark-raster", type=int, default=None,
                    help="also render a marked validation raster for this recording index")
    args = ap.parse_args()
    sessions = args.sessions or sorted(
        s for s in list_sessions() if s.startswith("sweep_"))
    for s in sessions:
        run_session(s, args)


if __name__ == "__main__":
    main()
