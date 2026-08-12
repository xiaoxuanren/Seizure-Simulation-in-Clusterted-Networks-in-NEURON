"""Re-render saved parallel-dataset rasters with a corrected title + finer dots.

Reads the already-saved recording###.npz spike data (no re-simulation) and the
session network_*.npz, then calls the SAME neuron_simulation.plotting.plot_raster
used by the pipeline -- so bursts / population panel match exactly, only the
title and dot size change.

Usage:
  py -3.9 render_rasters.py --folder <session_dir> --phenotype seizure \
      [--sahp 0.004] [--dot-size 4] [--indices 0] [--out <dir>] [--no-shuffled]

If --sahp is omitted it is read from the folder's _worker_config.pkl (ground
truth for what was actually simulated). If --indices is omitted, ALL recordings
in the folder are rendered. If --out is omitted, PNGs overwrite the folder's
existing rasters (batch mode); pass --out for a dry-run preview elsewhere.
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np


def _load_sahp_from_config(folder):
    cfg_path = os.path.join(folder, "_worker_config.pkl")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, "rb") as f:
        cfg = pickle.load(f)
    bk = cfg.get("build_kwargs", {}) or {}
    return bk.get("sahp_ainc_slow")


def _find_network_npz(folder):
    hits = glob.glob(os.path.join(folder, "network_*.npz"))
    if not hits:
        raise FileNotFoundError("no network_*.npz in %s" % folder)
    return hits[0]


def _recording_indices(folder):
    idx = []
    for p in glob.glob(os.path.join(folder, "recording*[0-9].npz")):
        m = re.search(r"recording(\d+)\.npz$", os.path.basename(p))
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--phenotype", required=True, help='e.g. "seizure" or "normal"')
    ap.add_argument("--sahp", type=float, default=None)
    ap.add_argument("--dot-size", type=float, default=4.0)
    ap.add_argument("--indices", default=None, help="comma-separated; default all")
    ap.add_argument("--out", default=None, help="output dir; default = in-place")
    ap.add_argument("--no-shuffled", action="store_true")
    ap.add_argument("--only-shuffled", action="store_true", help="render only the randomized-row raster")
    ap.add_argument("--burst-count", action="store_true", help="append (N network bursts) to title")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    args = ap.parse_args()

    # Make neuron_simulation importable (repo root that contains it).
    repo_root = args.repo
    if not os.path.isdir(os.path.join(repo_root, "neuron_simulation")):
        # fall back: walk up from folder
        repo_root = os.path.abspath(os.path.join(args.folder, "..", "..", "..", ".."))
    sys.path.insert(0, repo_root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neuron_simulation import plotting

    sahp = args.sahp if args.sahp is not None else _load_sahp_from_config(args.folder)
    if sahp is None:
        raise SystemExit("could not determine sahp_ainc_slow; pass --sahp")

    net = np.load(_find_network_npz(args.folder), allow_pickle=True)
    cluster_assignments = np.asarray(net["cluster_assignments"], dtype=int)
    is_inh = None
    if "neuron_is_inhibitory" in net.files:
        is_inh = np.asarray(net["neuron_is_inhibitory"]).astype(bool)

    if args.indices:
        indices = [int(x) for x in args.indices.split(",")]
    else:
        indices = _recording_indices(args.folder)

    out_dir = args.out or args.folder
    os.makedirs(out_dir, exist_ok=True)

    if args.only_shuffled:
        variants = [(True, "raster_shuffled")]
    else:
        variants = [(False, "raster")]
        if not args.no_shuffled:
            variants.append((True, "raster_shuffled"))

    print("folder     : %s" % args.folder)
    print("phenotype  : %s | sahp_ainc_slow=%.3f | dot_size=%.1f" % (args.phenotype, sahp, args.dot_size))
    print("recordings : %s" % (indices if len(indices) <= 10 else "%d recordings %d..%d" % (len(indices), indices[0], indices[-1])))
    print("out_dir    : %s%s" % (out_dir, "  (IN-PLACE overwrite)" if out_dir == args.folder else ""))

    for rec_idx in indices:
        rec_path = os.path.join(args.folder, "recording%03d.npz" % rec_idx)
        rec = np.load(rec_path, allow_pickle=True)
        spike_times = rec["spike_times"]
        n_neurons = len(spike_times)
        duration_ms = float(rec["duration"])
        spike_data = {i: np.asarray(spike_times[i], dtype=float) for i in range(n_neurons)}

        for shuffled, suffix in variants:
            title = "recording %03d - %s (sahp_ainc_slow=%.3f)%s" % (
                rec_idx, args.phenotype, sahp,
                " (randomized rows)" if shuffled else "")
            fig = plotting.plot_raster(
                spike_data, n_neurons, duration_ms,
                is_inhibitory=is_inh, cluster_assignments=cluster_assignments,
                burn_in_ms=0.0, title=title, randomize_rows=shuffled,
                dot_size=args.dot_size, show_burst_count=args.burst_count)
            fn = os.path.join(out_dir, "recording%03d_%s.png" % (rec_idx, suffix))
            fig.savefig(fn, dpi=120, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print("  wrote %s" % fn)


if __name__ == "__main__":
    main()
