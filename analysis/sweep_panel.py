"""Per-network standard panel for the sweep sessions.

For each session, produces in results/<state>/figures/:
  recordingNNN_raster.png   -- the session's REPRESENTATIVE example raster:
                               the recording whose burst count is closest to
                               the session median (ties -> lowest index),
                               rendered by render_rasters.py
  lag_auc.png               -- per-lag exc/inh AUC + AP from glm_lag_sweep.json

Run with the NEURON-capable interpreter (render_rasters imports the package):
    py -3.9 analysis/sweep_panel.py                  # all sweep_* sessions
    py -3.9 analysis/sweep_panel.py sweep_c50_seed01 ...
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from session_paths import resolve, results_dir, list_sessions  # noqa: E402

STATE = os.environ.get("DATASET_STATE", "normal")


def median_burst_recording(meta):
    counts = [(r.get("n_bursts", 0), r["index"]) for r in meta["recordings"]]
    med = float(np.median([c for c, _ in counts]))
    return min(counts, key=lambda ci: (abs(ci[0] - med), ci[1]))[1]


def raster(session, sd, figdir, meta):
    idx = median_burst_recording(meta)
    out = os.path.join(figdir, "recording%03d_raster.png" % idx)
    cmd = [sys.executable, os.path.join(HERE, "render_rasters.py"),
           "--folder", sd, "--phenotype", STATE, "--sahp", "0.01",
           "--indices", str(idx), "--out", figdir,
           "--no-shuffled", "--burst-count", "--dot-size", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("  raster FAILED:\n%s" % r.stdout[-500:] + r.stderr[-500:])
        return None
    print("  raster (median-burst rec %03d) -> %s" % (idx, out))
    return out


def lag_curve(session, sd, figdir):
    path = os.path.join(sd, "glm_lag_sweep.json")
    if not os.path.exists(path):
        print("  lag_auc SKIP (no glm_lag_sweep.json)")
        return None
    with open(path, "r", encoding="utf-8") as fh:
        sweep = json.load(fh)
    lags = [r["lag"] for r in sweep["per_lag"]]
    bin_ms = sweep["bin_ms"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, style, label in (("exc_auc", "o-", "exc AUC"), ("inh_auc", "s-", "inh AUC"),
                              ("exc_ap", "o--", "exc AP"), ("inh_ap", "s--", "inh AP")):
        ax.plot(lags, [r[key] for r in sweep["per_lag"]], style, ms=5, label=label)
    ax.axvline(sweep["best_lag"], color="grey", ls=":", lw=1)
    ax.set_xlabel("lag (bin = %g ms)" % bin_ms)
    ax.set_ylabel("score")
    ax.set_title("%s -- per-lag edge scores (best lag %d: %d-%d ms)" % (
        session, sweep["best_lag"],
        (sweep["best_lag"] - 1) * bin_ms, sweep["best_lag"] * bin_ms))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    out = os.path.join(figdir, "lag_auc.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    print("  lag_auc -> %s" % out)
    return out


def main():
    sessions = sys.argv[1:] or sorted(
        s for s in list_sessions() if s.startswith("sweep_"))
    for session in sessions:
        sd = resolve(session, STATE)
        figdir = results_dir(session, STATE, "figures")
        with open(os.path.join(sd, "session_metadata.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        print(session)
        raster(session, sd, figdir, meta)
        lag_curve(session, sd, figdir)


if __name__ == "__main__":
    main()
