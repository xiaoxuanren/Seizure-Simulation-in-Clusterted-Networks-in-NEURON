"""Back-fill the existing dephased pilot recordings to flagship layout.

The 5 pilot recordings were written before ``dephase_generate.py`` used the
flagship save path, so they carry only ``spike_times`` + voltage and are missing
the derived fields (``resampled_*``, ``cluster_spike_data``, ``burst_windows``,
``interburst_windows``, ``timestamp``) plus the two raster PNGs and the summary
JSON.

Every one of those is DERIVED from data already on disk, so this recomputes them
in seconds rather than re-running 5 x ~70 minutes of NEURON. No simulation.

Branch-specific fields already present (``snapshot_index``, ``snapshot_time_ms``,
``init_mode``) are preserved.

    python dephase_backfill.py            # rewrite in place
    python dephase_backfill.py --dry-run  # report what would change
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from neuron_simulation import analysis  # noqa: E402
from neuron_simulation.io import save_recording_data  # noqa: E402
from neuron_simulation.workflows import _bursts_to_windows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dephase_generate import (SAVE_ROOT, STATE_SAHP, out_dir,  # noqa: E402
                             write_rasters, write_summary)

FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")

REQUIRED = ("spike_times", "cluster_spike_data", "resampled_spikes",
            "resampled_time_points", "resampled_cluster_assignments",
            "resampling_frequency", "resampling_interval_ms",
            "resampled_spike_positions", "recording_index", "timestamp",
            "duration", "burst_windows", "interburst_windows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-rasters", action="store_true")
    ap.add_argument("--state", default="normal",
                    help="which dephased state folder to back-fill")
    a = ap.parse_args()

    OUT_DIR = out_dir(a.state)
    SESSION_TAG = a.state
    cfg = pickle.load(open(FLAGSHIP_CFG, "rb"))
    paths = sorted(glob.glob(os.path.join(OUT_DIR, "recording*.npz")))
    paths = [p for p in paths if "raster" not in os.path.basename(p)]
    if not paths:
        print("no recordings found in %s" % OUT_DIR)
        return
    print("found %d recordings in %s" % (len(paths), OUT_DIR), flush=True)

    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            old = {k: z[k] for k in z.files}
        rec_idx = int(old.get("recording_index", int(
            os.path.basename(p)[len("recording"):-len(".npz")])))
        missing = [k for k in REQUIRED if k not in old]
        dur = float(old["duration"])
        n = len(old["spike_times"])

        print("\nrecording%03d: %d/%d required fields present, missing %s"
              % (rec_idx, len(REQUIRED) - len(missing), len(REQUIRED),
                 missing if missing else "nothing"), flush=True)
        if a.dry_run:
            continue

        spike_data = {i: np.atleast_1d(np.asarray(old["spike_times"][i], float))
                      for i in range(n)}
        n_sp = sum(len(v) for v in spike_data.values())

        voltage_data = None
        if "voltage_traces" in old:
            voltage_data = dict(
                sample_rate=float(old["voltage_sample_rate"]),
                times=np.asarray(old["voltage_times"], np.float64),
                traces=np.asarray(old["voltage_traces"], np.float32),
                units=str(old.get("voltage_units", "mV")),
                storage_backend="inline_npz")

        bursts = analysis.detect_network_bursts(
            spike_data, n, dur,
            participation_threshold=cfg["participation_threshold"],
            burn_in_ms=0.0)
        bw, ibw = _bursts_to_windows(bursts, dur)
        stats = analysis.burst_statistics(bursts, dur, burn_in_ms=0.0)

        rec_file = save_recording_data(
            spike_data, voltage_data, cfg["cluster_info"], rec_idx, SESSION_TAG,
            SAVE_ROOT, target_freq=cfg["target_freq"], duration=int(dur),
            burst_windows=bw, interburst_windows=ibw)

        # Re-attach the branch-specific fields the pilot already carried.
        with np.load(rec_file, allow_pickle=True) as z:
            merged = {k: z[k] for k in z.files}
        for k in ("snapshot_index", "snapshot_time_ms", "init_mode",
                  "n_neurons", "voltage_gids", "voltage_mode",
                  "discard_transient_ms"):
            if k in old:
                merged[k] = old[k]
        merged.setdefault("init_mode", "dephased_warmstart")
        np.savez_compressed(rec_file, **merged)

        raster = raster_shuf = None
        if not a.no_rasters:
            raster, raster_shuf = write_rasters(
                spike_data, n, dur, cfg["topology"], rec_idx, OUT_DIR,
                sahp_ainc_slow=float(old["sahp_ainc_slow"])
                if "sahp_ainc_slow" in old else STATE_SAHP.get(a.state),
                state=a.state)
        write_summary(rec_idx, OUT_DIR, rec_file, raster, raster_shuf, stats,
                      n_sp,
                      extra=dict(snapshot_index=int(old["snapshot_index"])
                                 if "snapshot_index" in old else None,
                                 init_mode="dephased_warmstart",
                                 backfilled=True))
        print("  -> %d bursts, %d spikes, rasters %s"
              % (int(stats["n_bursts"]), n_sp,
                 "written" if raster else "skipped"), flush=True)

    if a.dry_run:
        print("\n(dry run - nothing written)")
        return

    # verify
    print("\n=== verification against the flagship layout ===", flush=True)
    fl = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                      "20260721_163430", "recording000.npz")
    with np.load(fl, allow_pickle=True) as z:
        flag_keys = set(z.files)
    ok = True
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            keys = set(z.files)
        miss = sorted(flag_keys - keys)
        base = os.path.basename(p)
        pngs = sum(os.path.exists(os.path.join(OUT_DIR, "%s_%s.png"
                                               % (base[:-4], s)))
                   for s in ("raster", "raster_shuffled"))
        summ = os.path.exists(os.path.join(
            OUT_DIR, "_summary_%03d.json" % int(base[len("recording"):-4])))
        print("  %s: missing vs flagship %s | rasters %d/2 | summary %s"
              % (base, miss if miss else "none", pngs, "yes" if summ else "no"))
        if miss or pngs != 2 or not summ:
            ok = False
    print("\n%s" % ("ALL RECORDINGS NOW FLAGSHIP-COMPATIBLE"
                    if ok else "some gaps remain (see above)"))


if __name__ == "__main__":
    main()
