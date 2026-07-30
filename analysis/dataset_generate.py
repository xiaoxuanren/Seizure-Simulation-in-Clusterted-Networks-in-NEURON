"""Generate recordings for one state from a session config.

Each recording restores a snapshot from that state's warm-start library
(``dataset_warmstart.py``) rather than starting from ``h.finitialize(-65)``, so
there is no shared adaptation phase within the population or across recordings.

Snapshot assignment is ``rec_idx % n_snapshots``. With fewer snapshots than
recordings the library is reused, but each reuse pairs it with a different noise
stream (``Random123(noise_seed_base, gid, rec_idx)``), so no two recordings are
identical. Using one snapshot for everything would spread the population and
still leave every recording starting the same way; the modulo avoids that.

Output per recording matches the project's standard layout: a full ``.npz``
(spikes, cluster-organised spikes, resampled raster, burst windows, optional
voltage), both raster variants, and a ``_summary_NNN.json``.

    python dataset_generate.py --config <session>/_session_config.pkl \\
        --state normal --start 0 --count 10
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from neuron import h  # noqa: E402
from neuron_simulation import analysis  # noqa: E402
from neuron_simulation.io import save_recording_data  # noqa: E402
from neuron_simulation.network_builder import build_network  # noqa: E402
from neuron_simulation.workflows import _bursts_to_windows  # noqa: E402
from dataset_warmstart import (build_for, library_path,  # noqa: E402
                               load_session, restore)


def write_rasters(spike_data, n, duration_ms, topology, rec_idx, out_dir,
                  state, sahp_ainc_slow, dot_size=20.0):
    """Both raster variants. Never raises: a plot must not lose a long run."""
    paths = [None, None]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from neuron_simulation import plotting
        for k, (shuffled, suffix) in enumerate(((False, "raster"),
                                                (True, "raster_shuffled"))):
            fig = plotting.plot_raster(
                spike_data, n, duration_ms,
                is_inhibitory=topology.get("neuron_is_inhibitory"),
                cluster_assignments=topology["cluster_assignments"],
                burn_in_ms=0.0,
                title="recording %03d - %s (sahp_ainc_slow=%.3f)%s"
                      % (rec_idx, state, sahp_ainc_slow,
                         "  (randomized rows)" if shuffled else ""),
                randomize_rows=shuffled, dot_size=dot_size,
                show_burst_count=True)
            fn = os.path.join(out_dir, "recording%03d_%s.png" % (rec_idx, suffix))
            fig.savefig(fn, dpi=120, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            paths[k] = fn
    except Exception as exc:
        print("  [warn] raster skipped for recording%03d: %s" % (rec_idx, exc),
              flush=True)
    return paths[0], paths[1]


def write_summary(rec_idx, out_dir, rec_file, raster, raster_shuffled, stats,
                  n_spikes, extra=None):
    payload = dict(index=int(rec_idx),
                   file=os.path.relpath(rec_file, REPO) if rec_file else None,
                   raster=os.path.relpath(raster, REPO) if raster else None,
                   raster_shuffled=(os.path.relpath(raster_shuffled, REPO)
                                    if raster_shuffled else None),
                   success=True,
                   n_bursts=int(stats["n_bursts"]),
                   burst_rate_hz=float(stats["burst_rate_hz"]),
                   mean_participation=float(stats["mean_participation"]),
                   num_spikes=int(n_spikes))
    if extra:
        payload.update(extra)
    p = os.path.join(out_dir, "_summary_%03d.json" % rec_idx)
    json.dump(payload, open(p, "w"), indent=1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="_session_config.pkl")
    ap.add_argument("--state", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    a = ap.parse_args()

    cfg, topology, cluster_info, session_dir = load_session(a.config)
    state = a.state
    out_dir = os.path.join(session_dir, state)
    os.makedirs(out_dir, exist_ok=True)

    lp = library_path(session_dir, state)
    if not os.path.exists(lp):
        raise SystemExit(
            "no warm-start library for state '%s' at %s\n"
            "  build it first:  python analysis/dataset_warmstart.py "
            "--config %s --state %s" % (state, lp, a.config, state))
    lib = np.load(lp)
    n_snap = lib["g_slow"].shape[0]
    lib_knob = float(lib["sahp_ainc_slow"])
    bk = build_for(cfg, state)
    if abs(lib_knob - bk["sahp_ainc_slow"]) > 1e-12:
        raise SystemExit(
            "library/state mismatch: %s was built at sahp_ainc_slow=%.4f but "
            "this run wants %.4f. Warm-starting one state from another's "
            "snapshots adds a relaxation transient."
            % (os.path.basename(lp), lib_knob, bk["sahp_ainc_slow"]))

    duration = float(cfg["recording_ms"])
    discard = float(cfg["sim"]["discard_transient_ms"]) + float(cfg["discard_extra_ms"])
    vmode = cfg["voltage"]

    print("state '%s': sahp_ainc_slow = %.4f uS, seed %d, %d snapshots -> %s"
          % (state, bk["sahp_ainc_slow"], cfg["noise_seed_base"], n_snap, out_dir),
          flush=True)
    net = build_network(topology, noise_seed=cfg["noise_seed_base"],
                        report_deviations=False, **bk)
    n = net.n_neurons
    probe = (np.arange(n) if vmode == "all"
             else np.linspace(0, n - 1, min(int(cfg["voltage_probe_n"]), n)).astype(int))

    for r in range(a.start, a.start + a.count):
        out = os.path.join(out_dir, "recording%03d.npz" % r)
        if os.path.exists(out):
            print("  skip recording%03d (exists)" % r, flush=True)
            continue
        k = r % n_snap
        t0 = time.time()
        for g in net.noise:
            g.reseed(r)

        vecs = tvec = None
        if vmode != "none":
            vecs = [h.Vector().record(net.cells[i].soma(0.5)._ref_v,
                                      float(cfg["voltage_dt"])) for i in probe]
            tvec = h.Vector().record(h._ref_t, float(cfg["voltage_dt"]))

        h.dt = float(cfg["sim"]["dt"])
        h.celsius = float(bk.get("celsius", h.celsius))
        h.finitialize(-65.0)
        restore(net, lib, k)                       # <- the warm start
        h.continuerun(discard + duration)

        spike_data, n_sp = {}, 0
        for gid, cell in enumerate(net.cells):
            t = np.asarray(cell.get_spike_times(), float)
            t = t[t >= discard] - discard
            spike_data[gid] = t
            n_sp += t.size

        voltage_data = None
        if vmode != "none":
            tt = np.asarray(tvec)
            keep = tt >= discard
            voltage_data = dict(
                sample_rate=float(cfg["voltage_dt"]),
                times=(tt[keep] - discard).astype(np.float64),
                traces=np.array([np.asarray(v, np.float32)[keep] for v in vecs],
                                dtype=np.float32),
                units="mV", storage_backend="inline_npz")

        bursts = analysis.detect_network_bursts(
            spike_data, n, duration,
            participation_threshold=float(cfg["sim"]["participation_threshold"]),
            burn_in_ms=0.0)
        bw, ibw = _bursts_to_windows(bursts, duration)
        stats = analysis.burst_statistics(bursts, duration, burn_in_ms=0.0)

        rec_file = save_recording_data(
            spike_data, voltage_data, cluster_info, r, state, session_dir,
            target_freq=int(cfg["sim"]["target_freq"]), duration=int(duration),
            burst_windows=bw, interburst_windows=ibw)

        with np.load(rec_file, allow_pickle=True) as z:
            merged = {kk: z[kk] for kk in z.files}
        merged.update(snapshot_index=k,
                      snapshot_time_ms=float(lib["snapshot_times_ms"][k]),
                      init_mode="warmstart", n_neurons=n,
                      state_name=state, sahp_ainc_slow=bk["sahp_ainc_slow"],
                      noise_seed_base=int(cfg["noise_seed_base"]),
                      discard_transient_ms=discard,
                      participation_threshold=float(cfg["sim"]["participation_threshold"]))
        if voltage_data is not None:
            merged.update(voltage_gids=probe.astype(np.int32), voltage_mode=vmode)
        np.savez_compressed(rec_file, **merged)

        raster, raster_shuf = write_rasters(
            spike_data, n, duration, topology, r, out_dir, state,
            bk["sahp_ainc_slow"])
        write_summary(r, out_dir, rec_file, raster, raster_shuf, stats, n_sp,
                      extra=dict(state_name=state, snapshot_index=k,
                                 sahp_ainc_slow=bk["sahp_ainc_slow"],
                                 init_mode="warmstart"))

        print("  recording%03d: snapshot %d, %d spikes (%.4f Hz), %d bursts, "
              "%.0f s wall" % (r, k, n_sp, n_sp / (n * duration / 1000.0),
                               int(stats["n_bursts"]), time.time() - t0),
              flush=True)
        del vecs, tvec


if __name__ == "__main__":
    main()
