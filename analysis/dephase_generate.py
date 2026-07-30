"""Stage 3 step 2: generate recordings from dephased initial conditions.

Each recording restores a DIFFERENT snapshot from the state library
(``dephase_snapshot.py``), so there is no shared adaptation phase either within
the population or across recordings. Per-recording noise reseeding
(``Random123(base_seed, gid, rec_idx)``) is unchanged from the flagship.

Snapshot assignment is ``rec_idx % n_snapshots``, so with fewer snapshots than
recordings the library is reused -- but each reuse pairs it with a different noise
stream, so no two recordings are identical. Using ONE snapshot for everything
would dephase the population and still leave every recording starting the same
way; that failure mode is what the modulo avoids.

Everything else matches the flagship: same topology, same build kwargs
(noise_weight = 0.007), 60 s kept, 1 s transient discarded.

    python dephase_generate.py --start 0 --count 5
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from neuron import h  # noqa: E402
from neuron_simulation import analysis  # noqa: E402
from neuron_simulation.io import save_recording_data  # noqa: E402
from neuron_simulation.network_builder import build_network  # noqa: E402
from neuron_simulation.workflows import _bursts_to_windows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")
# The single knob (states.py): normal 0.01, seizure 0.004. Nothing else differs.
STATE_SAHP = {"normal": 0.01, "seizure": 0.004}

# save_recording_data joins SAVE_ROOT / SESSION_TAG, so output lands in
# notebooks/NEURON data parallel/dephased_ic/<state>/ -- one folder per state,
# mirroring the flagship's normal/ and seizure/ split.
SAVE_ROOT = os.path.join(REPO, "notebooks", "NEURON data parallel", "dephased_ic")


def library_path(state):
    return os.path.join(HERE, "dephase_state_library_%s.npz" % state)


def out_dir(state):
    return os.path.join(SAVE_ROOT, state)


# Back-compat aliases for callers that predate the two-state split.
SESSION_TAG = "normal"
LIBRARY = library_path("normal")
OUT_DIR = out_dir("normal")

STATE_KEYS = ("v", "hh_m", "hh_h", "hh_n", "kA_m", "kA_h", "ko",
              "g_fast", "g_slow")


def write_rasters(spike_data, n, duration_ms, topology, rec_idx, out_dir,
                  sahp_ainc_slow=None, dot_size=20.0, state="normal"):
    """Both raster variants, matching the flagship pipeline's output.

    Returns ``(raster_path, raster_shuffled_path)``. Never raises -- a plotting
    failure must not lose a recording that took ~70 minutes to produce.
    """
    paths = [None, None]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from neuron_simulation import plotting
        knob = "" if sahp_ainc_slow is None else "  (sahp_ainc_slow=%.3f)" % sahp_ainc_slow
        for k, (shuffled, suffix) in enumerate(((False, "raster"),
                                                (True, "raster_shuffled"))):
            fig = plotting.plot_raster(
                spike_data, n, duration_ms,
                is_inhibitory=topology.get("neuron_is_inhibitory"),
                cluster_assignments=topology["cluster_assignments"],
                burn_in_ms=0.0,
                title="recording %03d - dephased_ic/%s%s%s"
                      % (rec_idx, state, knob,
                         " (randomized rows)" if shuffled else ""),
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
    """``_summary_<NNN>.json``, matching the flagship pipeline's schema."""
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


def restore(network, lib, k):
    """Overwrite every cell's state from snapshot ``k`` of the library.

    Must be called AFTER h.finitialize(), which would otherwise reset these.
    """
    for i, cell in enumerate(network.cells):
        seg = cell.soma(0.5)
        seg.v = lib["v"][k, i]
        seg.hh.m = lib["hh_m"][k, i]
        seg.hh.h = lib["hh_h"][k, i]
        seg.hh.n = lib["hh_n"][k, i]
        seg.kA.m = lib["kA_m"][k, i]
        seg.kA.h = lib["kA_h"][k, i]
        seg.kdyn.ko = lib["ko"][k, i]
        cell.sahp.g_fast = lib["g_fast"][k, i]
        cell.sahp.g_slow = lib["g_slow"][k, i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--duration", type=float, default=60000.0)
    ap.add_argument("--voltage", choices=("none", "probe", "all"), default="probe",
                    help="'all' = every cell (~77 MB/recording at 2 ms); "
                         "'probe' = an evenly spaced subset (default); 'none'")
    ap.add_argument("--voltage-probe-n", type=int, default=40)
    ap.add_argument("--voltage-dt", type=float, default=5.0)
    ap.add_argument("--state", choices=sorted(STATE_SAHP), default="normal",
                    help="single-knob state. Each state reads ITS OWN warm-start "
                         "library and writes to its own folder.")
    ap.add_argument("--sahp-ainc-slow", type=float, default=None,
                    help="override the state's knob value (uS)")
    ap.add_argument("--noise-seed-base", type=int, default=None,
                    help="base seed for the per-neuron Poisson streams "
                         "(Random123(base, gid, recording_index)). Defaults to "
                         "the flagship's value, which makes each dephased "
                         "recording share its noise with the flagship recording "
                         "of the same index -- a controlled comparison. Change it "
                         "for independent noise.")
    ap.add_argument("--discard-extra-ms", type=float, default=0.0,
                    help="added to the flagship's discard_transient_ms. The "
                         "warm start rebuilds synaptic conductance from zero over "
                         "~1 s and can ignite a burst at sim ~1030 ms, i.e. just "
                         "inside the kept window; 2000 moves it outside.")
    a = ap.parse_args()

    state = a.state
    session_tag = state
    outdir = out_dir(state)
    libpath = library_path(state)
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(libpath):
        raise SystemExit(
            "no warm-start library for state '%s' at %s\n"
            "  build it first:  python analysis/dephase_snapshot.py --state %s\n"
            "  (each state needs its own -- the seizure network's stationary "
            "state differs from normal's)" % (state, libpath, state))
    lib = np.load(libpath)
    n_snap = lib["g_slow"].shape[0]
    cfg = pickle.load(open(FLAGSHIP_CFG, "rb"))
    bk = dict(cfg["build_kwargs"])
    knob = a.sahp_ainc_slow if a.sahp_ainc_slow is not None else STATE_SAHP[state]
    bk["sahp_ainc_slow"] = float(knob)
    lib_knob = float(lib["sahp_ainc_slow"]) if "sahp_ainc_slow" in lib else None
    if lib_knob is not None and abs(lib_knob - knob) > 1e-12:
        raise SystemExit(
            "library/state mismatch: %s was warmed up at sahp_ainc_slow=%.4f but "
            "this run wants %.4f. Warm-starting one state from another's "
            "snapshots adds a relaxation transient." % (libpath, lib_knob, knob))
    seed_base = (a.noise_seed_base if a.noise_seed_base is not None
                 else cfg["noise_seed_base"])
    print("state '%s': sahp_ainc_slow = %.4f uS, noise_seed_base = %d -> %s"
          % (state, knob, seed_base, outdir), flush=True)
    net = build_network(cfg["topology"], noise_seed=seed_base,
                        report_deviations=False, **bk)
    n = net.n_neurons
    discard = float(cfg["discard_transient_ms"]) + float(a.discard_extra_ms)
    print("dephased generator: %d cells, %d snapshots in library" % (n, n_snap),
          flush=True)

    net_path = os.path.join(outdir, "network_dephased.npz")
    if a.start == 0 and not os.path.exists(net_path):
        topo = cfg["topology"]
        np.savez_compressed(
            net_path, connections=np.asarray(topo["connections"], dtype=object),
            neuron_is_inhibitory=np.asarray(topo["neuron_is_inhibitory"]),
            cluster_assignments=np.asarray(topo["cluster_assignments"]),
            neuron_positions=np.asarray(topo["neuron_positions"]),
            note="dephased-IC branch, state '%s' (sahp_ainc_slow=%.4f): identical "
                 "topology to the flagship; the initial condition and the single "
                 "knob are what differ" % (state, knob))
        print("  wrote ground truth -> %s" % os.path.basename(net_path), flush=True)

    for r in range(a.start, a.start + a.count):
        out = os.path.join(outdir, "recording%03d.npz" % r)
        if os.path.exists(out):
            print("  skip recording%03d (exists)" % r, flush=True)
            continue
        k = r % n_snap
        t0 = time.time()
        for g in net.noise:
            g.reseed(r)

        vecs, tvec, probe = None, None, None
        if a.voltage != "none":
            probe = (np.arange(n) if a.voltage == "all"
                     else np.linspace(0, n - 1, min(a.voltage_probe_n, n))
                     .astype(int))
            vecs = [h.Vector().record(net.cells[int(g)].soma(0.5)._ref_v,
                                      a.voltage_dt) for g in probe]
            tvec = h.Vector().record(h._ref_t, a.voltage_dt)

        h.dt = float(cfg["dt"])
        h.celsius = float(cfg["build_kwargs"].get("celsius", h.celsius))
        h.finitialize(-65.0)
        restore(net, lib, k)                 # <- the dephasing
        total = discard + a.duration
        h.continuerun(total)

        spike_data, n_sp = {}, 0
        for gid, cell in enumerate(net.cells):
            t = np.asarray(cell.get_spike_times(), float)
            t = t[t >= discard] - discard
            spike_data[gid] = t
            n_sp += t.size

        voltage_data = None
        if a.voltage != "none":
            tt = np.asarray(tvec)
            keep = tt >= discard
            voltage_data = dict(
                sample_rate=float(a.voltage_dt),
                times=(tt[keep] - discard).astype(np.float64),
                traces=np.array([np.asarray(v, np.float32)[keep] for v in vecs],
                                dtype=np.float32),
                units="mV", storage_backend="inline_npz")

        # Flagship-compatible payload: same save path, same field layout.
        bursts = analysis.detect_network_bursts(
            spike_data, n, a.duration,
            participation_threshold=cfg["participation_threshold"], burn_in_ms=0.0)
        bw, ibw = _bursts_to_windows(bursts, a.duration)
        stats = analysis.burst_statistics(bursts, a.duration, burn_in_ms=0.0)

        rec_file = save_recording_data(
            spike_data, voltage_data, cfg["cluster_info"], r, session_tag,
            SAVE_ROOT, target_freq=cfg["target_freq"], duration=int(a.duration),
            burst_windows=bw, interburst_windows=ibw)

        # Fields specific to this branch, appended without disturbing the layout.
        with np.load(rec_file, allow_pickle=True) as z:
            merged = {kk: z[kk] for kk in z.files}
        merged.update(snapshot_index=k,
                      snapshot_time_ms=float(lib["snapshot_times_ms"][k]),
                      init_mode="dephased_warmstart",
                      discard_transient_ms=discard,
                      state_name=state, sahp_ainc_slow=knob,
                      noise_seed_base=int(seed_base), n_neurons=n)
        if voltage_data is not None:
            merged.update(voltage_gids=probe.astype(np.int32),
                          voltage_mode=a.voltage)
        np.savez_compressed(rec_file, **merged)

        raster, raster_shuf = write_rasters(
            spike_data, n, a.duration, cfg["topology"], r, outdir,
            sahp_ainc_slow=knob, state=state)
        write_summary(r, outdir, rec_file, raster, raster_shuf, stats, n_sp,
                      extra=dict(snapshot_index=k, state_name=state,
                                 sahp_ainc_slow=knob,
                                 init_mode="dephased_warmstart",
                                 discard_transient_ms=discard))

        print("  recording%03d: snapshot %d (t=%.0f s), %d spikes (%.4f Hz), "
              "%d bursts, %.0f s wall"
              % (r, k, lib["snapshot_times_ms"][k] / 1000, n_sp,
                 n_sp / (n * a.duration / 1000.0), int(stats["n_bursts"]),
                 time.time() - t0), flush=True)
        del vecs, tvec


if __name__ == "__main__":
    main()
