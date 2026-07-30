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
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from neuron import h  # noqa: E402
from neuron_simulation.network_builder import build_network  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")
LIBRARY = os.path.join(HERE, "dephase_state_library.npz")
OUT_DIR = os.path.join(REPO, "notebooks", "NEURON data parallel",
                       "dephased_ic")

STATE_KEYS = ("v", "hh_m", "hh_h", "hh_n", "kA_m", "kA_h", "ko",
              "g_fast", "g_slow")


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
    ap.add_argument("--discard-extra-ms", type=float, default=0.0,
                    help="added to the flagship's discard_transient_ms. The "
                         "warm start rebuilds synaptic conductance from zero over "
                         "~1 s and can ignite a burst at sim ~1030 ms, i.e. just "
                         "inside the kept window; 2000 moves it outside.")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    lib = np.load(LIBRARY)
    n_snap = lib["g_slow"].shape[0]
    cfg = pickle.load(open(FLAGSHIP_CFG, "rb"))
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"],
                        report_deviations=False, **cfg["build_kwargs"])
    n = net.n_neurons
    discard = float(cfg["discard_transient_ms"]) + float(a.discard_extra_ms)
    print("dephased generator: %d cells, %d snapshots in library" % (n, n_snap),
          flush=True)

    net_path = os.path.join(OUT_DIR, "network_dephased.npz")
    if a.start == 0 and not os.path.exists(net_path):
        topo = cfg["topology"]
        np.savez_compressed(
            net_path, connections=np.asarray(topo["connections"], dtype=object),
            neuron_is_inhibitory=np.asarray(topo["neuron_is_inhibitory"]),
            cluster_assignments=np.asarray(topo["cluster_assignments"]),
            neuron_positions=np.asarray(topo["neuron_positions"]),
            note="dephased-IC branch: identical topology to the flagship; only "
                 "the initial condition differs")
        print("  wrote ground truth -> %s" % os.path.basename(net_path), flush=True)

    for r in range(a.start, a.start + a.count):
        out = os.path.join(OUT_DIR, "recording%03d.npz" % r)
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

        spikes = np.empty(n, dtype=object)
        n_sp = 0
        for gid, cell in enumerate(net.cells):
            t = np.asarray(cell.get_spike_times(), float)
            t = t[t >= discard] - discard
            spikes[gid] = t
            n_sp += t.size

        payload = dict(spike_times=spikes, duration=int(a.duration),
                       recording_index=r, n_neurons=n, snapshot_index=k,
                       snapshot_time_ms=float(lib["snapshot_times_ms"][k]),
                       init_mode="dephased_warmstart")
        if a.voltage != "none":
            tt = np.asarray(tvec)
            keep = tt >= discard
            traces = np.array([np.asarray(v, np.float32)[keep] for v in vecs],
                              dtype=np.float32)
            payload.update(voltage_times=(tt[keep] - discard),
                           voltage_traces=traces,
                           voltage_gids=probe.astype(np.int32),
                           voltage_mode=a.voltage,
                           voltage_sample_rate=float(a.voltage_dt),
                           voltage_units="mV")
        np.savez_compressed(out, **payload)
        print("  recording%03d: snapshot %d (t=%.0f s), %d spikes (%.4f Hz), "
              "%.0f s wall" % (r, k, lib["snapshot_times_ms"][k] / 1000, n_sp,
                               n_sp / (n * a.duration / 1000.0),
                               time.time() - t0), flush=True)
        del vecs, tvec


if __name__ == "__main__":
    main()
