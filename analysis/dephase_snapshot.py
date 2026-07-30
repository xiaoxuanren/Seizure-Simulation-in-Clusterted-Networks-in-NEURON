"""Stage 3 step 1: build a warm-start state library.

Runs ONE long simulation of the flagship network and snapshots every cell's full
state at several well-separated late times. Recordings then initialize from these
snapshots instead of ``h.finitialize(-65)``, which removes the shared initial
condition that ignites the population once at ~4.9 s.

Deliberately NOT a modelled stationary distribution: characterizing the sAHP
shot-noise process and getting it wrong is easy, so the states come from the
model's own late-time behaviour.

State captured per cell (the complete integrator state of this cell model):
    v, hh.m, hh.h, hh.n, kA.m, kA.h, kdyn.ko, sAHP.g_fast, sAHP.g_slow

NOT captured: synaptic conductances and the event queue. Each recording starts
with empty synapses, which costs a settling transient on the order of the slowest
synaptic time constant (tau_nmda = 350 ms) -- three orders of magnitude shorter
than the tau_slow = 6.5 s artifact this is removing.

    python dephase_snapshot.py --duration 130000 --snapshots 50000 70000 90000 110000 130000
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
from neuron_simulation.network_builder import build_network  # noqa: E402

FLAGSHIP_CFG = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                            "20260721_163430", "_worker_config.pkl")
HERE = os.path.dirname(os.path.abspath(__file__))

# The single knob (states.py): normal 0.01, seizure 0.004. Everything else is
# identical between states.
STATE_SAHP = {"normal": 0.01, "seizure": 0.004}


def library_path(state):
    return os.path.join(HERE, "dephase_state_library_%s.npz" % state)


def capture(network):
    """Read the full per-cell state vector."""
    n = network.n_neurons
    s = {k: np.zeros(n) for k in
         ("v", "hh_m", "hh_h", "hh_n", "kA_m", "kA_h", "ko", "g_fast", "g_slow")}
    for i, cell in enumerate(network.cells):
        seg = cell.soma(0.5)
        s["v"][i] = seg.v
        s["hh_m"][i] = seg.hh.m
        s["hh_h"][i] = seg.hh.h
        s["hh_n"][i] = seg.hh.n
        s["kA_m"][i] = seg.kA.m
        s["kA_h"][i] = seg.kA.h
        s["ko"][i] = seg.kdyn.ko
        s["g_fast"][i] = cell.sahp.g_fast
        s["g_slow"][i] = cell.sahp.g_slow
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=130000.0)
    ap.add_argument("--snapshots", type=float, nargs="+",
                    default=[50000., 70000., 90000., 110000., 130000.])
    ap.add_argument("--recording-index", type=int, default=9000,
                    help="noise stream for the warm-up run; kept well away from "
                         "the indices used for real recordings")
    ap.add_argument("--state", choices=sorted(STATE_SAHP), default="normal",
                    help="which single-knob state to warm up. EACH STATE NEEDS "
                         "ITS OWN LIBRARY: the stationary state of the seizure "
                         "network differs from the normal one, so warm-starting "
                         "seizure from normal snapshots would add a relaxation "
                         "transient.")
    ap.add_argument("--sahp-ainc-slow", type=float, default=None,
                    help="override the state's knob value (uS)")
    ap.add_argument("--topology-pkl", type=str, default=None,
                    help="worker-config pickle to take the topology from. "
                         "Defaults to the flagship's, so the graph matches the "
                         "200-recording dataset. A state library and the "
                         "recordings that use it MUST share one topology.")
    ap.add_argument("--build-overrides", type=str, default=None,
                    help="JSON dict merged over the flagship build_kwargs, so a "
                         "notebook can own the build config. Topology is NOT "
                         "affected -- the graph is loaded materialised.")
    ap.add_argument("--noise-seed-base", type=int, default=None,
                    help="base seed for the per-neuron Poisson streams "
                         "(Random123(base, gid, recording_index)). Defaults to "
                         "the flagship's value from the worker config.")
    a = ap.parse_args()

    cfg = pickle.load(open(a.topology_pkl or FLAGSHIP_CFG, "rb"))
    if a.topology_pkl:
        print("topology from %s (%d neurons, %d edges)"
              % (os.path.basename(a.topology_pkl), cfg["topology"]["n_neurons"],
                 len(cfg["topology"]["connections"])), flush=True)
    bk = dict(cfg["build_kwargs"])
    knob = a.sahp_ainc_slow if a.sahp_ainc_slow is not None else STATE_SAHP[a.state]
    if a.build_overrides:
        ov = json.loads(a.build_overrides)
        changed = {k: (bk.get(k), v) for k, v in ov.items() if bk.get(k) != v}
        bk.update(ov)
        if changed:
            print("build overrides applied:", flush=True)
            for k, (was, now) in sorted(changed.items()):
                print("    %-20s %r -> %r" % (k, was, now), flush=True)
    bk["sahp_ainc_slow"] = float(knob)
    seed_base = (a.noise_seed_base if a.noise_seed_base is not None
                 else cfg["noise_seed_base"])
    OUT = library_path(a.state)
    print("state '%s': sahp_ainc_slow = %.4f uS, noise_seed_base = %d -> %s"
          % (a.state, knob, seed_base, os.path.basename(OUT)), flush=True)
    net = build_network(cfg["topology"], noise_seed=seed_base,
                        report_deviations=False, **bk)
    for g in net.noise:
        g.reseed(a.recording_index)
    print("warm-up network built: %d cells | snapshots at %s s"
          % (net.n_neurons, [t / 1000 for t in a.snapshots]), flush=True)

    h.dt = float(cfg["dt"])
    h.celsius = float(bk.get("celsius", h.celsius))
    h.finitialize(-65.0)

    lib, t0 = [], time.time()
    for t_stop in sorted(a.snapshots):
        h.continuerun(float(t_stop))
        s = capture(net)
        lib.append(s)
        print("  t = %7.1f s  |  V %.2f mV [%.2f, %.2f]  g_slow %.5f uS "
              "[%.5f, %.5f]  ko %.4f mM  |  %.0f s elapsed"
              % (t_stop / 1000, s["v"].mean(), s["v"].min(), s["v"].max(),
                 s["g_slow"].mean(), s["g_slow"].min(), s["g_slow"].max(),
                 s["ko"].mean(), time.time() - t0), flush=True)

    keys = list(lib[0])
    payload = {k: np.stack([s[k] for s in lib]) for k in keys}
    payload["snapshot_times_ms"] = np.asarray(sorted(a.snapshots), float)
    payload["warmup_duration_ms"] = float(a.duration)
    payload["warmup_recording_index"] = int(a.recording_index)
    payload["state"] = a.state
    payload["sahp_ainc_slow"] = float(knob)
    payload["noise_seed_base"] = int(seed_base)
    np.savez_compressed(OUT, **payload)

    gs = payload["g_slow"]
    print("\n--- state library ---", flush=True)
    print("  %d snapshots x %d cells" % gs.shape, flush=True)
    print("  g_slow: mean %.5f uS, within-snapshot sd %.5f, across-snapshot "
          "sd of means %.5f" % (gs.mean(), gs.std(axis=1).mean(),
                                gs.mean(axis=1).std()), flush=True)
    print("  a shared-IC start would have g_slow = 0 for every cell; here the "
          "within-population spread is %.5f uS" % gs.std(axis=1).mean(), flush=True)
    print("  saved -> %s" % OUT, flush=True)
    print("  TOTAL %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
