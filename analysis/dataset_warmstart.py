"""Build a warm-start state library for one state.

Runs the network once for ``warmup_ms`` and snapshots every cell's full state at
several well-separated late times. Recordings then initialize from those
snapshots instead of ``h.finitialize(-65)``, which would put every cell in the
same artificial zero-adaptation condition and make the whole population ignite
together early in every recording.

Deliberately NOT a modelled stationary distribution: characterizing the sAHP
shot-noise process and getting it wrong is easy, so the states come from the
model's own late-time behaviour.

State captured per cell (the complete integrator state of this cell model):
    v, hh.m, hh.h, hh.n, kA.m, kA.h, kdyn.ko, sAHP.g_fast, sAHP.g_slow

NOT captured: synaptic conductances and the event queue. Each recording starts
with empty synapses, costing a settling transient on the order of the slowest
synaptic time constant -- that is what ``discard_extra_ms`` covers.

EACH STATE NEEDS ITS OWN LIBRARY. States differ in ``sahp_ainc_slow``, so their
stationary states differ; warm-starting one from another's snapshots would add a
relaxation transient.

    python dataset_warmstart.py --config <session>/_session_config.pkl --state normal
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

STATE_KEYS = ("v", "hh_m", "hh_h", "hh_n", "kA_m", "kA_h", "ko",
              "g_fast", "g_slow")


def library_path(session_dir, state):
    return os.path.join(session_dir, "_state_library_%s.npz" % state)


def load_session(config_path):
    """Load the session bundle written by ``dataset_nb.build_topology``."""
    with open(config_path, "rb") as fh:
        s = pickle.load(fh)
    return s["config"], s["topology"], s["cluster_info"], s["session_dir"]


def build_for(cfg, state):
    """build_network kwargs for one state: only ``sahp_ainc_slow`` differs."""
    b = dict(cfg["build"])
    b["sahp_ainc_slow"] = float(cfg["states"][state])
    return b


def capture(network):
    """Read the full per-cell state vector."""
    n = network.n_neurons
    s = {k: np.zeros(n) for k in STATE_KEYS}
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


def restore(network, lib, k):
    """Overwrite every cell's state from snapshot ``k``.

    Must run AFTER ``h.finitialize()``, which would otherwise reset these.
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
    ap.add_argument("--config", required=True, help="_session_config.pkl")
    ap.add_argument("--state", required=True)
    ap.add_argument("--recording-index", type=int, default=9000,
                    help="noise stream for the warm-up, kept away from the "
                         "indices real recordings use")
    a = ap.parse_args()

    cfg, topology, _ci, session_dir = load_session(a.config)
    bk = build_for(cfg, a.state)
    out = library_path(session_dir, a.state)
    snaps = sorted(float(t) for t in cfg["snapshot_times"])

    print("state '%s': sahp_ainc_slow = %.4f uS, noise_seed_base = %d"
          % (a.state, bk["sahp_ainc_slow"], cfg["noise_seed_base"]), flush=True)
    net = build_network(topology, noise_seed=cfg["noise_seed_base"],
                        report_deviations=False, **bk)
    for g in net.noise:
        g.reseed(a.recording_index)
    print("warm-up: %d cells, %.0f s, snapshots at %s s"
          % (net.n_neurons, cfg["warmup_ms"] / 1000,
             [t / 1000 for t in snaps]), flush=True)

    h.dt = float(cfg["sim"]["dt"])
    h.celsius = float(bk.get("celsius", h.celsius))
    h.finitialize(-65.0)

    lib, t0 = [], time.time()
    for t_stop in snaps:
        h.continuerun(t_stop)
        s = capture(net)
        lib.append(s)
        print("  t = %7.1f s | V %.2f mV | g_slow %.5f uS [%.5f, %.5f] | "
              "ko %.4f mM | %.0f s elapsed"
              % (t_stop / 1000, s["v"].mean(), s["g_slow"].mean(),
                 s["g_slow"].min(), s["g_slow"].max(), s["ko"].mean(),
                 time.time() - t0), flush=True)

    payload = {k: np.stack([s[k] for s in lib]) for k in STATE_KEYS}
    payload["snapshot_times_ms"] = np.asarray(snaps, float)
    payload["warmup_duration_ms"] = float(cfg["warmup_ms"])
    payload["state"] = a.state
    payload["sahp_ainc_slow"] = float(bk["sahp_ainc_slow"])
    payload["noise_seed_base"] = int(cfg["noise_seed_base"])
    np.savez_compressed(out, **payload)

    gs = payload["g_slow"]
    print("\nlibrary: %d snapshots x %d cells | g_slow mean %.5f uS, "
          "within-population sd %.5f uS"
          % (gs.shape[0], gs.shape[1], gs.mean(), gs.std(axis=1).mean()),
          flush=True)
    print("  (a finitialize start would have g_slow = 0 for every cell)",
          flush=True)
    print("  saved -> %s" % out, flush=True)
    print("  TOTAL %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
