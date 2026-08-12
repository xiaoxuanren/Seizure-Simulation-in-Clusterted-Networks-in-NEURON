"""Part C: generate decoupled (zero-edge) recordings in session format.

Same flagship topology, same build kwargs, same noise seeds -- but
``exc_weight_scale = inh_weight_scale = 0``, so the ground-truth connectivity is
EMPTY. Preprocessing matches the flagship exactly (60 s kept, 1 s transient
discarded) so the GLM sees data of the same shape and duration.

Writes into a NEW session directory; nothing existing is touched.

    python zeroedge_generate.py --start 0 --count 3
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from neuron_simulation.network_builder import build_network  # noqa: E402
from neuron_simulation.simulation import run_simulation  # noqa: E402
from session_paths import resolve, session_dir  # noqa: E402

FLAGSHIP_CFG = os.path.join(resolve("IC-locked_flagship_200rec", "normal"),
                            "_worker_config.pkl")
OUT_DIR = os.path.join(session_dir("IC-locked_zeroedge_control_15rec"),
                       "normal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--duration", type=float, default=60000.0)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = pickle.load(open(FLAGSHIP_CFG, "rb"))
    topology = cfg["topology"]
    bk = dict(cfg["build_kwargs"])
    bk["exc_weight_scale"] = 0.0
    bk["inh_weight_scale"] = 0.0

    network = build_network(topology, noise_seed=cfg["noise_seed_base"],
                            report_deviations=False, **bk)
    print("built zero-edge network: %d cells" % network.n_neurons, flush=True)

    # Ground truth for this session is an EMPTY edge set. Written once (worker 0).
    net_path = os.path.join(OUT_DIR, "network_zeroedge.npz")
    if a.start == 0 and not os.path.exists(net_path):
        np.savez_compressed(
            net_path,
            connections=np.empty((0, 4), dtype=object),
            neuron_is_inhibitory=np.asarray(topology["neuron_is_inhibitory"]),
            cluster_assignments=np.asarray(topology["cluster_assignments"]),
            neuron_positions=np.asarray(topology["neuron_positions"]),
            note="zero-edge control: exc_weight_scale=inh_weight_scale=0")
        print("wrote empty ground truth -> %s" % net_path, flush=True)

    for r in range(a.start, a.start + a.count):
        out = os.path.join(OUT_DIR, "recording%03d.npz" % r)
        if os.path.exists(out):
            print("  skip %s (exists)" % os.path.basename(out), flush=True)
            continue
        t0 = time.time()
        for gen in network.noise:
            gen.reseed(r)
        spike_data, _, _ = run_simulation(
            network, duration=a.duration, dt=cfg["dt"],
            discard_transient_ms=cfg["discard_transient_ms"],
            record_voltage=False, record_ko=False)
        st = np.empty(network.n_neurons, dtype=object)
        for gid in range(network.n_neurons):
            st[gid] = np.asarray(spike_data[gid], dtype=float)
        np.savez_compressed(out, spike_times=st, duration=int(a.duration),
                            recording_index=r,
                            n_neurons=network.n_neurons,
                            exc_weight_scale=0.0, inh_weight_scale=0.0)
        print("  recording%03d: %d spikes, %.1fs -> %s"
              % (r, sum(len(v) for v in spike_data.values()), time.time() - t0,
                 os.path.basename(out)), flush=True)


if __name__ == "__main__":
    main()
