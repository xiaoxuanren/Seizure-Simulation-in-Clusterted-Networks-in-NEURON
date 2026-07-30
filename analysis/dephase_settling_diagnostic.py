"""Why does a warm-started recording sometimes burst at t = 0?

Hypothesis: ``dephase_generate.restore()`` restores MEMBRANE state (v, hh gates,
kA gates, ko, sAHP) but NOT SYNAPTIC state. Every recording therefore starts with
all 13,356 synapses at zero conductance and an empty event queue. That removes
tonic inhibition as well as tonic excitation, and the net disinhibition can tip
the network into a burst while the synaptic conductances climb back to their
steady values.

Test: restore a snapshot, run 2 s, and record the population Vm, the spike raster,
and the SUMMED excitatory and inhibitory synaptic conductances. If the hypothesis
holds, the burst happens while g_inh is still ramping up from zero.

Run for a snapshot that burst (3) and one that did not (0) as a control.

    python dephase_settling_diagnostic.py
"""

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
OUT = os.path.join(HERE, "dephase_settling_diagnostic.npz")

SIM_MS = 2000.0
STEP_MS = 10.0


def main():
    lib = np.load(LIBRARY)
    cfg = pickle.load(open(FLAGSHIP_CFG, "rb"))
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"],
                        report_deviations=False, **cfg["build_kwargs"])
    sys.path.insert(0, HERE)
    from dephase_generate import restore

    # Classify synapses: excitatory ones are AmpaNmda (have g_ampa).
    exc_syn = [s for s in net.synapses if hasattr(s, "g_ampa")]
    inh_syn = [s for s in net.synapses if not hasattr(s, "g_ampa")]
    noise_syn = [g.syn for g in net.noise]
    print("synapses: %d excitatory (AmpaNmda), %d inhibitory, %d background"
          % (len(exc_syn), len(inh_syn), len(noise_syn)), flush=True)

    out = {}
    for snap, label in ((3, "snap3_burst"), (0, "snap0_control")):
        for gen in net.noise:
            gen.reseed(3 if snap == 3 else 0)
        h.dt = float(cfg["dt"])
        h.celsius = float(cfg["build_kwargs"].get("celsius", h.celsius))
        h.finitialize(-65.0)
        restore(net, lib, snap)

        n_steps = int(SIM_MS / STEP_MS)
        t_ax = np.arange(n_steps + 1) * STEP_MS
        vm = np.zeros(n_steps + 1)
        g_e = np.zeros(n_steps + 1)
        g_i = np.zeros(n_steps + 1)
        g_n = np.zeros(n_steps + 1)

        def sample(k):
            vm[k] = np.mean([c.soma(0.5).v for c in net.cells])
            g_e[k] = sum(s.g_ampa + s.g_nmda for s in exc_syn)
            g_i[k] = sum(s.g for s in inh_syn)
            g_n[k] = sum(s.g for s in noise_syn)

        t0 = time.time()
        sample(0)
        for k in range(1, n_steps + 1):
            h.continuerun(k * STEP_MS)
            sample(k)
        spikes = [np.asarray(c.get_spike_times(), float) for c in net.cells]
        n_sp = sum(len(s) for s in spikes)
        print("  %s: %d spikes in %.0f ms | %.0f s wall"
              % (label, n_sp, SIM_MS, time.time() - t0), flush=True)

        # when does g_inh reach 90% of its late value?
        late_i = g_i[int(1500 / STEP_MS):].mean()
        reach = t_ax[np.argmax(g_i >= 0.9 * late_i)] if late_i > 0 else np.nan
        first50 = sum(int((s < 50).sum()) for s in spikes)
        print("     g_inh: 0 -> %.4f uS (late mean); reaches 90%% at t = %.0f ms"
              % (late_i, reach), flush=True)
        print("     spikes in first 50 ms: %d | population Vm at t=0 %.2f, "
              "peak over 0-200 ms %.2f mV"
              % (first50, vm[0], vm[:int(200 / STEP_MS)].max()), flush=True)

        out[label] = dict(t=t_ax, vm=vm, g_exc=g_e, g_inh=g_i, g_noise=g_n,
                          late_g_inh=late_i, t_90pct_ms=reach,
                          spikes_first50=first50)
        out[label + "_spike_t"] = np.concatenate(
            [s for s in spikes if s.size]) if any(s.size for s in spikes) else np.array([])
        out[label + "_spike_i"] = np.concatenate(
            [np.full(s.size, i) for i, s in enumerate(spikes) if s.size]) \
            if any(s.size for s in spikes) else np.array([])

    flat = {}
    for k, v in out.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat["%s__%s" % (k, kk)] = vv
        else:
            flat[k] = v
    np.savez_compressed(OUT, **flat)
    print("\nsaved -> %s" % os.path.basename(OUT), flush=True)


if __name__ == "__main__":
    main()
