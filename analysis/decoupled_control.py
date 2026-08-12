"""Zero-recurrence control: is the ~4.9 s burst network-generated or an artifact
of every cell starting from the same initial condition?

Rebuilds the FLAGSHIP normal network (same topology, same build kwargs, same
noise seeds as notebooks/NEURON data parallel/normal/20260721_163430) and runs it
twice:

    --coupled 1   exc_weight_scale=2.0, inh_weight_scale=2.5   (flagship)
    --coupled 0   exc_weight_scale=0.0, inh_weight_scale=0.0   (control)

Everything else is byte-identical, including the per-cell Poisson streams
(Random123(noise_seed_base, gid, recording_index)). ``discard_transient_ms=0``
so the initialization transient -- the thing under test -- is KEPT.

Saves spikes, population Vm (mean + percentiles), [K+]o, and per-50 ms
participation to an npz, and prints the summary stats that separate the two
hypotheses:

  * shared-IC artifact  -> control shows a burst at ~4.9 s with high participation
  * network-generated   -> control shows only the early (~0.1 s) IC synchrony,
                           decaying by cycle two, and no 4.9 s event
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuron_simulation.network_builder import build_network
from neuron_simulation.simulation import run_simulation

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
FLAGSHIP = os.path.join(SESSION, "_worker_config.pkl")


def participation_series(spike_data, n_neurons, duration_ms, bin_ms=50.0):
    """Fraction of DISTINCT cells firing in each bin (participation, not rate)."""
    n_bins = int(np.ceil(duration_ms / bin_ms))
    counts = np.zeros(n_bins, dtype=np.int32)
    for gid, times in spike_data.items():
        if len(times) == 0:
            continue
        idx = np.unique((np.asarray(times) / bin_ms).astype(int))
        idx = idx[(idx >= 0) & (idx < n_bins)]
        counts[idx] += 1
    edges = np.arange(n_bins) * bin_ms
    return edges, counts / float(n_neurons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupled", type=int, default=0,
                    help="1 = flagship recurrent weights, 0 = zero recurrence")
    ap.add_argument("--duration", type=float, default=60000.0)
    ap.add_argument("--recording-index", type=int, default=0)
    ap.add_argument("--voltage-dt", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = pickle.load(open(FLAGSHIP, "rb"))
    topology = cfg["topology"]
    build_kwargs = dict(cfg["build_kwargs"])

    if not args.coupled:
        build_kwargs["exc_weight_scale"] = 0.0
        build_kwargs["inh_weight_scale"] = 0.0

    tag = "coupled" if args.coupled else "decoupled"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "decoupled_control_%s.npz" % tag)

    print("=== %s ===" % tag)
    print("  n_neurons        %d" % topology["n_neurons"])
    print("  exc_weight_scale %.2f" % build_kwargs["exc_weight_scale"])
    print("  inh_weight_scale %.2f" % build_kwargs["inh_weight_scale"])
    print("  noise_weight     %.4f uS @ %.1f Hz" % (build_kwargs["noise_weight"],
                                                    build_kwargs["noise_rate"]))
    print("  sahp_ainc_slow   %.4f uS, tau_slow %.0f ms"
          % (build_kwargs["sahp_ainc_slow"], build_kwargs["sahp_tau_slow"]))
    print("  noise_seed_base  %d, recording_index %d"
          % (cfg["noise_seed_base"], args.recording_index))

    t0 = time.time()
    network = build_network(topology, noise_seed=cfg["noise_seed_base"],
                            report_deviations=False, **build_kwargs)
    # Match the flagship's per-recording stream keying.
    for gen in network.noise:
        gen.reseed(args.recording_index)
    print("  built in %.1f s" % (time.time() - t0))

    # discard_transient_ms=0: the startup transient is the object of study.
    spike_data, voltage_data, ko_data = run_simulation(
        network,
        duration=args.duration,
        dt=cfg["dt"],
        discard_transient_ms=0.0,
        record_voltage=True,
        voltage_dt=args.voltage_dt,
        record_ko=True,
        ko_dt=10.0,
        progress_every_ms=10000.0,
    )

    n = topology["n_neurons"]
    traces = voltage_data["traces"]
    vm_times = voltage_data["times"]
    vm_mean = traces.mean(axis=0)
    vm_p10 = np.percentile(traces, 10, axis=0)
    vm_p90 = np.percentile(traces, 90, axis=0)

    edges, part = participation_series(spike_data, n, args.duration, bin_ms=50.0)

    first = np.array([times[0] if len(times) else np.nan
                      for times in spike_data.values()], dtype=float)
    n_spikes = int(sum(len(v) for v in spike_data.values()))
    rate = n_spikes / (n * args.duration / 1000.0)

    # ISI regularity. Pooling ISIs across cells inflates CV whenever cells
    # differ in rate, so report the WITHIN-cell CV (mean over cells) as the
    # regularity statistic and keep the pooled one only for comparison.
    per_cell = [np.diff(t) for t in spike_data.values() if len(t) >= 4]
    cvs = np.array([d.std() / d.mean() for d in per_cell if d.mean() > 0])
    cv_within = cvs.mean() if cvs.size else np.nan
    isis = np.concatenate(per_cell) if per_cell else np.array([])
    cv = (isis.std() / isis.mean()) if isis.size else np.nan

    order = np.argsort(part)[::-1]
    print("\n--- %s summary ---" % tag)
    print("  mean rate                  %.3f Hz" % rate)
    print("  silent cells               %d / %d" % (int(np.isnan(first).sum()), n))
    print("  first spike per cell       median %.0f ms  (IQR %.0f-%.0f)"
          % (np.nanmedian(first), np.nanpercentile(first, 25),
             np.nanpercentile(first, 75)))
    print("  ISI  mean %.0f ms  sd %.0f ms  (n=%d)"
          % (isis.mean() if isis.size else np.nan,
             isis.std() if isis.size else np.nan, isis.size))
    print("  CV   within-cell %.3f (mean over %d cells)  pooled %.3f"
          % (cv_within, cvs.size, cv))
    print("  peak participation         %.1f%% at t = %.2f-%.2f s"
          % (100 * part[order[0]], edges[order[0]] / 1000.0,
             (edges[order[0]] + 50) / 1000.0))
    print("  top 5 participation bins:")
    for i in order[:5]:
        print("      %6.2f s   %5.1f%%" % (edges[i] / 1000.0, 100 * part[i]))
    # Participation restricted to the window where the flagship burst sits.
    win = (edges >= 3000) & (edges <= 8000)
    if win.any():
        j = np.argmax(part[win])
        print("  best bin in 3-8 s window   %.1f%% at %.2f s"
              % (100 * part[win][j], edges[win][j] / 1000.0))
    print("  Vm mean over run           %.2f mV (min %.2f, max %.2f)"
          % (vm_mean.mean(), vm_mean.min(), vm_mean.max()))
    print("  [K+]o                      %.2f - %.2f mM"
          % (ko_data["mean_ko"].min(), ko_data["mean_ko"].max()))

    np.savez_compressed(
        out,
        tag=tag,
        spike_times=np.array([spike_data[g] for g in range(n)], dtype=object),
        vm_times=vm_times, vm_mean=vm_mean, vm_p10=vm_p10, vm_p90=vm_p90,
        vm_traces=traces[::20],  # every 20th cell, for example traces
        ko_times=ko_data["times"], ko_mean=ko_data["mean_ko"],
        part_edges=edges, part=part,
        first_spike=first, rate=rate, cv=cv, cv_within=cv_within, cvs=cvs,
        exc_weight_scale=build_kwargs["exc_weight_scale"],
        inh_weight_scale=build_kwargs["inh_weight_scale"],
        noise_weight=build_kwargs["noise_weight"],
        neuron_is_inhibitory=topology["neuron_is_inhibitory"],
        cluster_assignments=topology["cluster_assignments"],
    )
    print("  saved -> %s" % out)


if __name__ == "__main__":
    main()
