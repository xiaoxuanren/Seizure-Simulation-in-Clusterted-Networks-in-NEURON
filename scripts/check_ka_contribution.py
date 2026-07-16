"""Decide whether zeroing ``gbar_kA`` changes this network's output at all.

``tests/test_kA_characterization.py`` shows the shipped ``kA`` mechanism is inert
at the single-cell level: peak steady-state window conductance is 0.005% of the
``hh`` delayed rectifier, and over -65..-50 mV ``g_kA`` is under 2% of the leak
conductance. That is a *subthreshold* statement. It does not license deleting it:
``m^4`` is not negligible at the spike peak, so ``kA`` can still perturb spike
waveforms, and in a chaotic recurrent network any perturbation -- however tiny --
diverges into a different spike train.

This script settles that empirically. It runs the notebook's exact network twice,
identical in every respect including the noise seed, changing only ``gbar_kA``:

    run A: gbar_kA_exc = 0.006, gbar_kA_inh = 0.004   (the shipped values)
    run B: gbar_kA_exc = 0.0,   gbar_kA_inh = 0.0     (mechanism zeroed)

and compares ``spike_data`` **exactly** (``np.array_equal`` per neuron). If the
two are bitwise identical, zeroing ``gbar_kA`` is free and the numbers can be set
to 0. If they are not, the existing datasets would stop reproducing and the
shipped values must be retained.

Each arm runs in its own NEURON subprocess. Building two networks in one process
would leave the first network's 926 cells alive and integrating during the second
run, so a fresh process per arm keeps ``gbar_kA`` the only difference between them.

**The comparison is only meaningful on an active network.** The tuned normal state
is near-silent between bursts (~0.18 Hz burst rate), so a short run can produce two
identical *empty* spike trains and report a no-op that was never actually tested --
the divergence risk lives in spike waveforms *during* bursts. This script therefore
refuses to call a match conclusive unless both arms actually burst; see
:data:`MIN_SPIKES` / :data:`MIN_BURSTS`. Anything under ~30 s is likely to be
vacuous; the default 60 s spans ~11 bursts.

Exit codes:
    0  identical AND both arms active -> zeroing gbar_kA is a verified no-op
    1  spike trains diverge           -> retain the shipped values
    2  inconclusive (network too quiet to have tested anything)

Usage:
    python scripts/check_ka_contribution.py                   # full 60 s gate
    python scripts/check_ka_contribution.py --duration 60000  # explicit
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --------------------------------------------------------------------------- #
# The notebook's configuration, mirrored verbatim from
# notebooks/neuron_network_simulation.ipynb cell 2.
# --------------------------------------------------------------------------- #
SAHP_NORMAL = 0.01

TOPOLOGY_KWARGS = dict(
    num_clusters=50,
    neurons_per_cluster_range=(4, 40),
    inhibitory_probability=0.2,
    cluster_radius=1.0,
    space_size=15.0,
    seed=1,
    decay_sigma=3.0,
    max_connection_distance=6.0,
    cell_type_specific=True,
    p_ee_within=0.2,
    p_ee_between=0.1,
    p_ei_within=0.20,
    p_ei_between=0.08,
    p_ie_within=0.40,
    p_ii_within=0.50,
    within_cluster_prob=0.25,
    between_cluster_prob=0.06,
    ln_sigma=0.5,
    target_density=None,
)

BUILD_KWARGS = dict(
    synapse_model="ampa_nmda",
    exc_tau=5.0,
    tau_nmda=350.0,
    nmda_ratio=3.0,
    exc_weight_scale=2.0,
    inh_weight_scale=2.5,
    depression_d=0.2,
    tau_d=500.0,
    noise_rate=5.0,
    noise_weight=0.004,
    adapt=True,
    sahp_ainc_fast=0.005,
    sahp_tau_fast=300.0,
    sahp_ainc_slow=SAHP_NORMAL,
    sahp_tau_slow=6500.0,
    delay_per_distance=2.0,
)

SIM_KWARGS = dict(dt=0.05, discard_transient_ms=1000.0)
NOISE_SEED = 1000
LOOSE_PARTICIPATION_THRESHOLD = 0.35

#: A match only proves something if the network actually did something. Below
#: these, the run never exercised the spike waveforms where kA could matter, so
#: "identical" is vacuous rather than informative.
MIN_SPIKES = 1000
MIN_BURSTS = 3

#: The two arms: label -> (gbar_kA_exc, gbar_kA_inh).
ARMS = {"A": (0.006, 0.004), "B": (0.0, 0.0)}


def _build_topology():
    """Build the notebook's log-normal topology (seed 1)."""
    from neuron_simulation import topology as topology_module
    from neuron_simulation.topology import NeuronWeightParameters

    wp = NeuronWeightParameters()
    wp.within_exc_range = (0.0010, 0.0022)
    wp.between_exc_range = (0.0008, 0.0016)
    wp.within_inh_range = (0.0025, 0.0055)
    wp.between_inh_range = (0.0020, 0.0040)
    wp.use_lognormal = True
    wp.lognormal_sigma = 0.5

    return topology_module.build_topology_lognormal(weight_params=wp, **TOPOLOGY_KWARGS)


def run_arm(arm, duration, out_path):
    """Run one arm in this process and save its spike train to ``out_path``.

    Args:
        arm: ``"A"`` (shipped gbar_kA) or ``"B"`` (gbar_kA zeroed).
        duration: Kept recording duration (ms).
        out_path: Destination ``.npz`` path.

    Returns:
        None. Writes ``{"spikes_<gid>": np.ndarray}`` plus ``n_neurons``.
    """
    from neuron_simulation import states, workflows

    gbar_exc, gbar_inh = ARMS[arm]

    # run_single_state lets the state dict override build_kwargs' gbar_kA_* keys,
    # so the gbar override has to live in the state to take effect.
    state = dict(states.normal_state())
    state["gbar_kA_exc"] = gbar_exc
    state["gbar_kA_inh"] = gbar_inh
    state["state_name"] = "kA_%s_gbar%g" % (arm, gbar_exc)

    topo = _build_topology()
    result = workflows.run_single_state(
        topo,
        state=state,
        build_kwargs=dict(BUILD_KWARGS),
        duration=float(duration),
        noise_seed=NOISE_SEED,
        record_ko=False,
        **SIM_KWARGS
    )

    spike_data = result["spike_data"]
    payload = {"spikes_%d" % gid: np.asarray(times, dtype=float)
               for gid, times in spike_data.items()}
    payload["n_neurons"] = np.array(result["network"].n_neurons)
    np.savez(out_path, **payload)


def _load_spikes(path):
    """Load an arm's spike train back as ``{gid: np.ndarray}``."""
    with np.load(path) as data:
        n_neurons = int(data["n_neurons"])
        return {gid: data["spikes_%d" % gid] for gid in range(n_neurons)}


def _burst_stats(spike_data, n_neurons, duration):
    """Burst statistics under the notebook's loose (>=35% participation) detector."""
    from neuron_simulation import analysis

    bursts = analysis.detect_network_bursts(
        spike_data, n_neurons, duration,
        participation_threshold=LOOSE_PARTICIPATION_THRESHOLD, burn_in_ms=0.0,
    )
    return analysis.burst_statistics(bursts, duration, burn_in_ms=0.0)


def compare(spikes_a, spikes_b, duration):
    """Compare two spike trains exactly and report the deltas.

    Args:
        spikes_a: ``{gid: np.ndarray}`` from the shipped-gbar arm.
        spikes_b: ``{gid: np.ndarray}`` from the zeroed-gbar arm.
        duration: Recording duration (ms), for the burst-rate denominator.

    Returns:
        ``"identical"``, ``"diverged"``, or ``"inconclusive"`` (the run was too
        quiet for a match to have tested anything).
    """
    n_neurons = len(spikes_a)
    assert set(spikes_a) == set(spikes_b), "arms disagree on neuron ids"

    differing = [gid for gid in sorted(spikes_a)
                 if not np.array_equal(spikes_a[gid], spikes_b[gid])]
    identical = not differing

    total_a = sum(len(v) for v in spikes_a.values())
    total_b = sum(len(v) for v in spikes_b.values())
    stats_a = _burst_stats(spikes_a, n_neurons, duration)
    stats_b = _burst_stats(spikes_b, n_neurons, duration)
    rate_a = total_a / (n_neurons * duration / 1000.0)

    print()
    print("=" * 72)
    print("gbar_kA contribution check -- %d neurons, %.0f ms, noise_seed=%d"
          % (n_neurons, duration, NOISE_SEED))
    print("  run A: gbar_kA_exc=%g, gbar_kA_inh=%g" % ARMS["A"])
    print("  run B: gbar_kA_exc=%g, gbar_kA_inh=%g" % ARMS["B"])
    print("=" * 72)
    print()
    print("activity (arm A): %d spikes, mean rate %.3f Hz, %d bursts (%.3f Hz)"
          % (total_a, rate_a, stats_a["n_bursts"], stats_a["burst_rate_hz"]))
    print("activity (arm B): %d spikes, %d bursts" % (total_b, stats_b["n_bursts"]))
    print()
    print("BITWISE IDENTICAL: %s" % ("YES" if identical else "NO"))

    # A match on a silent network proves nothing: kA can only perturb the spike
    # waveform, so if nothing spiked, nothing was tested.
    too_quiet = (min(total_a, total_b) < MIN_SPIKES
                 or min(stats_a["n_bursts"], stats_b["n_bursts"]) < MIN_BURSTS)

    if identical and too_quiet:
        print()
        print("INCONCLUSIVE -- the network barely fired (need >=%d spikes and >=%d bursts"
              % (MIN_SPIKES, MIN_BURSTS))
        print("per arm). Two identical near-empty spike trains do not show that kA is a")
        print("no-op; they show the run never reached the bursts where kA could matter.")
        print("Re-run with a longer --duration.")
        return "inconclusive"

    if identical:
        print()
        print("Zeroing gbar_kA is a verified no-op on this network: every one of the")
        print("%d spike trains is bit-for-bit unchanged, across %d spikes in %d bursts."
              % (n_neurons, total_a, stats_a["n_bursts"]))
        return "identical"

    # Not identical -- quantify how far apart the arms are.
    max_delta = 0.0
    for gid in differing:
        a, b = spikes_a[gid], spikes_b[gid]
        n = min(len(a), len(b))
        if n:
            max_delta = max(max_delta, float(np.abs(a[:n] - b[:n]).max()))

    print()
    print("  neurons whose trains differ : %d / %d (%.1f%%)"
          % (len(differing), n_neurons, 100.0 * len(differing) / n_neurons))
    print("  max spike-time delta        : %.4f ms (over the common prefix)" % max_delta)
    print("  total spike count           : A=%d  B=%d  (delta %+d)"
          % (total_a, total_b, total_b - total_a))
    print()
    print("  burst statistics (loose detector, participation>=%.2f, burn_in=0):"
          % LOOSE_PARTICIPATION_THRESHOLD)
    print("    %-22s %12s %12s %12s" % ("", "A (0.006)", "B (0.0)", "delta"))
    for key in ("n_bursts", "burst_rate_hz", "mean_ibi_ms", "mean_duration_ms",
                "mean_participation"):
        va, vb = stats_a.get(key), stats_b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            print("    %-22s %12.4f %12.4f %+12.4f" % (key, va, vb, vb - va))
    print()
    print("The arms diverge, so zeroing gbar_kA would break reproduction of the")
    print("existing datasets. Retain the shipped values.")
    return "diverged"


def main():
    """Run both arms in isolated subprocesses and compare them exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60000.0,
                        help="kept recording duration in ms (default: 60000)")
    parser.add_argument("--run", choices=sorted(ARMS),
                        help="internal: run a single arm in this process")
    parser.add_argument("--out", help="internal: destination npz for --run")
    args = parser.parse_args()

    if args.run:
        run_arm(args.run, args.duration, args.out)
        return 0

    tmpdir = tempfile.mkdtemp(prefix="ka_check_")
    paths = {arm: os.path.join(tmpdir, "arm_%s.npz" % arm) for arm in ARMS}

    procs = {}
    for arm, path in paths.items():
        cmd = [sys.executable, os.path.abspath(__file__),
               "--run", arm, "--out", path, "--duration", str(args.duration)]
        print("[launch] arm %s -> %s" % (arm, path))
        procs[arm] = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)

    for arm, proc in procs.items():
        out, _ = proc.communicate()
        print("\n----- arm %s (exit %d) -----" % (arm, proc.returncode))
        print(out.strip())
        if proc.returncode != 0:
            raise SystemExit("arm %s failed" % arm)

    verdict = compare(_load_spikes(paths["A"]), _load_spikes(paths["B"]), args.duration)
    return {"identical": 0, "diverged": 1, "inconclusive": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
