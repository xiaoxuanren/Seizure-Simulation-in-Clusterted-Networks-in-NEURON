"""End-to-end workflows: topology -> network -> run -> save (+ ground truth).

Mirrors the LIF project's ``workflows.py`` (``sequential_simulation_individual_saves``):
build one network, save its ground-truth structure once, then run several
independent recordings and save each to disk in the inference-ready session
layout. Also provides :func:`run_single_state` for quick in-memory
normal-vs-4-AP comparisons without touching disk.

Independent recordings share the *same* wired network (so the ground truth is
constant) but use different Poisson-noise seeds, exactly like drawing repeated
trials from one culture.
"""

import json
import os
from datetime import datetime

from . import parameters as _params
from . import states as states_module
from .analysis import burst_statistics, detect_network_bursts
from .io import save_network_structure, save_recording_data
from .noise import reseed_noise
from .network_builder import build_network
from .simulation import run_simulation
from .topology import build_topology, build_topology_lognormal


def run_single_state(
    topology,
    state=None,
    build_kwargs=None,
    duration=8000.0,
    dt=0.025,
    discard_transient_ms=1000.0,
    record_voltage=False,
    voltage_dt=1.0,
    record_ko=True,
    noise_seed=1000,
):
    """Build a network in one pharmacological state, run it, and detect bursts.

    Intended for interactive verification and normal-vs-seizure comparisons;
    nothing is written to disk. Records mean [K+]o by default for seizure plots.

    Args:
        topology: A topology dict from :mod:`neuron_simulation.topology`.
        state: Optional state-override dict (e.g. from
            :func:`neuron_simulation.states.normal_state`); its ``gbar_kA_*``
            keys override ``build_kwargs``.
        build_kwargs: Base keyword arguments for
            :func:`neuron_simulation.network_builder.build_network`.
        duration: Kept recording duration in milliseconds.
        dt: Integration step in milliseconds.
        discard_transient_ms: Startup transient discarded before the kept window.
        record_voltage: Whether to record downsampled voltage.
        voltage_dt: Voltage sampling interval (ms).
        noise_seed: Base seed for the Poisson background.

    Returns:
        A dict with ``spike_data``, ``voltage_data``, ``bursts`` (list),
        ``burst_stats`` (dict), ``network``, and ``state_name``.
    """
    build_kwargs = dict(build_kwargs or {})
    build_kwargs["noise_seed"] = noise_seed
    state_name = "custom"
    if state is not None:
        state_name = state.get("state_name", "custom")
        # State values are DEFAULTS: an explicitly-passed build_kwargs entry wins.
        # (Previously this overrode build_kwargs, which would silently clobber an
        # explicit single-knob sahp_ainc_slow with the state's own value.)
        for key in states_module.STATE_BUILD_KEYS:
            if key in state:
                build_kwargs.setdefault(key, state[key])

    network = build_network(topology, **build_kwargs)
    spike_data, voltage_data, ko_data = run_simulation(
        network,
        duration=duration,
        dt=dt,
        discard_transient_ms=discard_transient_ms,
        record_voltage=record_voltage,
        voltage_dt=voltage_dt,
        record_ko=record_ko,
    )
    bursts = detect_network_bursts(spike_data, network.n_neurons, duration, burn_in_ms=0.0)
    stats = burst_statistics(bursts, duration, burn_in_ms=0.0)
    ko_note = ""
    if ko_data is not None:
        ko_note = f", [K+]o {ko_data['mean_ko'].min():.1f}-{ko_data['mean_ko'].max():.1f} mM"
    print(
        f"[{state_name}] {stats['n_bursts']} network bursts "
        f"({stats['burst_rate_hz']:.2f} Hz), mean participation "
        f"{stats['mean_participation']:.2f}{ko_note}"
    )
    return {
        "spike_data": spike_data,
        "voltage_data": voltage_data,
        "ko_data": ko_data,
        "bursts": bursts,
        "burst_stats": stats,
        "network": network,
        "state_name": state_name,
    }


def _bursts_to_windows(bursts, duration_ms):
    """Convert detected bursts into ``(start, end)`` burst/inter-burst windows.

    Args:
        bursts: List of burst dicts from :func:`detect_network_bursts`.
        duration_ms: Recording duration in milliseconds.

    Returns:
        A tuple ``(burst_windows, interburst_windows)`` of ``(start, end)`` lists.
    """
    burst_windows = [(b["start_ms"], b["end_ms"]) for b in bursts]
    interburst = []
    prev = 0.0
    for start, end in burst_windows:
        if start > prev:
            interburst.append((prev, start))
        prev = end
    if prev < duration_ms:
        interburst.append((prev, duration_ms))
    return burst_windows, interburst


def generate_dataset(
    n_recordings=50,
    recording_duration=60000.0,
    topology_kind="lognormal",
    topology_kwargs=None,
    build_kwargs=None,
    state=None,
    dt=0.05,
    discard_transient_ms=1000.0,
    record_voltage=False,
    voltage_dt=1.0,
    voltage_storage_backend="inline_npz",
    target_freq=10,
    save_dir="NEURON data",
    participation_threshold=0.35,
    noise_seed_base=1000,
    topology_seed=0,
):
    """Generate a multi-recording session saved in the inference-ready layout.

    Biophysical analogue of ``sequential_simulation_individual_saves``: builds a
    topology and network once, saves the ground-truth structure, then runs
    ``n_recordings`` independent recordings (distinct noise seeds) and saves each.

    Args:
        n_recordings: Number of independent recordings to generate.
        recording_duration: Kept duration of each recording in milliseconds.
        topology_kind: ``"lognormal"`` (preferred) or ``"discrete_hub"``.
        topology_kwargs: Extra keyword args for the chosen topology builder.
        build_kwargs: Keyword args for
            :func:`neuron_simulation.network_builder.build_network`.
        state: Optional state-override dict (defaults to
            :func:`neuron_simulation.states.normal_state`).
        dt: Integration step in milliseconds.
        discard_transient_ms: Startup transient discarded from each recording.
        record_voltage: Whether to record and save downsampled voltage.
        voltage_dt: Voltage sampling interval (ms).
        voltage_storage_backend: ``"inline_npz"`` or ``"hdf5_external"``.
        target_freq: Resampling frequency (Hz) for the saved raster.
        save_dir: Root output directory for the session bundle.
        participation_threshold: Fraction of neurons required for a network burst
            (used when tagging saved burst windows).
        noise_seed_base: Base Poisson seed; neuron ``gid`` in recording ``r``
            draws from the independent stream ``Random123(noise_seed_base, gid,
            r)``, so recordings are distinct trials rather than duplicates.
        topology_seed: Seed for the topology builder.

    Returns:
        A tuple ``(session_metadata, session_dir)``.
    """
    topology_kwargs = dict(topology_kwargs or {})
    topology_kwargs.setdefault("seed", topology_seed)
    build_kwargs = dict(build_kwargs or {})
    if state is None:
        state = states_module.normal_state()
    # Merge ALL state keys (this previously dropped sahp_ainc_slow/_fast, so a
    # seizure_state() passed here silently generated a NORMAL dataset).
    for key in states_module.STATE_BUILD_KEYS:
        if key in state:
            build_kwargs.setdefault(key, state[key])

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(save_dir, timestamp)

    print("=" * 70)
    print(f"NEURON SESSION {timestamp}  ({topology_kind}, state={state.get('state_name')})")
    print("=" * 70)

    # --- topology + network (built once; ground truth is constant) ---
    if topology_kind == "lognormal":
        topology = build_topology_lognormal(**topology_kwargs)
    elif topology_kind == "discrete_hub":
        topology = build_topology(**topology_kwargs)
    else:
        raise ValueError(f"Unknown topology_kind={topology_kind!r}")

    network = build_network(topology, noise_seed=noise_seed_base, **build_kwargs)
    cluster_info = topology["cluster_info"]

    network_file = save_network_structure(
        topology["connections"],
        topology["neuron_positions"],
        cluster_info,
        topology["weight_params"],
        timestamp,
        save_dir,
        provenance=dict(
            topology_seed=topology_kwargs.get("seed"),
            noise_seed_base=noise_seed_base,
            num_clusters=topology_kwargs.get("num_clusters"),
            space_size=topology_kwargs.get("space_size"),
            topology_kind=topology_kind,
            builder_params=topology_kwargs),
    )

    session_metadata = {
        "timestamp": timestamp,
        "session_dir": session_dir,
        "simulator": "NEURON",
        "n_recordings": n_recordings,
        "recording_duration": recording_duration,
        "num_neurons": topology["n_neurons"],
        "num_connections": int(len(topology["connections"])),
        "topology_seed": topology_kwargs.get("seed"),
        "noise_seed_base": noise_seed_base,
        "topology_kind": topology_kind,
        "density": float(cluster_info.get("density", 0.0)),
        "target_freq": target_freq,
        "dt": dt,
        "discard_transient_ms": discard_transient_ms,
        "record_voltage": record_voltage,
        "voltage_sample_rate": voltage_dt if record_voltage else None,
        "voltage_storage_backend": voltage_storage_backend if record_voltage else None,
        "state": state,
        # Provenance: the RESOLVED build parameters actually used. Without this the
        # single knob (sahp_ainc_slow) is not recorded anywhere in a saved session.
        "build_kwargs": {k: v for k, v in build_kwargs.items()},
        # Self-describing provenance: every resolved parameter with its units,
        # meaning, and effect of increasing (from neuron_simulation/parameters.py).
        "parameters": _params.document(
            {**topology_kwargs, **build_kwargs, "dt": dt,
             "recording_duration": recording_duration, "n_recordings": n_recordings,
             "discard_transient_ms": discard_transient_ms,
             "participation_threshold": participation_threshold,
             "record_voltage": record_voltage, "target_freq": target_freq}),
        # How this session differs from the canonical operating point.
        "deviations_from_default": {
            k: {"value": v, "default": d}
            for k, (v, d) in _params.deviations(
                {**topology_kwargs, **build_kwargs, "dt": dt,
                 "recording_duration": recording_duration,
                 "n_recordings": n_recordings}).items()},
        "build_config": network.config,
        "network_file": network_file,
        "mode": "spontaneous_bursting",
        "background_input": True,
        "participation_threshold": participation_threshold,
        "recordings": [],
    }

    # --- topology overview figure + connection stats (saved once) ---
    # Guarded and lazily imported so matplotlib stays optional for headless
    # data generation; a plotting failure must never break the dataset.
    try:
        from . import plotting
        import matplotlib.pyplot as _plt

        topo_stats = plotting.cluster_connection_stats(
            topology["connections"],
            topology["cluster_assignments"],
            topology["n_neurons"],
        )
        print(plotting.format_topology_stats(topo_stats))
        _fig = plotting.plot_topology_overview(
            topology["neuron_positions"],
            topology["connections"],
            topology["cluster_assignments"],
            is_inhibitory=topology.get("neuron_is_inhibitory"),
        )
        topology_figure = os.path.join(session_dir, f"topology_{timestamp}.png")
        _fig.savefig(topology_figure, dpi=130, facecolor="white", bbox_inches="tight")
        _plt.close(_fig)
        session_metadata["topology_stats"] = topo_stats
        session_metadata["topology_figure"] = topology_figure
        print(f"  topology figure -> {topology_figure}")
    except Exception as exc:  # pragma: no cover - never break generation
        print(f"  [warn] topology overview skipped: {exc}")
        session_metadata["topology_stats"] = None

    for rec_idx in range(n_recordings):
        print(f"\n--- recording {rec_idx + 1}/{n_recordings} ---")
        # Re-key every neuron's Poisson stream on the recording index so each
        # recording is a genuinely different (yet reproducible) trial. Each
        # generator's stream is Random123(base_seed, gid, rec_idx).
        reseed_noise(network.noise, rec_idx)

        try:
            spike_data, voltage_data, _ko_data = run_simulation(
                network,
                duration=recording_duration,
                dt=dt,
                discard_transient_ms=discard_transient_ms,
                record_voltage=record_voltage,
                voltage_dt=voltage_dt,
            )
            if voltage_data is not None:
                voltage_data["storage_backend"] = voltage_storage_backend
            bursts = detect_network_bursts(
                spike_data, network.n_neurons, recording_duration,
                participation_threshold=participation_threshold, burn_in_ms=0.0,
            )
            burst_windows, interburst_windows = _bursts_to_windows(bursts, recording_duration)
            stats = burst_statistics(bursts, recording_duration, burn_in_ms=0.0)
            print(
                f"  {stats['n_bursts']} bursts ({stats['burst_rate_hz']:.2f} Hz), "
                f"mean participation {stats['mean_participation']:.2f}"
            )

            recording_file = save_recording_data(
                spike_data,
                voltage_data,
                cluster_info,
                rec_idx,
                timestamp,
                save_dir,
                target_freq=target_freq,
                duration=int(recording_duration),
                burst_windows=burst_windows,
                interburst_windows=interburst_windows,
            )
            # per-recording raster (spike + population panel + burst shading)
            raster_file = None
            raster_shuffled_file = None
            try:
                from . import plotting as _plotting
                import matplotlib.pyplot as _plt

                _rfig = _plotting.plot_raster(
                    spike_data,
                    network.n_neurons,
                    recording_duration,
                    is_inhibitory=topology.get("neuron_is_inhibitory"),
                    cluster_assignments=topology["cluster_assignments"],
                    burn_in_ms=0.0,
                    title=f"recording {rec_idx:03d} - {state.get('state_name')}",
                )
                raster_file = os.path.join(session_dir, f"recording{rec_idx:03d}_raster.png")
                _rfig.savefig(raster_file, dpi=120, facecolor="white", bbox_inches="tight")
                _plt.close(_rfig)

                # companion raster with randomized neuron rows: if the burst
                # synchrony is genuinely network-wide it survives the shuffle;
                # if it were an artifact of grouping clusters by index it smears.
                _sfig = _plotting.plot_raster(
                    spike_data,
                    network.n_neurons,
                    recording_duration,
                    is_inhibitory=topology.get("neuron_is_inhibitory"),
                    cluster_assignments=topology["cluster_assignments"],
                    burn_in_ms=0.0,
                    title=f"recording {rec_idx:03d} - {state.get('state_name')} (randomized rows)",
                    randomize_rows=True,
                )
                raster_shuffled_file = os.path.join(session_dir, f"recording{rec_idx:03d}_raster_shuffled.png")
                _sfig.savefig(raster_shuffled_file, dpi=120, facecolor="white", bbox_inches="tight")
                _plt.close(_sfig)
                print(f"  raster -> {raster_file} (+ shuffled)")
            except Exception as exc:  # pragma: no cover - never break generation
                print(f"  [warn] raster skipped: {exc}")

            session_metadata["recordings"].append(
                {
                    "index": rec_idx,
                    "file": recording_file,
                    "raster": raster_file,
                    "raster_shuffled": raster_shuffled_file,
                    "success": True,
                    "n_bursts": stats["n_bursts"],
                    "burst_rate_hz": stats["burst_rate_hz"],
                    "num_spikes": int(sum(len(s) for s in spike_data.values())),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive, mirrors LIF
            import traceback

            traceback.print_exc()
            session_metadata["recordings"].append(
                {"index": rec_idx, "file": None, "success": False, "error": str(exc)}
            )

    metadata_file = os.path.join(session_dir, "session_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as handle:
        json.dump(session_metadata, handle, indent=2, default=str)

    print(f"\nSESSION COMPLETE -> {session_dir}")
    return session_metadata, session_dir
