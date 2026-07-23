"""Parallel dataset generation -- run independent recordings across CPU cores.

Each recording is a *separate OS process* (bulletproof on Windows + Jupyter: no
pickling of notebook-defined functions, and every process gets its own fresh
NEURON global state, sidestepping the in-process reset problems). The recordings
share ONE fixed network (built once, deterministic from ``topology_seed``); they
differ only by the per-recording noise reseed ``rec_idx`` (``Random123(base, gid,
rec_idx)``) -- so each recording is a genuinely different trial, and the result
is equivalent to :func:`neuron_simulation.workflows.generate_dataset`, just
distributed across cores instead of run back-to-back.

A memory-aware worker cap keeps peak RAM = (workers x ~per_worker_gb) under the
currently-free RAM, so parallelism does not OOM the machine.

Usage (from a notebook)::

    from neuron_simulation import parallel_dataset as pds
    meta, session_dir = pds.generate_dataset_parallel(
        n_recordings=15, recording_duration=60000.0,
        topology_kwargs=CONFIG['topology'], build_kwargs=CONFIG['build'],
        state=states.normal_state(), dt=0.05, save_dir='NEURON data',
        max_workers=None, per_worker_gb=0.6, headroom_gb=4.0)

It can also be invoked as the per-recording worker (this is what the orchestrator
spawns)::  ``python parallel_dataset.py <config.pkl> <rec_idx> <summary.json>``
"""
import os
import sys
import json
import time
import pickle
import inspect
import subprocess
from datetime import datetime


# --------------------------------------------------------------------------- #
# Memory-aware worker count
# --------------------------------------------------------------------------- #
def free_ram_gb():
    """Best-effort available physical RAM in GB (psutil, else Windows API)."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    try:  # Windows fallback -- GlobalMemoryStatusEx
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 1e9
    except Exception:
        return float("inf")


def pick_worker_count(n_recordings, per_worker_gb=0.6, headroom_gb=4.0, max_workers=None):
    """Choose a safe number of concurrent worker processes.

    Bounded by: number of recordings, CPU count, an optional hard ``max_workers``,
    and -- crucially -- the currently-free RAM minus a headroom reserve.

    Returns ``(workers, free_gb, cpu)``.
    """
    cpu = os.cpu_count() or 1
    free = free_ram_gb()
    if free == float("inf"):
        mem_cap = cpu
    else:
        mem_cap = int((free - headroom_gb) // max(per_worker_gb, 1e-6))
    w = min(n_recordings, cpu, max(1, mem_cap))
    if max_workers:
        w = min(w, int(max_workers))
    return max(1, w), free, cpu


# --------------------------------------------------------------------------- #
# Worker: run ONE recording (mirrors generate_dataset's per-recording body)
# --------------------------------------------------------------------------- #
def run_one_recording(cfg, rec_idx):
    """Build the fixed network, reseed noise to ``rec_idx``, simulate, save."""
    from neuron_simulation import run_simulation, analysis
    from neuron_simulation.io import save_recording_data
    from neuron_simulation.noise import reseed_noise
    from neuron_simulation.network_builder import build_network
    from neuron_simulation.workflows import _bursts_to_windows

    topology = cfg["topology"]
    network = build_network(topology, noise_seed=cfg["noise_seed_base"], **cfg["build_kwargs"])
    reseed_noise(network.noise, rec_idx)  # Random123(base, gid, rec_idx) -> distinct trial

    spike_data, voltage_data, _ko = run_simulation(
        network, duration=cfg["recording_duration"], dt=cfg["dt"],
        discard_transient_ms=cfg["discard_transient_ms"],
        record_voltage=cfg["record_voltage"], voltage_dt=cfg["voltage_dt"])
    if voltage_data is not None:
        voltage_data["storage_backend"] = cfg["voltage_storage_backend"]

    bursts = analysis.detect_network_bursts(
        spike_data, network.n_neurons, cfg["recording_duration"],
        participation_threshold=cfg["participation_threshold"], burn_in_ms=0.0)
    burst_windows, interburst_windows = _bursts_to_windows(bursts, cfg["recording_duration"])
    stats = analysis.burst_statistics(bursts, cfg["recording_duration"], burn_in_ms=0.0)

    rec_file = save_recording_data(
        spike_data, voltage_data, cfg["cluster_info"], rec_idx, cfg["timestamp"], cfg["save_dir"],
        target_freq=cfg["target_freq"], duration=int(cfg["recording_duration"]),
        burst_windows=burst_windows, interburst_windows=interburst_windows)

    raster_file = raster_shuffled_file = None
    if cfg["save_rasters"]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from neuron_simulation import plotting
            session_dir = os.path.join(cfg["save_dir"], cfg["timestamp"])
            for shuffled, suffix in ((False, "raster"), (True, "raster_shuffled")):
                fig = plotting.plot_raster(
                    spike_data, network.n_neurons, cfg["recording_duration"],
                    is_inhibitory=topology.get("neuron_is_inhibitory"),
                    cluster_assignments=topology["cluster_assignments"], burn_in_ms=0.0,
                    title="recording %03d - %s%s" % (rec_idx, cfg["state_name"],
                                                     " (randomized rows)" if shuffled else ""),
                    randomize_rows=shuffled)
                fn = os.path.join(session_dir, "recording%03d_%s.png" % (rec_idx, suffix))
                fig.savefig(fn, dpi=120, facecolor="white", bbox_inches="tight"); plt.close(fig)
                if shuffled:
                    raster_shuffled_file = fn
                else:
                    raster_file = fn
        except Exception as exc:  # never fail a recording over a plot
            print("  [warn] raster skipped: %s" % exc)

    return {"index": rec_idx, "file": rec_file, "raster": raster_file,
            "raster_shuffled": raster_shuffled_file, "success": True,
            "n_bursts": int(stats["n_bursts"]), "burst_rate_hz": float(stats["burst_rate_hz"]),
            "mean_participation": float(stats["mean_participation"]),
            "num_spikes": int(sum(len(s) for s in spike_data.values()))}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def generate_dataset_parallel(
    n_recordings=50, recording_duration=60000.0, topology_kind="lognormal",
    topology_kwargs=None, build_kwargs=None, state=None, dt=0.05,
    discard_transient_ms=1000.0, record_voltage=False, voltage_dt=1.0,
    voltage_storage_backend="inline_npz", target_freq=10, save_dir="NEURON data",
    participation_threshold=0.35, noise_seed_base=1000, topology_seed=0,
    max_workers=None, per_worker_gb=0.6, headroom_gb=4.0, save_rasters=True,
    timestamp=None, poll_s=1.0,
):
    """Parallel drop-in for :func:`workflows.generate_dataset`.

    Builds the network once (shared ground truth), writes the same session bundle
    (network_*.npz, recording###.npz, session_metadata.json), but runs the N
    recordings concurrently as separate processes with a memory-aware cap.
    """
    from neuron_simulation.topology import build_topology_lognormal, build_topology
    from neuron_simulation.io import save_network_structure
    from neuron_simulation.network_builder import build_network
    from neuron_simulation import parameters as _params
    from neuron_simulation import states as states_module

    topology_kwargs = dict(topology_kwargs or {})
    topology_kwargs.setdefault("seed", topology_seed)
    build_kwargs = dict(build_kwargs or {})
    if state is None:
        state = states_module.normal_state()
    # Merge state overrides as *defaults* (explicit build_kwargs win), for any key
    # that is a real build_network parameter -- generalizes generate_dataset.
    bn_params = set(inspect.signature(build_network).parameters)
    for key, val in state.items():
        if key in bn_params and key != "state_name":
            build_kwargs.setdefault(key, val)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    if topology_kind == "lognormal":
        topology = build_topology_lognormal(**topology_kwargs)
    elif topology_kind == "discrete_hub":
        topology = build_topology(**topology_kwargs)
    else:
        raise ValueError("Unknown topology_kind=%r" % topology_kind)
    cluster_info = topology["cluster_info"]
    network_file = save_network_structure(
        topology["connections"], topology["neuron_positions"], cluster_info,
        topology["weight_params"], timestamp, save_dir)

    cfg = dict(
        topology=topology, cluster_info=cluster_info, build_kwargs=build_kwargs,
        state_name=state.get("state_name", "custom"), recording_duration=recording_duration,
        dt=dt, discard_transient_ms=discard_transient_ms, record_voltage=record_voltage,
        voltage_dt=voltage_dt, voltage_storage_backend=voltage_storage_backend,
        target_freq=target_freq, participation_threshold=participation_threshold,
        noise_seed_base=noise_seed_base, timestamp=timestamp, save_dir=save_dir,
        save_rasters=save_rasters)
    cfg_path = os.path.join(session_dir, "_worker_config.pkl")
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)

    workers, free_gb, cpu = pick_worker_count(n_recordings, per_worker_gb, headroom_gb, max_workers)
    print("=" * 74)
    print("PARALLEL SESSION %s  (state=%s)" % (timestamp, cfg["state_name"]))
    print("  %d recordings x %.0fs | N=%d cells, %d edges" % (
        n_recordings, recording_duration / 1000, topology["n_neurons"], len(topology["connections"])))
    print("  workers=%d  (cpu=%d, free=%.1f GB, ~%.2f GB/worker, headroom %.0f GB)%s" % (
        workers, cpu, free_gb, per_worker_gb, headroom_gb,
        "" if not max_workers else "  [capped at max_workers=%d]" % max_workers))
    print("  record_voltage=%s -> %s per-worker footprint" % (
        record_voltage, "LARGER" if record_voltage else "small"))
    print("=" * 74, flush=True)

    worker_py = os.path.abspath(__file__)
    pending = list(range(n_recordings))
    running = {}     # rec_idx -> (Popen, summary_path, log_fh)
    results = {}
    t0 = time.time()

    def _launch(idx):
        summ = os.path.join(session_dir, "_summary_%03d.json" % idx)
        logf = open(os.path.join(session_dir, "recording%03d.log" % idx), "w")
        proc = subprocess.Popen([sys.executable, worker_py, cfg_path, str(idx), summ],
                                stdout=logf, stderr=subprocess.STDOUT)
        return proc, summ, logf

    while pending or running:
        while pending and len(running) < workers:
            idx = pending.pop(0)
            running[idx] = _launch(idx)
            print("  [launch] recording %03d  (%d running, %d queued)" % (idx, len(running), len(pending)), flush=True)
        done = [i for i, (p, _, _) in running.items() if p.poll() is not None]
        for idx in done:
            proc, summ, logf = running.pop(idx)
            logf.close()
            if proc.returncode == 0 and os.path.exists(summ):
                with open(summ) as f:
                    results[idx] = json.load(f)
            else:
                results[idx] = {"index": idx, "file": None, "success": False,
                                "error": "worker exit=%s (see recording%03d.log)" % (proc.returncode, idx)}
            r = results[idx]
            print("  [done]   recording %03d  %s  spikes=%s bursts=%s  (%.0fs elapsed)" % (
                idx, "OK" if r.get("success") else "FAIL", r.get("num_spikes", "?"),
                r.get("n_bursts", "?"), time.time() - t0), flush=True)
        if running and not done:
            time.sleep(poll_s)

    session_metadata = dict(
        timestamp=timestamp, session_dir=session_dir, simulator="NEURON",
        n_recordings=n_recordings, recording_duration=recording_duration,
        num_neurons=topology["n_neurons"], num_connections=int(len(topology["connections"])),
        topology_kind=topology_kind, density=float(cluster_info.get("density", 0.0)),
        target_freq=target_freq, dt=dt, discard_transient_ms=discard_transient_ms,
        record_voltage=record_voltage, state=state, network_file=network_file,
        build_kwargs={k: v for k, v in build_kwargs.items()},
        parameters=_params.document(
            {**topology_kwargs, **build_kwargs, "dt": dt,
             "recording_duration": recording_duration, "n_recordings": n_recordings,
             "discard_transient_ms": discard_transient_ms,
             "participation_threshold": participation_threshold,
             "record_voltage": record_voltage, "target_freq": target_freq}),
        deviations_from_default={
            k: {"value": v, "default": d}
            for k, (v, d) in _params.deviations(
                {**topology_kwargs, **build_kwargs, "dt": dt,
                 "recording_duration": recording_duration,
                 "n_recordings": n_recordings}).items()},
        mode="spontaneous_bursting", background_input=True,
        participation_threshold=participation_threshold, parallel=True, n_workers=workers,
        recordings=[results[i] for i in sorted(results)])
    with open(os.path.join(session_dir, "session_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(session_metadata, f, indent=2, default=str)

    n_ok = sum(1 for r in results.values() if r.get("success"))
    print("=" * 74)
    print("SESSION COMPLETE: %d/%d recordings OK in %.0fs (%.1f min) -> %s" % (
        n_ok, n_recordings, time.time() - t0, (time.time() - t0) / 60, session_dir))
    print("=" * 74, flush=True)
    return session_metadata, session_dir


# --------------------------------------------------------------------------- #
# Entry point: the orchestrator spawns THIS as one-recording workers.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Make `import neuron_simulation` resolvable regardless of CWD.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    cfg_path, rec_idx, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    with open(cfg_path, "rb") as f:
        _cfg = pickle.load(f)
    try:
        res = run_one_recording(_cfg, rec_idx)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        res = {"index": rec_idx, "file": None, "success": False, "error": str(exc)}
    with open(summary_path, "w") as f:
        json.dump(res, f, default=str)
