"""Generate ONE recording of ONE sweep session (a single CHTC job).

Everything is deterministic from the sweep config: the topology is rebuilt
from (num_clusters, space_size, topology_seed), the background noise stream
for recording ``rec_idx`` is Random123(noise_seed_base, gid, rec_idx), and
``noise_seed_base = noise_seed_offset + noise_seed_step * topology_seed`` so
no two networks share background streams. The seeds are recorded in the
network npz (``topology_seed``, ``noise_seed_base``, ``num_clusters``,
``space_size``, ``builder_params_json``) and in the session metadata that
``collect.py`` assembles.

Usage (one job):
    python chtc/generate_one.py --sweep chtc/sweep_config.json \
        --session sweep_c50_seed01 --rec-idx 0 --out out

Output layout (under --out):
    <session>/recordingNNN.npz             the recording (voltage inline)
    <session>/recordingNNN_summary.json    per-recording stats for collect.py
    <session>/network_<session>.npz        ground truth (rec-idx 0 only,
                                           or --save-network)
"""
import argparse
import inspect
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_sweep(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def session_spec(sweep, session_name):
    """Resolve a session name -> (group dict, topology_seed)."""
    for group in sweep["groups"]:
        for seed in group["seeds"]:
            if "%s_seed%02d" % (group["prefix"], seed) == session_name:
                return group, seed
    raise SystemExit(
        "unknown session %r; expected one of:\n  %s"
        % (session_name, "\n  ".join(all_session_names(sweep))))


def all_session_names(sweep):
    return ["%s_seed%02d" % (g["prefix"], s)
            for g in sweep["groups"] for s in g["seeds"]]


def noise_seed_base_for(sweep, topology_seed):
    return int(sweep["noise_seed_offset"]) + int(sweep["noise_seed_step"]) * int(topology_seed)


def build_cfg(sweep, session_name):
    """Assemble the run_one_recording() worker config for a session."""
    from neuron_simulation.topology import build_topology_lognormal, build_topology
    from neuron_simulation.network_builder import build_network
    from neuron_simulation import states as states_module

    group, topology_seed = session_spec(sweep, session_name)

    topology_kwargs = dict(group.get("topology_overrides") or {})
    topology_kwargs["num_clusters"] = int(group["num_clusters"])
    topology_kwargs["space_size"] = float(group["space_size"])
    topology_kwargs["seed"] = int(topology_seed)

    kind = sweep.get("topology_kind", "lognormal")
    if kind == "lognormal":
        topology = build_topology_lognormal(**topology_kwargs)
    elif kind == "discrete_hub":
        topology = build_topology(**topology_kwargs)
    else:
        raise SystemExit("unknown topology_kind %r" % kind)

    if sweep["state"] == "normal":
        state = states_module.normal_state()
    elif sweep["state"] == "seizure":
        state = states_module.seizure_state(sweep["sahp_ainc_slow_seizure"]) \
            if "sahp_ainc_slow_seizure" in sweep else states_module.seizure_state()
    else:
        raise SystemExit("unknown state %r (use 'normal' or 'seizure')" % sweep["state"])

    # Explicit build overrides WIN; state values fill in as defaults -- the same
    # precedence as generate_dataset_parallel(build_kwargs=..., state=...).
    # build_overrides carries the project's operating point (noise_weight=0.007,
    # NOT the registry default) -- see sweep_config.json.
    build_kwargs = dict(sweep.get("build_overrides") or {})
    bn_params = set(inspect.signature(build_network).parameters)
    unknown = [k for k in build_kwargs if k not in bn_params]
    if unknown:
        raise SystemExit("build_overrides contains non-build_network keys: %r" % unknown)
    for key, val in state.items():
        if key in bn_params and key != "state_name":
            build_kwargs.setdefault(key, val)

    noise_seed_base = noise_seed_base_for(sweep, topology_seed)
    cfg = dict(
        topology=topology, cluster_info=topology["cluster_info"],
        build_kwargs=build_kwargs,
        state_name=state.get("state_name", "custom"),
        recording_duration=float(sweep["recording_ms"]),
        dt=float(sweep["dt"]),
        discard_transient_ms=float(sweep["discard_transient_ms"]),
        record_voltage=bool(sweep["record_voltage"]),
        voltage_dt=float(sweep["voltage_dt"]),
        voltage_storage_backend=sweep["voltage_storage_backend"],
        target_freq=sweep["target_freq"],
        participation_threshold=float(sweep["participation_threshold"]),
        noise_seed_base=noise_seed_base,
        timestamp=session_name,      # names the folder and the network npz
        save_dir=None,               # filled in by main() from --out
        save_rasters=bool(sweep.get("save_rasters", False)),
        raster_dot_size=20.0, raster_show_burst_count=True,
    )
    provenance = dict(
        topology_seed=int(topology_seed),
        noise_seed_base=int(noise_seed_base),
        num_clusters=int(group["num_clusters"]),
        space_size=float(group["space_size"]),
        topology_kind=kind,
        builder_params=topology_kwargs,
        build_params=build_kwargs,   # resolved build_network kwargs (state + overrides)
        state=state,                 # full state dict incl. the sahp knob value
    )
    return cfg, provenance


def write_session_provenance(session_dir, sweep, provenance):
    """Self-contained provenance sidecar (collect.py merges it into metadata)."""
    versions = {}
    for mod in ("numpy", "neuron"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "unavailable"
    doc = dict(provenance)
    doc["sweep_config"] = sweep
    doc["versions"] = versions
    path = os.path.join(session_dir, "session_provenance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str, sort_keys=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True, help="path to sweep_config.json")
    ap.add_argument("--session", help="e.g. sweep_c50_seed01")
    ap.add_argument("--rec-idx", type=int)
    ap.add_argument("--out", default="out", help="output root (default ./out)")
    ap.add_argument("--save-network", action="store_true",
                    help="also write network_<session>.npz (implied at --rec-idx 0)")
    ap.add_argument("--list-sessions", action="store_true")
    a = ap.parse_args()

    sweep = load_sweep(a.sweep)
    if a.list_sessions:
        print("\n".join(all_session_names(sweep)))
        return
    if a.session is None or a.rec_idx is None:
        ap.error("--session and --rec-idx are required (or use --list-sessions)")

    cfg, provenance = build_cfg(sweep, a.session)
    cfg["save_dir"] = a.out
    session_dir = os.path.join(a.out, a.session)
    os.makedirs(session_dir, exist_ok=True)

    if a.rec_idx == 0 or a.save_network:
        from neuron_simulation.io import save_network_structure
        topo = cfg["topology"]
        save_network_structure(
            topo["connections"], topo["neuron_positions"], topo["cluster_info"],
            topo["weight_params"], a.session, a.out, provenance=provenance)
        write_session_provenance(session_dir, sweep, provenance)

    from neuron_simulation.parallel_dataset import run_one_recording
    print("[%s rec %03d] N=%d neurons, %d edges, noise_seed_base=%d"
          % (a.session, a.rec_idx, cfg["topology"]["n_neurons"],
             len(cfg["topology"]["connections"]), cfg["noise_seed_base"]),
          flush=True)
    result = run_one_recording(cfg, a.rec_idx)

    result.update(session=a.session, topology_seed=provenance["topology_seed"],
                  noise_seed_base=provenance["noise_seed_base"])
    summary = os.path.join(session_dir, "recording%03d_summary.json" % a.rec_idx)
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("[%s rec %03d] DONE: %d spikes, %d bursts -> %s"
          % (a.session, a.rec_idx, result["num_spikes"], result["n_bursts"],
             result["file"]), flush=True)


if __name__ == "__main__":
    main()
