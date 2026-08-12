"""Assemble CHTC sweep outputs into the repo's session layout and validate.

Point --src at the downloaded staging tree (one folder per session, as written
by job.sh). For each session this script:

  1. validates every recording in the source (readable npz with spike_times)
     BEFORE copying -- unreadable/corrupt files are reported and NOT copied,
     so a later `make_manifest.py --done-root` re-emits exactly those jobs;
  2. moves/copies files into  notebooks/NEURON data parallel/<session>/<state>/
     (including recordingNNN_voltage.h5 sidecars when present);
  3. writes session_metadata.json: seeds, the full resolved state/build
     parameters merged from session_provenance.json (written by the rec-0
     job), per-recording stats from the recordingNNN_summary.json files,
     and repo-relative file paths;
  4. checks the network npz is present with matching topology_seed /
     num_clusters.

Usage:
    python chtc/collect.py --src "D:/staging_download/neuron_sweeps"
    python chtc/collect.py --src ... --sessions sweep_c50_seed01 sweep_c50_seed02
    python chtc/collect.py --src ... --move       # move instead of copy
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "notebooks", "NEURON data parallel")


def load_sweep(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def expected_sessions(sweep):
    out = {}
    for g in sweep["groups"]:
        for s in g["seeds"]:
            name = "%s_seed%02d" % (g["prefix"], s)
            out[name] = dict(group=g, topology_seed=s,
                             n_recordings=int(g["n_recordings"]))
    return out


def rel(path):
    """Repo-relative path for metadata (portable across machines)."""
    try:
        return os.path.relpath(path, REPO).replace("\\", "/")
    except ValueError:            # different drive on Windows
        return path


def validate_recording(path):
    """Return (n_spikes, problem_or_None) without touching the voltage array."""
    try:
        d = np.load(path, allow_pickle=True)
        n_spk = int(sum(len(s) for s in d["spike_times"]))
    except Exception as exc:
        return None, "unreadable (%s: %s)" % (type(exc).__name__, exc)
    if n_spk == 0:
        return 0, "ZERO spikes"
    return n_spk, None


def collect_session(src_dir, session, spec, sweep, state, move=False):
    dest = os.path.join(DATA, session, state)
    os.makedirs(dest, exist_ok=True)
    transfer = shutil.move if move else shutil.copy2

    problems = []

    # --- provenance sidecar (rec-0 job) ------------------------------------
    provenance = {}
    prov_src = os.path.join(src_dir, "session_provenance.json")
    prov_dest = os.path.join(dest, "session_provenance.json")
    if os.path.exists(prov_src) and not os.path.exists(prov_dest):
        transfer(prov_src, prov_dest)
    if os.path.exists(prov_dest):
        with open(prov_dest, "r", encoding="utf-8") as fh:
            provenance = json.load(fh)
    else:
        problems.append("session_provenance.json MISSING (rec-0 job incomplete)")

    # --- network file ------------------------------------------------------
    net_src = os.path.join(src_dir, "network_%s.npz" % session)
    net_dest = os.path.join(dest, "network_%s.npz" % session)
    if os.path.exists(net_src) and not os.path.exists(net_dest):
        transfer(net_src, net_dest)
    if not os.path.exists(net_dest):
        problems.append("network npz MISSING (re-run the rec-0 job)")
    else:
        net = np.load(net_dest, allow_pickle=True)
        for field, want in (("topology_seed", spec["topology_seed"]),
                            ("num_clusters", spec["group"]["num_clusters"])):
            if field not in net.files:
                problems.append("network npz lacks %r" % field)
            elif int(net[field]) != int(want):
                problems.append("network %s=%s != expected %s"
                                % (field, net[field], want))

    # --- recordings: validate at the SOURCE, then copy ---------------------
    n = spec["n_recordings"]
    missing, bad, recordings = [], [], []
    for r in range(n):
        rec_name = "recording%03d.npz" % r
        rec_src = os.path.join(src_dir, rec_name)
        rec_dest = os.path.join(dest, rec_name)

        if not os.path.exists(rec_dest):
            if not os.path.exists(rec_src):
                missing.append(r)
                continue
            n_spk, problem = validate_recording(rec_src)
            if problem and n_spk is None:      # unreadable: do NOT copy
                bad.append((r, problem))
                continue
            transfer(rec_src, rec_dest)
            h5 = "recording%03d_voltage.h5" % r
            if os.path.exists(os.path.join(src_dir, h5)):
                transfer(os.path.join(src_dir, h5), os.path.join(dest, h5))
        else:
            n_spk, problem = validate_recording(rec_dest)
            if problem and n_spk is None:
                bad.append((r, problem + " [already in dataset -- delete it and recollect]"))
                continue
        if problem:                             # readable but suspicious (0 spikes)
            bad.append((r, problem))

        entry = {"index": r, "file": rel(rec_dest), "success": True,
                 "num_spikes": n_spk}
        summ = os.path.join(src_dir, "recording%03d_summary.json" % r)
        if os.path.exists(summ):
            with open(summ, "r", encoding="utf-8") as fh:
                entry.update(json.load(fh))
            entry["file"] = rel(rec_dest)
        recordings.append(entry)

    if missing:
        problems.append("missing %d/%d recordings (e.g. %s)"
                        % (len(missing), n, missing[:8]))
    for r, why in bad:
        problems.append("recording%03d: %s" % (r, why))

    # --- session metadata ---------------------------------------------------
    group = spec["group"]
    meta = dict(
        timestamp=session, session_dir=rel(dest), simulator="NEURON",
        generated_on="CHTC", n_recordings=n,
        recording_duration=sweep["recording_ms"],
        topology_seed=spec["topology_seed"],
        noise_seed_base=(int(sweep["noise_seed_offset"])
                         + int(sweep["noise_seed_step"]) * spec["topology_seed"]),
        num_clusters=group["num_clusters"], space_size=group["space_size"],
        topology_kind=sweep.get("topology_kind", "lognormal"),
        # full resolved provenance from the rec-0 job (state dict incl. the
        # sahp knob, build_network kwargs incl. noise_weight, lib versions)
        state=provenance.get("state", sweep["state"]),
        build_kwargs=provenance.get("build_params"),
        builder_params=provenance.get("builder_params"),
        versions=provenance.get("versions"),
        dt=sweep["dt"],
        discard_transient_ms=sweep["discard_transient_ms"],
        record_voltage=sweep["record_voltage"],
        voltage_sample_rate=sweep["voltage_dt"],
        voltage_storage_backend=sweep["voltage_storage_backend"],
        participation_threshold=sweep["participation_threshold"],
        target_freq=sweep["target_freq"],
        network_file=rel(net_dest), parallel=True,
        recordings=recordings,
    )
    with open(os.path.join(dest, "session_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="downloaded staging root (holds one folder per session)")
    ap.add_argument("--sweep", default=os.path.join(HERE, "sweep_config.json"))
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="subset of sessions (default: everything in the sweep)")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    a = ap.parse_args()

    sweep = load_sweep(a.sweep)
    state = sweep["state"]
    todo = expected_sessions(sweep)
    if a.sessions:
        todo = {k: v for k, v in todo.items() if k in set(a.sessions)}

    any_problem, found = False, 0
    for session, spec in sorted(todo.items()):
        src_dir = os.path.join(a.src, session)
        if not os.path.isdir(src_dir):
            print("%-22s SKIP (no folder under --src)" % session)
            continue
        found += 1
        problems = collect_session(src_dir, session, spec, sweep, state, a.move)
        if problems:
            any_problem = True
            print("%-22s PROBLEMS:" % session)
            for p in problems:
                print("    - %s" % p)
        else:
            print("%-22s OK (%d recordings + network + metadata)"
                  % (session, spec["n_recordings"]))

    if found == 0:
        print("\nERROR: no session folders found under --src %r -- wrong path?" % a.src)
        sys.exit(2)
    if any_problem:
        print("\nRe-emit only the missing/invalid jobs with:\n"
              "  python chtc/make_manifest.py --sweep \"%s\" --done-root \"%s\""
              % (a.sweep, DATA))
        sys.exit(1)


if __name__ == "__main__":
    main()
