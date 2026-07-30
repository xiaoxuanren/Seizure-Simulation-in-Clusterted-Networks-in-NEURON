"""Migrate pre-existing datasets into the session layout.

Old layout   NEURON data parallel/<state>/<timestamp>/recordingNNN.npz
New layout   NEURON data parallel/<session>/<state>/recordingNNN.npz

The datasets being moved were all generated with ``h.finitialize(-65)``, which
starts every cell in the same zero-adaptation condition -- hence the
``IC-locked_`` prefix, distinguishing them from warm-start datasets.

Moves are renames on one filesystem, so they are instant and copy no data.

A ``_session_config.pkl`` is synthesised from each source ``_worker_config.pkl``
so the migrated sessions are readable by the same tooling as new ones. Where a
session has two states, their topologies are asserted byte-identical first.

    python migrate_sessions.py --dry-run     # show every move, change nothing
    python migrate_sessions.py               # do it
"""

import argparse
import os
import pickle
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DATA = os.path.join(REPO, "notebooks", "NEURON data parallel")

#: session -> {state: source directory relative to DATA}
PLAN = {
    "IC-locked_flagship_200rec": {
        "normal": "normal/20260721_163430",
        "seizure": "seizure/20260723_150907",
    },
    "IC-locked_flagship_spikeonly_50rec": {
        "normal": "normal/20260721_163430_spikeonly",
    },
    "IC-locked_zeroedge_control_15rec": {
        "normal": "zeroedge_control_15rec",
    },
    "IC-locked_exploratory": {
        "normal_20260721_151926": "normal/20260721_151926",
        "normal_20260724_112610": "normal/20260724_112610",
        "seizure_20260722_212536": "seizure/20260722_212536",
        "seizure_20260724_123722": "seizure/20260724_123722",
    },
}

#: never touched -- an active run, or the live new-pipeline session
PROTECTED = ("normal/20260730_095119", "seizure/20260730_100426", "dataset_v1")


def _topo_fingerprint(topo):
    c = np.asarray(topo["connections"], dtype=object)
    return (int(topo["n_neurons"]), len(c),
            hash(np.asarray(topo["neuron_is_inhibitory"]).tobytes()),
            hash(np.asarray(topo["cluster_assignments"]).tobytes()))


def synth_config(session, states_src, session_path):
    """Build a session config from the source worker configs.

    Only fields that are actually recoverable are filled. ``topology`` kwargs are
    NOT recoverable -- the pickle stores the materialised graph, not the
    parameters that produced it -- so it is left None and flagged.
    """
    per_state, fingerprints = {}, {}
    for state, src in states_src.items():
        wc = os.path.join(src, "_worker_config.pkl")
        if not os.path.exists(wc):
            print("    [warn] no _worker_config.pkl in %s" % os.path.basename(src))
            continue
        with open(wc, "rb") as fh:
            per_state[state] = pickle.load(fh)
        fingerprints[state] = _topo_fingerprint(per_state[state]["topology"])

    if not per_state:
        return None
    if len(set(fingerprints.values())) > 1:
        raise SystemExit(
            "REFUSING to merge '%s': states have different topologies\n  %s"
            % (session, fingerprints))

    ref = per_state[sorted(per_state)[0]]
    build = dict(ref["build_kwargs"])
    build.pop("sahp_ainc_slow", None)
    cfg = dict(
        session=session,
        states={s: float(c["build_kwargs"]["sahp_ainc_slow"])
                for s, c in per_state.items()},
        states_to_run=sorted(per_state),
        topology_kind=ref["topology"].get("topology_kind", "lognormal"),
        topology=None,                 # graph is materialised; kwargs not stored
        build=build,
        sim=dict(dt=ref.get("dt", 0.05),
                 discard_transient_ms=ref.get("discard_transient_ms", 1000.0),
                 target_freq=ref.get("target_freq", 10),
                 participation_threshold=ref.get("participation_threshold", 0.35)),
        n_recordings=None, n_workers=None,
        recording_ms=ref.get("recording_duration", 60000.0),
        noise_seed_base=ref.get("noise_seed_base", 1000),
        voltage="all", voltage_probe_n=None, voltage_dt=ref.get("voltage_dt", 1.0),
        # These datasets predate the warm start.
        init_mode="finitialize",
        warmup_ms=None, snapshot_times=None, discard_extra_ms=0.0,
        migrated_from={s: os.path.relpath(p, DATA) for s, p in states_src.items()},
        note=("Migrated from the timestamped layout. Generated with "
              "h.finitialize(-65), so every recording starts from an identical "
              "zero-adaptation state. topology kwargs are not recoverable; the "
              "materialised graph is stored instead."),
    )
    return dict(config=cfg, topology=ref["topology"],
                cluster_info=ref["cluster_info"], session_dir=session_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print("data root: %s" % DATA)
    print("protected (never touched): %s\n" % ", ".join(PROTECTED))
    moves, n_rec_total = [], 0

    for session, states_src in PLAN.items():
        session_path = os.path.join(DATA, session)
        print("=== %s ===" % session)
        resolved = {}
        for state, rel in states_src.items():
            src = os.path.join(DATA, rel.replace("/", os.sep))
            dst = os.path.join(session_path, state)
            if rel in PROTECTED:
                raise SystemExit("refusing to move protected path %s" % rel)
            if not os.path.isdir(src):
                print("  [skip] missing: %s" % rel)
                continue
            if os.path.exists(dst):
                print("  [skip] destination exists: %s/%s" % (session, state))
                continue
            n = len([f for f in os.listdir(src)
                     if f.startswith("recording") and f.endswith(".npz")])
            n_rec_total += n
            print("  %-26s <- %-40s (%d recordings)" % (state, rel, n))
            moves.append((src, dst))
            resolved[state] = src

        if not a.dry_run:
            os.makedirs(session_path, exist_ok=True)
            for src, dst in [(s, d) for s, d in moves if s in resolved.values()]:
                shutil.move(src, dst)
            bundle = synth_config(session,
                                  {s: os.path.join(session_path, s) for s in resolved},
                                  session_path)
            if bundle:
                with open(os.path.join(session_path, "_session_config.pkl"), "wb") as fh:
                    pickle.dump(bundle, fh)
                print("  wrote _session_config.pkl (states: %s)"
                      % ", ".join(sorted(bundle["config"]["states"])))
            moves = []
        print("")

    print("%s: %d directories, %d recordings"
          % ("WOULD MOVE" if a.dry_run else "MOVED", len(moves) if a.dry_run else 0,
             n_rec_total))
    if a.dry_run:
        print("\n(dry run - nothing changed)")


if __name__ == "__main__":
    main()
