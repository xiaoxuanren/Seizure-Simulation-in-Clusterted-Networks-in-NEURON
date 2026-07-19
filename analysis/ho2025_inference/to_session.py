#!/usr/bin/env python3
"""
to_session.py -- convert a flat Ho2025 export (``<label>.npz`` from
``export_npz.py``) into a session DIRECTORY in the EXACT format the LIF /
NEURON connectivity-inference pipeline consumes, so that pipeline runs on
Ho2025 (ModelDB 2018263) output unmodified.

Target layout (identical to ``neuron_simulation/io.py`` in the NEURON project):

    <out>/<timestamp>/
        network_<timestamp>.npz     # ground-truth topology + positions
        recording000.npz            # spikes (+ resampled raster); one per chunk
        recording001.npz ...
        session_metadata.json       # session-level params + recordings list

Field contract the inference actually reads (verified against
``inference/adapter.py`` + ``lif_inference/learned_lif_connectivity.py``):

    network file : connections            object [pre, post, weight, 'exc'|'inh']
                   neuron_positions       (N, 2) float
                   cluster_assignments    (N,)  int
                   (+ cluster_centers/sizes/neuron_groups, weight_params, E/I)
    recording    : spike_times            object, length N, per-neuron ms arrays
                   duration               float ms  (> 0)
                   resampled_spikes       (N, T) int  (validation gate)

Mapping from the flat Ho2025 export:
    A (N,N signed pre->post)  -> connections rows (sign -> 'exc'/'inh'),
                                 magnitude from W
    x, y (soma coords, um)    -> neuron_positions[:, 0], [:, 1]   (y = depth)
    cell_pop (E2/I2/E5)       -> cluster per population (KNN candidate
                                 selection is SPATIAL over positions, so the
                                 exact cluster labels only need to be valid)
    spkt/spkid (ms, gid)      -> per-neuron spike_times (remapped via cell_gid)

Runs anywhere with numpy (no NEURON / NetPyNE / torch needed):

    python to_session.py normal
    python to_session.py seizure --out "D:/HAI Lab/2026/NEURON model/07 July 2026/NEURON data"
    python to_session.py normal --chunks 5 --timestamp 20260717_ho_normal
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EPOPS = ("E2", "E5")          # excitatory populations (must match export_npz.py)
IPOPS = ("I2",)               # inhibitory populations


# --------------------------------------------------------------------------- #
# Writer helpers -- vendored verbatim (in behaviour) from the NEURON project's
# neuron_simulation/io.py so the on-disk bytes match field-for-field.
# --------------------------------------------------------------------------- #
def _organize_spike_data_by_cluster(spike_data, cluster_neuron_groups):
    out = [[] for _ in range(len(cluster_neuron_groups))]
    for cid, group in enumerate(cluster_neuron_groups):
        for nid in group:
            out[cid].append(spike_data[int(nid)])
    return out


def _resample_data(spike_data, target_freq, duration):
    if target_freq <= 0:
        raise ValueError("target_freq must be positive.")
    interval_ms = 1000.0 / target_freq
    n_points = int(np.ceil(duration / interval_ms))
    n_neurons = len(spike_data)
    resampled = np.zeros((n_neurons, n_points), dtype=int)
    time_points = np.arange(n_points) * interval_ms
    for nid, spikes in spike_data.items():
        if len(spikes) == 0:
            continue
        st = np.asarray(spikes, dtype=float)
        b = np.floor(st / interval_ms).astype(int)
        b = b[(b >= 0) & (b < n_points)]
        if b.size:
            resampled[nid, np.unique(b)] = 1
    return resampled, time_points, np.argwhere(resampled)


def _write_network(session_dir, timestamp, connections, positions, cluster_info,
                   weight_params):
    fn = os.path.join(session_dir, f"network_{timestamp}.npz")
    groups = np.array([np.asarray(g, dtype=int)
                       for g in cluster_info["cluster_neuron_groups"]], dtype=object)
    np.savez_compressed(
        fn,
        connections=np.asarray(connections, dtype=object),
        neuron_positions=np.asarray(positions, dtype=float),
        cluster_centers=np.asarray(cluster_info["cluster_centers"], dtype=float),
        cluster_sizes=np.asarray(cluster_info["cluster_sizes"], dtype=int),
        cluster_assignments=np.asarray(cluster_info["cluster_assignments"], dtype=int),
        cluster_neuron_groups=groups,
        neuron_is_inhibitory=np.asarray(cluster_info["neuron_is_inhibitory"], dtype=int),
        weight_params=weight_params,
        topology_kind="ho2025_cortex",
    )
    return fn


def _write_recording(session_dir, timestamp, rec_idx, spike_data, cluster_info,
                     target_freq, duration):
    fn = os.path.join(session_dir, f"recording{rec_idx:03d}.npz")
    spike_times = np.array([spike_data[i] for i in range(len(spike_data))], dtype=object)
    cs = _organize_spike_data_by_cluster(spike_data, cluster_info["cluster_neuron_groups"])
    cs = np.array([np.array(c, dtype=object) for c in cs], dtype=object)
    resampled, tpts, spos = _resample_data(spike_data, target_freq, duration)
    np.savez_compressed(
        fn,
        spike_times=spike_times,
        cluster_spike_data=cs,
        resampled_spikes=resampled,
        resampled_time_points=tpts,
        resampled_cluster_assignments=np.asarray(cluster_info["cluster_assignments"], dtype=int),
        resampling_frequency=target_freq,
        resampling_interval_ms=1000.0 / target_freq,
        resampled_spike_positions=spos,
        recording_index=rec_idx,
        timestamp=timestamp,
        duration=duration,
    )
    return fn, int(resampled.shape[1])


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def convert(flat, timestamp, duration, chunks, target_freq):
    cell_gid = np.asarray(flat["cell_gid"], dtype=np.int64)
    N = int(flat["N"]) if "N" in flat.files else len(cell_gid)
    gid2idx = {int(g): i for i, g in enumerate(cell_gid)}
    cell_pop = np.asarray(flat["cell_pop"]).astype(str)

    # positions: (x, y); y is cortical depth. KNN candidate selection is spatial,
    # so this 2-D projection defines which pairs are candidates (raise K to >= N
    # in the inference call to make candidate selection position-agnostic).
    x = np.asarray(flat["x"], dtype=float)
    y = np.asarray(flat["y"], dtype=float)
    positions = np.column_stack([x, y])

    # connections from the signed adjacency A (pre i -> post j), weight from W.
    A = np.asarray(flat["A"])
    W = np.asarray(flat["W"]) if "W" in flat.files else np.abs(A)
    connections = []
    pre_arr, post_arr = np.nonzero(A)
    for i, j in zip(pre_arr.tolist(), post_arr.tolist()):
        typ = "exc" if A[i, j] > 0 else "inh"
        connections.append([int(i), int(j), float(abs(W[i, j])), typ])

    # clusters by population (valid + meaningful; not used for candidate KNN).
    pops = sorted(set(cell_pop.tolist()))
    pop2c = {p: c for c, p in enumerate(pops)}
    cluster_assignments = np.array([pop2c[p] for p in cell_pop], dtype=int)
    groups = [np.where(cluster_assignments == c)[0].astype(int) for c in range(len(pops))]
    centers = np.array([positions[g].mean(axis=0) if len(g) else [0.0, 0.0]
                        for g in groups], dtype=float)
    sizes = np.array([len(g) for g in groups], dtype=int)
    neuron_is_inhibitory = np.array([1 if p in IPOPS else 0 for p in cell_pop], dtype=int)
    cluster_info = {
        "cluster_centers": centers,
        "cluster_sizes": sizes,
        "cluster_assignments": cluster_assignments,
        "cluster_neuron_groups": groups,
        "neuron_is_inhibitory": neuron_is_inhibitory,
    }

    # per-neuron spike trains (ms, already re-zeroed to window start by export_npz).
    spkt = np.asarray(flat["spkt"], dtype=float)
    spkid = np.asarray(flat["spkid"], dtype=np.int64)
    idx = np.array([gid2idx.get(int(g), -1) for g in spkid])
    keep = idx >= 0
    spkt, idx = spkt[keep], idx[keep]

    # duration: prefer explicit; else cover spikes + coarse [K+]o axis.
    if duration is None:
        cand = [float(spkt.max()) if spkt.size else 0.0]
        if "t_ko" in flat.files and np.asarray(flat["t_ko"]).size:
            cand.append(float(np.asarray(flat["t_ko"]).max()))
        duration = float(np.ceil(max(cand)))
    if duration <= 0:
        raise SystemExit("non-positive duration; pass --duration")

    weight_params = {
        "source": "ho2025_modeldb_2018263",
        "note": "NetPyNE synaptic weights; sign carried by connection 'type'",
    }

    edges = len(connections)
    n_inh = sum(1 for c in connections if c[3] == "inh")
    return dict(
        N=N, positions=positions, connections=connections, cluster_info=cluster_info,
        weight_params=weight_params, spkt=spkt, idx=idx, duration=duration,
        chunks=max(1, int(chunks)), target_freq=target_freq,
        stats=dict(edges=edges, n_inh=n_inh, n_exc=edges - n_inh,
                   n_spk=int(spkt.size), pops=pops),
    )


def write_session(conv, out_root, timestamp):
    session_dir = os.path.join(out_root, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    net_file = _write_network(session_dir, timestamp, conv["connections"],
                              conv["positions"], conv["cluster_info"],
                              conv["weight_params"])

    N, dur, k = conv["N"], conv["duration"], conv["chunks"]
    edges = np.arange(k + 1) * (dur / k)
    recordings = []
    for r in range(k):
        lo, hi = edges[r], edges[r + 1]
        m = (conv["spkt"] >= lo) & (conv["spkt"] < hi)
        spike_data = {i: [] for i in range(N)}
        for t, nid in zip((conv["spkt"][m] - lo).tolist(), conv["idx"][m].tolist()):
            spike_data[nid].append(t)
        spike_data = {i: np.asarray(v, dtype=float) for i, v in spike_data.items()}
        rec_dur = float(hi - lo)
        fn, T = _write_recording(session_dir, timestamp, r, spike_data,
                                 conv["cluster_info"], conv["target_freq"], rec_dur)
        recordings.append({"index": r, "file": fn, "success": True,
                           "num_spikes": int(m.sum()), "duration": rec_dur,
                           "resampled_points": T})

    meta = {
        "timestamp": timestamp,
        "session_dir": session_dir,
        "simulator": "Ho2025_NetPyNE_ModelDB_2018263",
        "n_recordings": k,
        "recording_duration": dur / k,
        "num_neurons": N,
        "num_connections": conv["stats"]["edges"],
        "topology_kind": "ho2025_cortex",
        "target_freq": conv["target_freq"],
        "record_voltage": False,
        "populations": conv["stats"]["pops"],
        "network_file": net_file,
        "recordings": recordings,
        "provenance": "converted from Ho2025 export_npz.py flat .npz via to_session.py",
    }
    with open(os.path.join(session_dir, "session_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return session_dir, net_file, recordings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="condition label (finds <label>.npz next to this "
                                   "script) OR a path to a flat .npz")
    ap.add_argument("--out", default=os.path.join(HERE, "inference_sessions"),
                    help="root dir to hold the <timestamp>/ session (default: "
                         "local_run/inference_sessions). Point at the NEURON "
                         "project's 'NEURON data' to use adapter.py 'latest'.")
    ap.add_argument("--timestamp", default=None,
                    help="session folder name (default: 20260101_ho_<label>)")
    ap.add_argument("--duration", type=float, default=None,
                    help="window length in ms (default: derived from spikes/[K+]o)")
    ap.add_argument("--chunks", type=int, default=1,
                    help="split the window into K contiguous recordings (default 1)")
    ap.add_argument("--freq", dest="freq", type=int, default=10,
                    help="resampling frequency in Hz for resampled_spikes (default 10)")
    args = ap.parse_args()

    src = args.label if args.label.endswith(".npz") else os.path.join(HERE, args.label + ".npz")
    if not os.path.isfile(src):
        raise SystemExit("flat npz not found: " + src)
    label = os.path.splitext(os.path.basename(src))[0]
    timestamp = args.timestamp or ("20260101_ho_" + label)

    flat = np.load(src, allow_pickle=True)
    conv = convert(flat, timestamp, args.duration, args.chunks, args.freq)
    session_dir, net_file, recs = write_session(conv, args.out, timestamp)

    s = conv["stats"]
    print("converted %s -> %s" % (src, session_dir))
    print("  N=%d | duration=%.0f ms | recordings=%d | freq=%d Hz"
          % (conv["N"], conv["duration"], conv["chunks"], conv["target_freq"]))
    print("  connections=%d  (exc=%d inh=%d) | populations->clusters=%s"
          % (s["edges"], s["n_exc"], s["n_inh"], s["pops"]))
    print("  spikes placed=%d | network=%s" % (s["n_spk"], os.path.basename(net_file)))


if __name__ == "__main__":
    main()
