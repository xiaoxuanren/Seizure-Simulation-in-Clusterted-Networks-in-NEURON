"""Session save/load in the EXACT format the LIF inference pipeline consumes.

The inference code (CCG baseline + learned-LIF, spike-only and voltage-augmented)
reads a session as a folder of compressed ``.npz`` files. This module writes the
identical layout so that pipeline runs against NEURON output unmodified:

    <save_dir>/<timestamp>/
        network_<timestamp>.npz     # ground-truth topology (saved once)
        recording000.npz            # one per recording (spikes; optional voltage)
        recording001.npz ...
        recording000_voltage.h5     # optional external voltage sidecar
        session_metadata.json       # session-level params

``save_network_structure`` and ``save_recording_data`` replicate the LIF
functions of the same name (``lif_simulation/session_io.py``); ``_resample_data``
and ``_organize_spike_data_by_cluster`` are vendored verbatim (in behaviour) from
``lif_simulation/analysis.py``.

Source of these two helpers and the file layout: LIF-Project
(``LIF-simulation``, branch ``chore/repo-cleanup``). The LIF project remains the
source of truth for the data contract; this is a faithful copy so the NEURON
project is self-contained.

HARD RULES (never rename these fields -- inference relies on them):
    network file : ``connections``, ``neuron_positions``, ``cluster_assignments``
    recording    : ``spike_times`` (ms), ``resampled_*``, ``duration``
Only NEW fields are added (log-normal / hub / E-I metadata).
"""

import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Vendored analysis helpers (source: LIF-Project lif_simulation/analysis.py)
# --------------------------------------------------------------------------- #
def _organize_spike_data_by_cluster(spike_data, cluster_info):
    """Group per-neuron spike trains into cluster-ordered lists.

    Vendored from the LIF project's ``organize_spike_data_by_cluster`` so saved
    ``cluster_spike_data`` matches byte-for-byte.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        cluster_info: Cluster metadata containing ``cluster_neuron_groups``.

    Returns:
        A list of per-cluster lists of spike-time arrays.
    """
    num_clusters = len(cluster_info["cluster_neuron_groups"])
    cluster_spike_data = [[] for _ in range(num_clusters)]
    for cluster_id in range(num_clusters):
        for neuron_id in cluster_info["cluster_neuron_groups"][cluster_id]:
            cluster_spike_data[cluster_id].append(spike_data[neuron_id])
    return cluster_spike_data


def _resample_data(spike_data, cluster_assignments, target_freq=10, duration=60000):
    """Bin spike times onto a fixed grid for saved raster-style outputs.

    Vendored from the LIF project's ``resample_data``.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        cluster_assignments: Legacy argument kept for API compatibility.
        target_freq: Resampling frequency in hertz.
        duration: Recording duration in milliseconds.

    Returns:
        A tuple ``(resampled_spikes, resampled_time_points, resampled_positions)``.
    """
    if target_freq <= 0:
        raise ValueError("target_freq must be positive.")
    interval_ms = 1000.0 / target_freq
    n_points = int(np.ceil(duration / interval_ms))
    n_neurons = len(spike_data)
    resampled_spikes = np.zeros((n_neurons, n_points), dtype=int)
    resampled_time_points = np.arange(n_points) * interval_ms
    for neuron_id, spikes in spike_data.items():
        if len(spikes) == 0:
            continue
        spike_times = np.asarray(spikes, dtype=float)
        bin_indices = np.floor(spike_times / interval_ms).astype(int)
        valid = bin_indices[(bin_indices >= 0) & (bin_indices < n_points)]
        if valid.size > 0:
            resampled_spikes[neuron_id, np.unique(valid)] = 1
    resampled_spike_positions = np.argwhere(resampled_spikes)
    return resampled_spikes, resampled_time_points, resampled_spike_positions


# --------------------------------------------------------------------------- #
# Lazy loader (mirrors LIF LoadedRecording)
# --------------------------------------------------------------------------- #
class LoadedRecording:
    """Lazy wrapper around one saved recording file and its filesystem path."""

    def __init__(self, path):
        self.path = path
        self._data = np.load(path, allow_pickle=True)

    @property
    def files(self):
        return self._data.files

    def __contains__(self, key):
        return key in self._data.files

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data[key] if key in self else default

    def close(self):
        self._data.close()


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_network_structure(connections, neuron_positions, cluster_info, weight_params, timestamp, save_dir):
    """Save ground-truth topology + spatial metadata (once per session).

    Replicates the LIF ``save_network_structure`` field layout and adds new
    fields for the biophysical builders (never renaming existing ones).

    Args:
        connections: Ground-truth connection table, one row per directed synapse
            ``[pre_id, post_id, weight, type('exc'|'inh')]``.
        neuron_positions: ``(N, 2)`` neuron coordinates.
        cluster_info: Cluster metadata dict from the topology builder.
        weight_params: The ``NeuronWeightParameters`` used (saved via ``vars``).
        timestamp: Session timestamp used to name the folder and file.
        save_dir: Root output directory for sessions.

    Returns:
        The path to the saved ``network_<timestamp>.npz`` file.
    """
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    filename = os.path.join(session_dir, f"network_{timestamp}.npz")

    cluster_neuron_groups = np.array(
        [np.array(g, dtype=int) for g in cluster_info["cluster_neuron_groups"]], dtype=object
    )
    save_dict = {
        # --- fields the inference relies on: NEVER rename ---
        "connections": np.asarray(connections, dtype=object),
        "neuron_positions": np.asarray(neuron_positions, dtype=float),
        "cluster_centers": np.asarray(cluster_info["cluster_centers"], dtype=float),
        "cluster_sizes": np.asarray(cluster_info["cluster_sizes"], dtype=int),
        "cluster_assignments": np.asarray(cluster_info["cluster_assignments"], dtype=int),
        "cluster_neuron_groups": cluster_neuron_groups,
        "weight_params": vars(weight_params) if not isinstance(weight_params, dict) else weight_params,
    }

    # --- new metadata fields (E/I labels for every neuron) ---
    if "neuron_is_inhibitory" in cluster_info:
        save_dict["neuron_is_inhibitory"] = np.asarray(cluster_info["neuron_is_inhibitory"], dtype=int)
    if "density" in cluster_info:
        save_dict["density"] = float(cluster_info["density"])
    if "topology_kind" in cluster_info:
        save_dict["topology_kind"] = str(cluster_info["topology_kind"])

    # --- log-normal build fields ---
    if "connection_propensity" in cluster_info:
        save_dict["connection_propensity"] = np.asarray(cluster_info["connection_propensity"], dtype=float)
        save_dict["ln_sigma"] = float(cluster_info.get("ln_sigma", 0.0))

    # --- discrete-hub build fields ---
    if "hub_neuron_ids" in cluster_info:
        save_dict["hub_neuron_ids"] = np.asarray(cluster_info["hub_neuron_ids"], dtype=int)
        save_dict["hub_fraction"] = float(cluster_info.get("hub_fraction", 0.1))
        save_dict["hub_between_prob"] = float(cluster_info.get("hub_between_prob", 0.4))
        save_dict["hub_weight_scale"] = float(cluster_info.get("hub_weight_scale", 1.5))
        save_dict["hub_reciprocal_factor"] = float(cluster_info.get("hub_reciprocal_factor", 2.0))
        save_dict["n_hub_connections"] = int(cluster_info.get("n_hub_connections", 0))

    np.savez_compressed(filename, **save_dict)
    print(f"Network structure saved to: {filename}")
    return filename


def _save_voltage_external_hdf5(voltage_data, session_dir, recording_idx):
    """Write voltage traces to an external HDF5 sidecar and return its metadata.

    Args:
        voltage_data: Voltage bundle with ``traces``/``times``/``sample_rate``.
        session_dir: Session directory to write the sidecar into.
        recording_idx: Recording index used to name the sidecar.

    Returns:
        A dict of the ``voltage_*`` fields to merge into the recording npz.
    """
    import h5py

    sidecar = os.path.join(session_dir, f"recording{recording_idx:03d}_voltage.h5")
    traces = np.asarray(voltage_data["traces"], dtype=np.float32)
    with h5py.File(sidecar, "w") as handle:
        handle.create_dataset("voltage_traces", data=traces, compression="gzip")
    return {
        "voltage_storage_backend": "hdf5_external",
        "voltage_hdf5_file": os.path.relpath(sidecar, session_dir),
        "voltage_hdf5_dataset": "voltage_traces",
        "voltage_n_samples": int(traces.shape[1]),
        "voltage_dtype": "float32",
    }


def save_recording_data(
    spike_data,
    voltage_data,
    cluster_info,
    recording_idx,
    timestamp,
    save_dir,
    target_freq=10,
    duration=60000,
    burst_windows=None,
    interburst_windows=None,
):
    """Persist one recording's spikes, resampled raster, and optional voltage.

    Replicates the LIF ``save_recording_data`` field layout.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        voltage_data: Optional voltage bundle from
            :func:`neuron_simulation.simulation.run_simulation`. When its
            ``storage_backend`` is ``"hdf5_external"`` an ``.h5`` sidecar is
            written; otherwise traces are stored inline.
        cluster_info: Saved cluster metadata (for clustered outputs).
        recording_idx: Recording index within the session.
        timestamp: Session timestamp used in filenames.
        save_dir: Root output directory.
        target_freq: Resampling frequency (Hz) for the saved raster.
        duration: Recording duration in milliseconds.
        burst_windows: Optional detected ``(start, end)`` network-burst windows.
        interburst_windows: Optional complement windows between bursts.

    Returns:
        The path to the saved ``recording<NNN>.npz`` file.
    """
    session_dir = os.path.join(save_dir, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    filename = os.path.join(session_dir, f"recording{recording_idx:03d}.npz")

    spike_times_list = np.array([spike_data[i] for i in range(len(spike_data))], dtype=object)
    cluster_spike_data = _organize_spike_data_by_cluster(spike_data, cluster_info)
    cluster_spike_data = np.array(
        [np.array(cluster, dtype=object) for cluster in cluster_spike_data], dtype=object
    )
    resampled_spikes, resampled_time_points, resampled_spike_positions = _resample_data(
        spike_data, cluster_info["cluster_assignments"], target_freq, duration
    )

    save_dict = {
        "spike_times": spike_times_list,
        "cluster_spike_data": cluster_spike_data,
        "resampled_spikes": resampled_spikes,
        "resampled_time_points": resampled_time_points,
        "resampled_cluster_assignments": np.asarray(cluster_info["cluster_assignments"], dtype=int),
        "resampling_frequency": target_freq,
        "resampling_interval_ms": 1000.0 / target_freq,
        "resampled_spike_positions": resampled_spike_positions,
        "recording_index": recording_idx,
        "timestamp": timestamp,
        "duration": duration,
    }

    if voltage_data is not None:
        save_dict["voltage_sample_rate"] = voltage_data["sample_rate"]
        save_dict["voltage_times"] = np.asarray(voltage_data["times"], dtype=np.float64)
        save_dict["voltage_units"] = voltage_data.get("units", "mV")
        backend = voltage_data.get("storage_backend", "inline_npz")
        if backend == "hdf5_external":
            save_dict.update(_save_voltage_external_hdf5(voltage_data, session_dir, recording_idx))
        else:
            save_dict["voltage_storage_backend"] = "inline_npz"
            save_dict["voltage_traces"] = np.asarray(voltage_data["traces"], dtype=np.float32)

    if burst_windows is not None:
        save_dict["burst_windows"] = np.asarray(burst_windows, dtype=float).reshape(-1, 2)
    if interburst_windows is not None:
        save_dict["interburst_windows"] = np.asarray(interburst_windows, dtype=float).reshape(-1, 2)

    np.savez_compressed(filename, **save_dict)
    print(
        f"Recording {recording_idx} saved to: {filename} "
        f"({len(spike_times_list)} neurons, {resampled_spikes.shape[1]} resampled points)"
    )
    return filename


# --------------------------------------------------------------------------- #
# Load (mirror LIF session_io loaders)
# --------------------------------------------------------------------------- #
def load_single_recording(filename):
    """Load one saved recording bundle from disk.

    Args:
        filename: Path to a saved ``recordingXXX.npz`` file.

    Returns:
        A :class:`LoadedRecording` wrapper around the stored arrays.
    """
    return LoadedRecording(filename)


def load_session_recordings(session_source):
    """Load every successful recording referenced by a session's metadata.

    Args:
        session_source: Path to ``session_metadata.json`` or its containing
            session directory.

    Returns:
        A tuple ``(recordings, metadata)``.
    """
    if os.path.isdir(session_source):
        metadata_file = os.path.join(session_source, "session_metadata.json")
    else:
        metadata_file = session_source
    with open(metadata_file, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    recordings = [
        load_single_recording(rec["file"]) for rec in metadata["recordings"] if rec["success"]
    ]
    return recordings, metadata


def find_session_folders(base_dir="NEURON data"):
    """Discover saved session folders under an output root.

    Args:
        base_dir: Root directory containing timestamped session subfolders.

    Returns:
        A sorted list of ``(timestamp, session_dir, metadata_file)`` tuples.
    """
    sessions = []
    if not os.path.exists(base_dir):
        return sessions
    for root, dirs, files in os.walk(base_dir):
        if "session_metadata.json" in files:
            sessions.append((os.path.basename(root), root, os.path.join(root, "session_metadata.json")))
            dirs[:] = []
    sessions.sort(key=lambda item: (item[0], item[1]))
    return sessions


def latest_session(base_dir="NEURON data"):
    """Return the most recent session directory under ``base_dir``.

    Args:
        base_dir: Root directory containing timestamped session subfolders.

    Returns:
        The path to the newest session directory, or ``None`` if none exist.
    """
    sessions = find_session_folders(base_dir)
    return sessions[-1][1] if sessions else None
