# Spike-only recordings — 20260721_163430 (normal, 926 neurons, 50 recordings)

Voltage-stripped copies of the 50 recordings in the sibling `20260721_163430/`
session, created so the dataset fits within GitHub's file-size limits.

- **Dropped:** `voltage_traces` and the `voltage_*` metadata — ~222 MB per recording
  (98% of each file).
- **Kept (everything the spike-only inference uses):** `spike_times`,
  `resampled_spikes`, `cluster_spike_data`, `resampled_*`, `duration`,
  `burst_windows`/`interburst_windows`, `recording_index`, `timestamp`.
- **Ground truth:** `network_20260721_163430.npz` (connections, positions, clusters)
  plus `session_metadata.json`.
- **Size:** ~14 MB total (vs 6.4 GB with voltage — a ~460× reduction).

The inference pipeline runs unmodified on this directory, e.g.:

```
python _run_lif_full.py --session "notebooks/NEURON data parallel/normal/20260721_163430_spikeonly"
python _run_inference.py --session "notebooks/NEURON data parallel/normal/20260721_163430_spikeonly" glm --edges
```

The full voltage recordings are retained locally and are not in git; the
voltage-augmented inference path requires them.
