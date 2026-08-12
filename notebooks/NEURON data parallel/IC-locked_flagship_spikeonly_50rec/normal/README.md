# Spike-only recordings — IC-locked_flagship_spikeonly_50rec (normal, 926 neurons, 50 recordings)

Voltage-stripped copies of 50 recordings from the full 200-recording flagship
session, `IC-locked_flagship_200rec` (whose raw data is not in this repo — only
its `results/` tree is committed), created so the dataset fits within GitHub's
file-size limits.

- **Dropped:** `voltage_traces` and the `voltage_*` metadata — ~222 MB per recording
  (98% of each file).
- **Kept (everything the spike-only inference uses):** `spike_times`,
  `resampled_spikes`, `cluster_spike_data`, `resampled_*`, `duration`,
  `burst_windows`/`interburst_windows`, `recording_index`, `timestamp`.
- **Ground truth:** `network_20260721_163430.npz` (connections, positions, clusters)
  plus `session_metadata.json`.
- **Size:** ~14 MB total (vs 6.4 GB with voltage — a ~460× reduction).

The live inference pipeline (sparse GLM; the learned-LIF runner is retired to
`archive/`) runs unmodified on this directory. `--session` is required, e.g.:

```
python scripts/run_inference.py --session "notebooks/NEURON data parallel/IC-locked_flagship_spikeonly_50rec/normal" glm --readout sum4
```

The full voltage recordings are retained locally and are not in git; the
voltage-augmented inference path requires them.
