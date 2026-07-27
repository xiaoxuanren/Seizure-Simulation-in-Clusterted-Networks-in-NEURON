# analysis/

Standalone scripts that regenerate the GLM connectivity + voltage figures from a
saved NEURON session. Each derives the repo root from its own location, so run
them from anywhere:

```bash
py -3.9 analysis/<script>.py
```

**Requirements:** Python 3.9 with `numpy`, `scipy`, `scikit-learn`, `matplotlib`
(no NEURON needed — these read saved `.npz` outputs). The GLM scaling scripts
import `sparse_glm.py` from the repo root. Figures are written to `figures/` or
into the session folder; metrics to `*_metrics.json`.

Default session: `notebooks/NEURON data parallel/normal/20260721_163430`
(the 926-neuron, 100-recording normal flagship). Edit `SESSION`/`SD` in a script
to point elsewhere.

| script | produces | what it shows |
|---|---|---|
| `glm_scaling.py` | `glm_scaling_vs_duration.png`, `glm_scaling_metrics.json` | edge recovery (TP / precision / recall / F1 / AUC / AP) vs recording duration, at **oracle** operating points (best-F1 and ground-truth 10% FDR) |
| `glm_labelfree_scaling.py` | `glm_labelfree_scaling_vs_duration.png`, `glm_labelfree_scaling_metrics.json` | the same, but at the **actual label-free** jitter-FDR operating point (sum4 @ target 0.70); shows realized FDR drifting with data size |
| `glm_labelfree_fig.py` | `figures/glm_predicted_topology_labelfree_sum4_100rec.png` | predicted-vs-true wiring (spatial map + cluster-sorted adjacency), TP/FP/FN |
| `glm_cluster_recovery.py` | `figures/glm_cluster_recovery_heatmap_sum4_100rec.png` | per-cluster recall heatmap; within- vs between-cluster recovery (common-input confound) |
| `glm_distance_recovery.py` | `figures/glm_distance_recovery_sum4_100rec.png` | recall/precision vs inter-neuron distance, within/between split, signal-vs-null strength |
| `voltage_traces.py` | `figures/voltage_burst_vs_interburst.png`, `figures/voltage_connected_pairs.png` | Vm during vs outside a network burst; connected-pair traces + spike-triggered EPSP/IPSP |
| `spike_transmission.py` | `figures/spike_transmission_probability.png` | pre→post spike-transmission probability (is one pre-spike enough? — no, ~5%) |
| `render_rasters.py` | re-rendered `recording###_raster*.png` | re-titles/re-styles saved rasters from a session (CLI: `--folder --phenotype --dot-size ...`) |

The GLM readout/FDR machinery itself lives in
`inference/lif_inference/glm_connectivity.py` and `sparse_glm.py`; the CLI runner
is `_run_inference.py` (`glm --calibrate` / `--edges --target-fdr ...`).
