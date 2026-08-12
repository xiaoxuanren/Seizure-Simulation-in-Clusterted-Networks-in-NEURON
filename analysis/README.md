# analysis/

Standalone scripts for generating, validating and analysing the parallel NEURON
datasets. Each script derives the repo root from its own location, so run them
from anywhere:

```bash
py -3.9 analysis/<script>.py
```

**Requirements:** Python 3.9 with `numpy`, `scipy`, `scikit-learn`,
`matplotlib`. Most scripts read saved `.npz` outputs and need no NEURON; the
exceptions are the dataset-generation scripts (`dataset_*.py`,
`zeroedge_generate.py`), `decoupled_control.py` and
`measure_characterization.py`, which build and run the model.

## How paths work

`session_paths.py` is the single source of truth for where data lives:

```
notebooks/NEURON data parallel/<session>/<state>/recordingNNN.npz    raw data
                                                 network_*.npz       ground truth
notebooks/NEURON data parallel/<session>/results/<state>/<category>/ analysis output
```

Categories in use: `glm`, `bursts`, `ic_artifact`, `figures`, `other`. Raw-data
folders hold only data; everything a script writes goes under
`results/<state>/`. New figure renders land in `results/<state>/figures/` — the
curated PNGs in the root `figures/` directory are historical keepers (table at
the bottom).

Scripts select their dataset one of two ways:

* CLI: `--session <name> --state <name>` (via `session_paths.add_args`); an
  absolute path also works as `--session`.
* Env vars: `DATASET_SESSION` / `DATASET_STATE` for the scripts that read them.

Defaults are `IC-locked_flagship_200rec` / `normal`. To see what is actually on
disk (with recording counts):

```bash
python analysis/session_paths.py
```

Note the flagship 200-rec session ships `results/` only — its raw recordings
were never committed. The committed runnable raw dataset is
`IC-locked_flagship_spikeonly_50rec/normal` (50 recordings).

## Script catalog

### Dataset generation & validation

| script | what it does |
|---|---|
| `dataset_nb.py` | backend for `notebooks/dataset_generation.ipynb`: builds topology, writes `_session_config.pkl`, spawns the warmstart/generate/validate workers, polls their logs |
| `dataset_warmstart.py` | runs one warm-up simulation per state and snapshots full cell state into `<session>/_state_library_<state>.npz` |
| `dataset_generate.py` | worker: generates recordings for one state from the session config, restoring warm-start snapshots so recordings don't share an adaptation phase |
| `dataset_validate.py` | self-referential validation (no startup burst, stationary rate, flat V_rest, distinct recordings); writes `_validation_<state>.json/.png` |
| `zeroedge_generate.py` | generates the zero-edge (all synapses off) control recordings in session format — empty ground truth by construction |

### GLM connectivity analyses

These import `sparse_glm.py` from the repo root (see below).
`burstexcl_glm_arm.py` doubles as the shared constants module (bin/lag/l2
settings, `fdr_threshold`, `sum4_W`) imported by `fdr_duration_200_worker.py`,
`fdr_duration_200_combine.py` and `zeroedge_glm.py`.

| script | what it does |
|---|---|
| `glm_fit_current.py` | fits the shipped label-free sum4 GLM (with the A1 typing fix) and saves `glm_connectivity_sum4_5ms.npz` to `results/<state>/glm/` — input for the topology/recovery figures |
| `glm_scaling.py` | edge recovery vs recording duration at **oracle** operating points (best-F1 and ground-truth 10% FDR) |
| `glm_labelfree_scaling.py` | the same, but at the **actual label-free** jitter-FDR operating point (sum4 @ target 0.70), all three edge layers |
| `glm_labelfree_fdr_duration.py` | target-FDR (0.1–1.0) × duration grid: realized FDR / recall / F1 per cell |
| `fdr_duration_200_worker.py` | sharded worker extending the FDR × duration sweep to the full 200-rec session |
| `fdr_duration_200_combine.py` | merges the worker shards; writes the `fdrdur10to200_*` metrics + calibration/performance figures |
| `glm_predicted_topology.py` | predicted-vs-true wiring: spatial map + cluster-sorted adjacency (TP/FP/FN) |
| `glm_topology_zoom.py` | per-cluster zoomed wiring views so individual neurons and edges are countable by eye |
| `glm_cluster_recovery.py` | per-cluster recall heatmap; within- vs between-cluster recovery |
| `glm_distance_recovery.py` | recall/precision vs inter-neuron distance, within/between split, signal-vs-null strength |
| `glm_labelfree_fig.py` | earlier label-free predicted-vs-true figure (source of the curated root-`figures/` topology PNG) |
| `a1_typing_fix.py` | measures the E/I typing-rule fix (sign vs rank) at n=200; writes tables to `results/<state>/glm/` |
| `zeroedge_glm.py` | negative control: runs the shipped operating point on the zero-edge dataset vs matched flagship recordings |

### Burst / raster / control analyses

| script | what it does |
|---|---|
| `burst_windows_p035.py` | recomputes burst windows at the project's 0.35 gate; splits the IC-locked initialization event from spontaneous bursts |
| `burst_gate_sensitivity.py` | burst count vs acceptance gate sweep (bracketing vs acceptance disentangled) |
| `burstmarked_rasters.py` | rasters with the 0.35-gate bursts marked and classed (IC-locked vs spontaneous), overview + per-burst zoom |
| `burstexcl_glm_arm.py` | three-arm test: does the IC-locked first burst drive GLM edge recovery? (full / drop 0–6 s / drop 30–36 s); also the shared constants module noted above |
| `burstexcl_combine.py` | combines the three arms: table, Jaccard overlap, figure |
| `compare_states_glm.py` | normal vs seizure: does the label-free operating point transfer? calibration + F1 vs duration per state |
| `decoupled_control.py` | NEURON run: rebuilds the flagship network coupled vs zero-recurrence to test whether the early burst is network-generated; writes the `decoupled_control_*.npz` files in this directory |
| `plot_decoupled_control.py` | side-by-side coupled/decoupled figure (raster, participation, population Vm) from those npz files |
| `compare_control_windows.py` | window-matched participation comparison of the two control arms (console only; its loaders are imported by `control_periodicity.py`) |
| `control_periodicity.py` | burst period/decay per control arm via peak trains + autocorrelation |
| `spike_transmission.py` | pre→post spike-transmission probability (is one pre-spike enough? — no) |
| `voltage_traces.py` | Vm during vs outside a network burst; connected-pair traces + spike-triggered EPSP/IPSP |
| `render_rasters.py` | re-renders saved rasters with corrected title/dot size (CLI: `--folder --phenotype --dot-size ...`) |

### Characterization

| script | what it does |
|---|---|
| `measure_characterization.py` | measures every number in `MODEL_CHARACTERIZATION.md` from the built model; writes `characterization_measurements.json` next to itself |

`characterization_measurements.json` is that script's committed output — the
measured numbers behind the characterization doc.

### Results-tree tooling

| script | what it does |
|---|---|
| `session_paths.py` | the path registry itself; run directly to list datasets |
| `write_figures_index.py` | writes `<session>/results/FIGURES.md` mapping each results figure to its generating script and settings |
| `reorganize_results.py` | **completed one-off migration** (kept for the record): moved analysis outputs out of raw-data folders into `results/<state>/<category>/` |
| `migrate_sessions.py` | **completed one-off migration** (kept for the record): renamed the old `<state>/<timestamp>/` layout into named `IC-locked_*` sessions; its PLAN dict is the record of which timestamp became which session |

## Repo-root dependencies

The GLM scripts `sys.path`-insert the repo root and import by name:

* `sparse_glm.py` — the memory-efficient sparse lag-resolved ridge GLM library
  (`load_session`, `fit_B`, readouts, jitter nulls, `load_ground_truth`)
* `glm_connectivity.py` — GLM edge prediction and E/I typing
  (`infer_inhibitory` etc.)

The CLI runner is `scripts/run_inference.py` (GLM only; `--session` is
required and goes before the subcommand):

```bash
python scripts/run_inference.py --session "notebooks/NEURON data parallel/IC-locked_flagship_spikeonly_50rec/normal" glm --readout sum4
```

## Curated root `figures/`

These six PNGs are historical keepers referenced from the docs; new renders of
the same analyses go to `results/<state>/figures/` via the session-paths
versions of the scripts.

| figure | source script | what it shows |
|---|---|---|
| `glm_predicted_topology_labelfree_sum4_100rec.png` | `glm_labelfree_fig.py` | predicted-vs-true wiring (spatial map + cluster-sorted adjacency), TP/FP/FN |
| `glm_cluster_recovery_heatmap_sum4_100rec.png` | `glm_cluster_recovery.py` | per-cluster recall heatmap; within- vs between-cluster recovery |
| `glm_distance_recovery_sum4_100rec.png` | `glm_distance_recovery.py` | recall/precision vs inter-neuron distance |
| `spike_transmission_probability.png` | `spike_transmission.py` | pre→post spike-transmission probability, in- vs out-of-burst |
| `voltage_burst_vs_interburst.png` | `voltage_traces.py` | Vm during vs outside a network burst |
| `voltage_connected_pairs.png` | `voltage_traces.py` | connected-pair traces + spike-triggered EPSP/IPSP |
