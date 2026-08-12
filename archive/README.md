# Archive

Superseded or historical material moved here during the 2026-08 housekeeping pass
(branch `general-dataset-generator-cleaned`). Nothing in this folder is imported
or executed by the live code. Each item is kept for provenance; delete freely if
you no longer need the history.

## logs/

Console captures of completed July-2026-era runs. Their structured outputs
(metrics JSON, figures) already live in the per-session `results/` trees; no
script or document reads these logs.

| file | produced by | note |
|------|-------------|------|
| `bench.log` | `_bench_gpu.py` (older version) | superseded by `bench2.log` |
| `bench2.log` | `_bench_gpu.py` | benchmark verdict of record: GPU overtakes CPU only at batch ~1024 (12,291 vs 8,816 windows/s) |
| `glm_100rec.log`, `glm_calibrate.log`, `glm_peak.log`, `glm_run.log`, `glm_sum4_edges.log` | `_run_inference.py glm` | GLM lag-sweep / FDR-calibration / edge-prediction runs on the flagship session |
| `glm_sweep.log` | `glm_sweep.py` | the 175-config label-free hyperparameter sweep |
| `lif_full_K100.log`, `lif_full_K200.log` | `_run_lif_full.py` via the .bat launchers | full learned-LIF training runs (pipeline now retired) |

## Launchers (stale)

`run_lif_full.bat`, `run_lif_full_K200.bat` — Windows Scheduled-Task wrappers
for the learned-LIF runs. Both `cd` into the **old July repo copy**
(`D:\HAI Lab\2026\NEURON model\07 July 2026`) and use the pre-migration session
layout, so they never operated on this repository. Kept only to preserve the
launch parameters of the K=100/K=200 runs.

## Retired learned-LIF inference pipeline

`inference/` — the vendored LIF-project inference package
(`lif_inference/`, the `lif_simulation/` shim, and `adapter.py`), plus the
runners `_run_lif_full.py` and `_bench_gpu.py`. The learned-LIF/CCG pipeline is
no longer used; connectivity inference is now done by the sparse GLM
(`sparse_glm.py` + `glm_connectivity.py` at repo root). Note:
`glm_connectivity.py` was **extracted** from `inference/lif_inference/` to the
repo root before archiving because the live GLM pipeline imports it — it was a
local addition, never part of the vendored LIF code (see `inference/lif_inference/SOURCE.md`).

`lif_checkpoints/` (untracked, gitignored) — the two `lif_train_checkpoint.pt`
resume checkpoints (K=100 and K=200) removed from version control; they are only
consumable by the retired learned-LIF trainer. Also recoverable from the
`general-dataset-generator` branch history at
`notebooks/NEURON data parallel/IC-locked_flagship_200rec/results/normal/other/`.

## Superseded notebooks

`notebooks/parallel_dataset_generation.ipynb` and
`notebooks/parallel_dataset_generation_continue.ipynb` — earlier generations of
the parallel dataset workflow, superseded by `notebooks/dataset_generation.ipynb`
+ `analysis/dataset_nb.py` (parallel, resumable, warm-start). The `_continue`
notebook additionally targets a pre-migration session path that no longer
exists.

## One-off outputs

- `glm_sweep_results.json` — persisted results of the completed 175-config
  `glm_sweep.py` run (the sweep's checkpoint/resume file).
- `decoupled_control_decoupled_rec7.npz` — orphaned single-recording variant of
  the decoupled-control experiment; no script reads the `decoupled_rec7` tag.
