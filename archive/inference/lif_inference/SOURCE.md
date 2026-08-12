# Vendored inference package — source: LIF-Project

This directory is a **verbatim vendored copy** of the `lif_inference/` package
from the LIF project:

- Repository: https://github.com/xiaoxuanren/LIF-Project
- Path: `LIF-simulation/lif_inference/`
- Branch: `chore/repo-cleanup`

It is included here so this NEURON repository is self-contained and runnable
without cloning the LIF project. **The LIF project remains the source of truth
for the inference code.** Do not fork the science here: fix bugs and add
features in LIF-Project, then re-vendor.

Nothing in this folder has been modified. The NEURON project couples to it only
through the **data format** (see `neuron_simulation/io.py`), which reproduces the
LIF session layout exactly, so this pipeline consumes NEURON output unchanged.

Public entry points used by `inference/adapter.py`:

- `learned_lif_connectivity.run_pipeline` — spike-only learned-LIF inference
- `voltage_augmented_learned_lif_connectivity.run_pipeline` — voltage-augmented
- `ccg_baseline.run_ccg_baseline` — training-free CCG baseline
- `shared_data.build_ground_truth`, `connectivity_metrics` — ground truth + metrics
