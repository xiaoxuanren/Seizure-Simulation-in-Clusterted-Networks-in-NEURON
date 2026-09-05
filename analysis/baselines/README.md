# Baseline connectivity inference vs the shipped ridge GLM

Standard spike-train connectivity methods from Kobayashi & Shinomoto (2025,
*Neurosci Res* 215:37, "Inference of monosynaptic connections from parallel
spike trains: A review"), run on every sweep dataset (20 networks x 2 states)
in both arms (full recordings / burst windows excluded), and scored
like-for-like against the shipped GLM. Results and figures live in
`notebooks/NEURON data parallel/sweep_summary/baselines/` (`RESULTS.md`,
`metrics.csv`, `metrics_wide.csv`, `figures/`, `maps/`).

## Layout

| file | role |
|---|---|
| `accumulate.py` | the shared accumulator: one pass of per-recording sparse products gives the lag products `C[l][pre, post]` (raw CCG), the rate-normalised Pearson form, and the ridge Gram/RHS of the shipped GLM (`F X^T` at lag l **is** `C[l]`). Fixes the orientation invariant. |
| `test_accumulate.py` | orientation, boundary and reproduction tests: `solve_ridge(gram())` is bit-identical to `sparse_glm.fit_B` and to the shipped `glm_connectivity_sum4_5ms.npz`. |
| `common.py` | `Session` / `SpikeData`: passes at any bin width, cached accumulations, the shared jitter surrogates (same generator stream as the shipped null), burst-window recomputation at the 0.35 gate. |
| `labelfree.py` | the shipped jitter-null FDR rule (imported from `chtc/analysis_one.py`) and the typed protocol; spikes only. |
| `evaluate.py` | the only place ground truth is read; reuses `chtc/analysis_one.py`'s metrics; four read-outs, two denominators, distance / weight / delay recovery. |
| `registry.py` | the method contract, the registry names and parameter VARIANTS, the skipped methods. |
| `<method>.py` | one module per method (see below). |
| `run_dataset.py` | one dataset, all methods, both arms -> `<session>/results/<state>/baselines/`. |
| `run_sweep.py` | local process pool over all (session, state). |
| `gate_delay.py` | delay-recovery validation gate (r >= 0.3). |
| `aggregate.py` | `metrics.csv` (long) + `metrics_wide.csv` + `structure.json` + `compute.json`. |
| `figures.py` | maps, ROC/PR overlays, AP heatmaps, burst gap, gate, TE bin-width. |
| `write_results.py` | `RESULTS.md`. |

CHTC: `chtc/make_baselines_manifest.py` (manifest + tarballs, built locally),
`chtc/baselines.sub`, `chtc/baselines_job.sh`, `chtc/baselines_one.py`,
`chtc/collect_baselines.py`; submit with `SWEEP=baselines bash chtc/submit_all.sh NETID`.

## Methods

| registry name | module | tier | source |
|---|---|---|---|
| `glm_ridge` | `glm_ridge.py` | reference | this repo (sum4, jitter-null FDR 0.70) |
| `pearson0` | `pearson0.py` | 1 | zero-lag Pearson, undirected floor |
| `pearson_lag` | `pearson_lag.py` | 1 | max-|r| over lags 1-6 at 5 ms, signed |
| `ccg_classical` | `ccg_classical.py` | 1 | Perkel, Gerstein & Moore 1967 |
| `ccg_jitter` | `ccg_jitter.py` | 1 | Fujisawa et al. 2008; Amarasingham et al. 2012 -- the shipped null on the raw CCG |
| `tspe` | `tspe.py` | 1 | De Blasi et al. 2019 |
| `fncch` | `fncch.py` | 1 | Pastore et al. 2018 |
| `glmcc`, `glmcc_d14` | `glmcc.py` | 2 | Kobayashi et al. 2019 (d = 1-4 ms as published; d = 1-14 ms probe) |
| `te_1ms` .. `te_10ms` | `transfer_entropy.py` | 2 | Schreiber 2000; Ito et al. 2011 (bin width swept) |
| `stg`, `stg_roi15` | `stg.py` | 2 | Spivak et al. 2022 (ROI 0-5 ms as published; 0-15 ms probe) |

Skipped, one line each:

- **CoNNECT** -- needs a convolutional net pre-trained on statistics matched to the target network; no matched model exists for this simulator.
- **Ising / kinetic Ising** -- estimates synchrony, is bin-width unstable, and is not directed in the sense the ground truth requires.
- **Granger causality** -- the review's own STG validation found it inconsistent with known physiology where the GLM was consistent.

## Invariants (see `accumulate.py`)

1. Every `(N, N)` array is `[pre, post]`; `C[l][i, j]` counts bins where i fired l bins before j.
2. Candidate set = all ordered off-diagonal pairs, for every method.
3. Ground truth only in `evaluate.py`; hyperparameters chosen against labels are reported as *oracle*.
4. Positive = excitatory, negative = inhibitory.
5. Every F1 / recall under both denominators (`all_edges`, `exc_only`).

## Running

```bash
python analysis/baselines/test_accumulate.py                         # invariants + reproduction
python analysis/baselines/gate_delay.py --session sweep_c50_seed01 --state normal --n-recordings 50 --methods tier1
python analysis/baselines/run_dataset.py --session sweep_c50_seed01 --state normal --n-recordings 50 --methods all
python analysis/baselines/run_sweep.py --n-recordings 200 --workers 8 --threads 3   # the full sweep, locally
python analysis/baselines/aggregate.py && python analysis/baselines/figures.py && python analysis/baselines/write_results.py
```
