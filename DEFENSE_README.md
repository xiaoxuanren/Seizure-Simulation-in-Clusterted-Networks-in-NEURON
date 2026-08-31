# PhD-defense branch: materials index

This branch carries everything needed to build defense materials (slides,
figures, talking points) without access to the raw simulation data: all
per-dataset figures, the cross-network analysis figures, the summary
statistics, and the thesis section draft. Raw recordings (~1.1 TB with
voltage; ~2.4 GB spike-only) and fitted weight matrices are NOT here —
only their figure and JSON derivatives.

## The study in one paragraph

Twenty clustered spiking networks (NEURON, single-compartment HH + A-current
+ dynamic extracellular K+ + two-timescale sAHP; 629–1010 neurons; ten with
50 spatial clusters, ten with 40, neuron density held constant) were each
recorded 200 × 60 s in two states. The seizure state changes exactly two
parameters, both on the slow AHP: `sahp_ainc_slow` 0.010 → 0.004 µS
(recruitment depth) and `sahp_tau_slow` 6500 → 3000 ms (recovery clock).
Seizure recording *n* shares its exact background-noise realization with
normal recording *n* (paired-trial design). Connectivity was inferred with a
sparse lag-resolved ridge GLM (5 ms bins, lags 1–6, summed 1–4) thresholded
label-free by jitter-null FDR (25 ms, 8 surrogates), and evaluated against
the exact wired graph.

## Headline numbers

| Quantity | Normal | Seizure |
|---|---|---|
| Mean firing rate | 0.294 Hz (0.261–0.360) | 1.687 Hz (1.580–1.773) |
| Network bursts / 60 s | 9.41 | 25.48 |
| Burst participation | 0.51 (0.27–0.91 across nets) | 0.95 |
| Burst duration | 149 ms | 442 ms |

Inference (normal state, all 20 networks, full 200 recordings):
excitatory AUC 0.997 mean (worst 0.981); typed precision at FDR-target 0.70
= 0.751 mean, recall 0.809. Key findings: (1) burstiness — not network
size, density, or cluster count — governs edge precision; (2) excluding
detected burst windows (<2% of bins normal, 6.9% seizure) restores
precision (+0.11 / +0.55); (3) typed precision PEAKS near 50 recordings and
then declines as recall keeps rising — thresholded output is not
monotonically improved by more data; (4) the jitter-null FDR control is
~4× conservative (nominal 0.70 realizes 0.17); (5) the label-free operating
point runs ~0.10 F1 below the ground-truth oracle ceiling. The 4-AP
mechanism ladder shows A-current block alone does nothing at any dose; only
the two-parameter sAHP axis reproduces the culture/MEA 4-AP signature, and
impaired K+ clearance yields depolarization block, not ictal lengthening.

## Where things are

    notebooks/NEURON data parallel/
      sweep_c50_seed01 ... sweep_c50_seed10     50-cluster networks (topology seeds 1-10)
      sweep_c40_seed11 ... sweep_c40_seed20     40-cluster networks (topology seeds 11-20)
        results/{normal,seizure}/
          figures/          ALL per-dataset figures (see catalog below)
          bursts/burst_stats.json               per-recording burst stats
          glm/*.json        edge metrics, FDR calibration, burst-exclusion
                            arms, split-half, scaling (weight matrices excluded)
      sweep_summary/        cross-network everything:
        thesis_section.docx     thesis draft: methods, results, Figs 1-3 + S1-S3
        thesis_fig{1,2,3}.png, thesis_figS1.png  composite thesis figures
        durgrid/            260 JSONs: the CHTC duration-grid (20 nets x 13 durations)
        durgrid_all_*.png   cross-network duration/calibration/oracle figures
        sweep_summary.csv, firing_rate_change.csv, burst_change.csv
        fov_rate_maps/      field-of-view rate + participation maps, both states
        predicted_connectivity/  predicted-vs-true topology renders per network
        ladder_mechanisms.png, ladder_ictal_test.png, ladder_rasters/  4-AP ladders
        scaling_by_group.png, weight_vs_detection.png, ...

## Per-dataset figure catalog (results/<state>/figures/)

- `recording###_raster_shuffled.png` — 60 s rasters, randomized y-index,
  burst-duration bars + participation rates in the title strip (10 per dataset)
- `topology_true.png` — the wired ground-truth graph
- `glm_predicted_topology.png` — predicted vs true, whole network
- `glm_topology_zoom_cluster*.png` / `_group*.png` / `_combined.png` —
  correctness-colored EDGE drawings at legible scale (TP green, FP red,
  FN dotted orange; excitatory filled, inhibitory ringed) with adjacency submatrix
- `glm_distance_recovery.png` — detection vs connection distance
- `durgrid_scaling.png` — AUC/AP/typed P+R vs recording count (10–200)
- `durgrid_calibration.png` — nominal vs realized FDR per duration
- `durgrid_oracle.png` — achieved operating point vs oracle ceiling

## Reproducing / extending

Code lives on this branch too: `neuron_simulation/` (model; states.py defines
the two-parameter knob), `sparse_glm.py` + `glm_connectivity.py` (inference),
`analysis/` (all figure scripts), `chtc/` (HTCondor kits for generation and
for the spike-only duration-grid analysis). The duration grid was produced by
`chtc/analysis_one.py` on CHTC; CHTC and local fits agree to 1e-9.
