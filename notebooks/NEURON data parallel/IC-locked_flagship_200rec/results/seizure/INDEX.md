# Results — IC-locked_flagship_200rec / seizure

Analysis output for this dataset. Raw recordings, rasters and per-recording
summaries stay in `../../seizure/`.

Regenerate anything here by pointing the producing script at this dataset:

```bash
DATASET_SESSION=IC-locked_flagship_200rec DATASET_STATE=seizure python analysis/<script>.py
```

| folder | what |
|---|---|
| `glm/` | connectivity fits, E/I typing tables, scaling and FDR-calibration metrics |
| `bursts/` | burst windows at the 0.35 gate, acceptance-gate sensitivity |
| `ic_artifact/` | three-arm test of whether the initialization burst drives edge recovery |
| `figures/` | every figure |
| `other/` | earlier work: learned-LIF, p3 sweeps, transmission, Vm analyses |

## Files

| file | description |
|---|---|
| `figures/fdrdur10to200_calibration.png` | FDR target x duration grid, 10-200 recordings |
| `figures/fdrdur10to200_performance.png` | FDR target x duration grid, 10-200 recordings |
| `figures/glm_labelfree_scaling_vs_duration.png` | sum4 @FDR0.70 edge recovery vs recording duration |
| `figures/glm_scaling_vs_duration.png` | oracle upper bounds (best-F1, @10% FDR) vs duration |
| `glm/a1_typing_fix_results.json` | E/I typing fix: four result tables (JSON) |
| `glm/a1_typing_fix_results_n100.json` | E/I typing fix: four result tables (JSON) |
| `glm/a1_typing_fix_tables.csv` | E/I typing fix: same tables as CSV |
| `glm/a1_typing_fix_tables_n100.csv` | E/I typing fix: same tables as CSV |
| `glm/fdrdur10to200_grid.csv` | FDR target x duration grid, 10-200 recordings |
| `glm/fdrdur10to200_metrics.json` | FDR target x duration grid, 10-200 recordings |
| `glm/fdrdur10to200_operating_points.csv` | FDR target x duration grid, 10-200 recordings |
| `glm/fdrdur10to200_parts` | FDR target x duration grid, 10-200 recordings |
| `glm/glm_labelfree_scaling_metrics.json` | sum4 @FDR0.70 edge recovery vs recording duration |
| `glm/glm_scaling_metrics.json` | oracle upper bounds (best-F1, @10% FDR) vs duration |
