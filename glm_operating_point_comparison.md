# GLM operating-point check: sweep-AUC-best vs shipped default

The leakage-free hyperparameter sweep (`glm_sweep_results.json`) ranked configs by
**AUC** and found `bin=7/lag=2` marginally best (0.9648) over the shipped default
`bin=5/lag=6` (0.962). But AUC is a *ranking* metric; the deployable answer is the
**jitter-FDR TP/FP/FN** at the operating point. Running both through the actual
jitter-FDR (peak readout, target FDR 0.10, spike-only flagship session) shows the
AUC edge does **not** translate — it even inverts:

| config | AUC | TP | FP | FN | Precision | Recall | F1 |
|--------|-----|----|----|----|-----------|--------|----|
| bin=5 / lag=6  (shipped default) | 0.962  | 7001 | 313 | 6355 | 0.96 | 0.52 | 0.68 |
| bin=7 / lag=2  (sweep AUC-best)  | 0.9648 | 6889 | 553 | 6467 | 0.93 | 0.52 | 0.66 |

**Conclusion:** the shipped default (`bin=5/lag=6/peak`) is the equal-or-better
operating-point choice despite the sweep-best's hair-higher AUC. Connectivity
performance is a flat plateau (see the sweep), and AUC differences of ~0.003 do not
map to the jitter-FDR confusion — they can invert, because the confusion depends on
the score distribution near the threshold and the null calibration, not the global
edge ranking. No retuning needed.

Raw logs: `glm_bin7lag2.log`, `glm_peak.log`.
