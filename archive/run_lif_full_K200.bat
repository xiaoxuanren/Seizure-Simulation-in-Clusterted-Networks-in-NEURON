@echo off
REM K=200 flagship run with conn_AUC checkpoint selection.
REM Writes SEPARATE outputs (output-tag K200, own JSON, own checkpoint dir) so it
REM does NOT touch the K=100 results. Resumable from lif_checkpoints_K200.
cd /d "D:\HAI Lab\2026\NEURON model\07 July 2026"
"C:\Users\rxxya\AppData\Local\Programs\Python\Python39\python.exe" -u _run_lif_full.py ^
  --session "notebooks/NEURON data parallel/normal/20260721_163430" ^
  --device cpu --num-workers 4 --epochs 15 --K 200 --max-delay 5 --batch-size 256 ^
  --exclude-bursts true --target-precision 0.9 --select conn_auc --output-tag K200 ^
  --checkpoint-dir "notebooks/NEURON data parallel/normal/20260721_163430/lif_checkpoints_K200" ^
  --out "notebooks/NEURON data parallel/normal/20260721_163430/learned_lif_full_results_K200.json" ^
  >> lif_full_K200.log 2>> lif_full_K200.err
