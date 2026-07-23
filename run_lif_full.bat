@echo off
REM Wrapper to run the full learned-LIF flagship inference, resumable from checkpoint.
REM Safe to re-run: _run_lif_full.py resumes from --checkpoint-dir if a checkpoint exists.
REM Called directly or by a Windows Scheduled Task so the run survives session exit.
cd /d "D:\HAI Lab\2026\NEURON model\07 July 2026"
"C:\Users\rxxya\AppData\Local\Programs\Python\Python39\python.exe" -u _run_lif_full.py ^
  --session "notebooks/NEURON data parallel/normal/20260721_163430" ^
  --device cpu --num-workers 4 --epochs 30 --K 100 --max-delay 5 --batch-size 256 ^
  --exclude-bursts true --target-precision 0.9 >> lif_full.log 2>> lif_full.err
