"""Scan background-drive parameters for a target normal-state firing rate.

Finds the (noise_rate, noise_weight) operating point that puts the NORMAL
state's per-neuron mean firing rate in a target band (e.g. the 0.7-1 Hz
range of real 2p data) without touching the seizure knob (sahp_ainc_slow)
or any other biophysics. Each grid point rebuilds the flagship topology
(sweep_c50_seed01: num_clusters=50, space_size=15, topology_seed=1) via the
CHTC kit's own ``build_cfg`` and runs one short recording with the same
noise stream (rec_idx 0), so points differ ONLY in the two drive knobs.

Needs the NEURON-capable Python (3.9). Single-point worker mode::

    py39 analysis/rate_operating_point_scan.py --noise-rate 18 --noise-weight 0.007 \
        --duration-ms 30000 --out <dir>

Grid driver mode (spawns py39 workers in parallel, any Python)::

    python analysis/rate_operating_point_scan.py --grid "18,0.007;36,0.007;..." \
        --duration-ms 30000 --workers 6 --out <dir>

Each worker appends one JSON line to ``<out>/scan_results.jsonl``.
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY39 = r"C:\Users\rxxya\AppData\Local\Programs\Python\Python39\python.exe"


def run_point(noise_rate, noise_weight, duration_ms, out_dir,
              config_path=None, overrides=None):
    """Worker: one simulation, prints + appends a JSON result line.

    ``overrides`` are extra build_network kwargs (e.g. sahp_tau_slow) that,
    like the noise knobs, WIN over the config's state defaults.
    """
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "chtc"))
    import numpy as np
    from generate_one import load_sweep, build_cfg

    overrides = dict(overrides or {})
    if config_path is None:
        config_path = os.path.join(REPO, "chtc", "sweep_config.json")
    sweep = load_sweep(config_path)
    sweep["recording_ms"] = float(duration_ms)
    sweep["record_voltage"] = False
    sweep["save_rasters"] = False
    sweep["build_overrides"] = {"noise_weight": float(noise_weight),
                                "noise_rate": float(noise_rate),
                                **overrides}

    tag = f"nr{noise_rate:g}_nw{noise_weight:g}" + "".join(
        f"_{k}{v:g}" for k, v in sorted(overrides.items()))
    cfg, _prov = build_cfg(sweep, "sweep_c50_seed01")
    cfg["save_dir"] = os.path.join(out_dir, tag)
    cfg["timestamp"] = "scan"
    os.makedirs(os.path.join(cfg["save_dir"], "scan"), exist_ok=True)

    from neuron_simulation.parallel_dataset import run_one_recording
    result = run_one_recording(cfg, 0)

    rec = np.load(result["file"], allow_pickle=True)
    st = rec["spike_times"]
    dur_s = duration_ms / 1000.0
    rates = np.array([len(s) for s in st], dtype=float) / dur_s
    row = {
        "state": sweep.get("state", "normal"),
        "noise_rate": float(noise_rate),
        "noise_weight": float(noise_weight),
        **{k: float(v) for k, v in overrides.items()},
        "duration_s": dur_s,
        "n_neurons": len(rates),
        "rate_mean": float(rates.mean()),
        "rate_p5": float(np.percentile(rates, 5)),
        "rate_median": float(np.median(rates)),
        "rate_p95": float(np.percentile(rates, 95)),
        "rate_max": float(rates.max()),
        "n_bursts": int(result["n_bursts"]),
        "bursts_per_min": float(result["n_bursts"]) / (dur_s / 60.0),
        "mean_participation": float(result["mean_participation"]),
    }
    line = json.dumps(row)
    with open(os.path.join(out_dir, "scan_results.jsonl"), "a",
              encoding="utf-8") as fh:
        fh.write(line + "\n")
    print("RESULT " + line, flush=True)


def run_grid(grid, duration_ms, out_dir, workers, config_path=None):
    """Driver: run every grid point via py39 workers.

    Grid chunk syntax: ``nr,nw[,key=val[,key=val...]]`` -- extra pairs become
    per-point build overrides (e.g. ``18,0.007,sahp_tau_slow=5000``).
    """
    points = []
    for chunk in grid.split(";"):
        parts = chunk.split(",")
        nr, nw = float(parts[0]), float(parts[1])
        ovr = dict(p.split("=") for p in parts[2:])
        points.append((nr, nw, ovr))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(points)} points, {workers} concurrent, "
          f"{duration_ms:.0f} ms each")

    pending = list(points)
    running = []  # (proc, point)
    failed = []
    while pending or running:
        while pending and len(running) < workers:
            nr, nw, ovr = pending.pop(0)
            cmd = [PY39, os.path.abspath(__file__),
                   "--noise-rate", str(nr), "--noise-weight", str(nw),
                   "--duration-ms", str(duration_ms), "--out", out_dir]
            if config_path:
                cmd += ["--config", config_path]
            for k, v in ovr.items():
                cmd += ["--override", f"{k}={v}"]
            proc = subprocess.Popen(
                cmd, cwd=REPO, stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
            running.append((proc, (nr, nw)))
            print(f"launched nr={nr:g} nw={nw:g} {ovr}", flush=True)
        still = []
        for proc, pt in running:
            if proc.poll() is None:
                still.append((proc, pt))
            elif proc.returncode != 0:
                failed.append(pt)
                print(f"FAILED nr={pt[0]:g} nw={pt[1]:g} "
                      f"(exit {proc.returncode})", flush=True)
            else:
                print(f"done nr={pt[0]:g} nw={pt[1]:g}", flush=True)
        running = still
        if running:
            import time
            time.sleep(10)

    print(f"\nresults -> {os.path.join(out_dir, 'scan_results.jsonl')}")
    if failed:
        print(f"{len(failed)} FAILED points: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise-rate", type=float)
    ap.add_argument("--noise-weight", type=float)
    ap.add_argument("--grid", help='chunks "nr,nw[,key=val...]" joined by ";"')
    ap.add_argument("--duration-ms", type=float, default=30000.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None,
                    help="sweep config (default chtc/sweep_config.json; use "
                         "chtc/sweep_config_seizure.json for seizure state)")
    ap.add_argument("--override", action="append", default=[],
                    help="extra build_network kwarg key=val (repeatable)")
    a = ap.parse_args()

    if a.grid:
        run_grid(a.grid, a.duration_ms, a.out, a.workers,
                 config_path=a.config)
    else:
        if a.noise_rate is None or a.noise_weight is None:
            ap.error("need --grid, or --noise-rate + --noise-weight")
        overrides = {}
        for item in a.override:
            k, v = item.split("=", 1)
            overrides[k] = float(v)
        run_point(a.noise_rate, a.noise_weight, a.duration_ms, a.out,
                  config_path=a.config, overrides=overrides)


if __name__ == "__main__":
    main()
