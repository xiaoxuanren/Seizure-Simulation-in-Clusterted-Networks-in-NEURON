"""Does impaired K+ clearance reach the 4-AP slice ICTAL regime?

The sAHP-deficit seizure state reproduces the 4-AP CULTURE/MEA signature
(higher firing rate, more frequent and more synchronous network bursts) but
its events last ~400 ms, whereas the 4-AP SLICE literature reports ictal
discharges of ~31 s (range 31-103 s) recurring on a ~26 s interval, with
interictal discharges of ~1-2 s (Avoli and colleagues; see the reference
constants below). In that literature ictogenesis depends on extracellular K+
accumulation and interneuron depolarization block -- and K+ accumulation is
exactly the route this project holds FIXED (tau_k = 200 ms) while using the
sAHP knob.

This script walks tau_k from the normal 200 ms up to the impaired values used
by states.kclearance_seizure_state, on two arms:

    arm "kclear"  -- tau_k ladder on the NORMAL sAHP (K+ route alone)
    arm "both"    -- tau_k ladder on top of the seizure sAHP state

and measures event duration / rate / participation with the sweep's detector.
Recordings default to 180 s because a ~31 s ictal event cannot be
characterised in a 60 s window.

    py -3.9 analysis/tauk_ictal_ladder.py --point-session sweep_c50_seed01 \
        --arm both --tau-k 6000 --point-out point.json
    py -3.9 analysis/tauk_ictal_ladder.py --combine <dir>
"""
import argparse
import glob as _glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "chtc"))
from session_paths import DATA, resolve, results_dir  # noqa: E402

# 4-AP slice literature reference values (seconds)
LIT_ICTAL_S = (31.0, 103.0)      # ictal discharge duration range
LIT_INTERICTAL_S = (1.1, 2.34)   # interictal discharge duration
LIT_IBI_S = 25.75                # interictal recurrence interval

TAU_K_LADDER = [200.0, 1000.0, 2500.0, 6000.0, 12000.0]
SEIZURE_AINC, SEIZURE_TAU_SLOW = 0.004, 3000.0

_spec = importlib.util.spec_from_file_location(
    "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)


def phenotype(spike_data, n, duration_ms):
    bursts, _ = _an.detect_network_bursts_all(spike_data, n, duration_ms)
    parts = [b["participation"] for b in bursts]
    durs = [b["duration_ms"] for b in bursts]
    peaks = sorted(b["peak_time_ms"] for b in bursts)
    ibis = np.diff(peaks) if len(peaks) > 1 else np.array([])
    n_spikes = sum(len(s) for s in spike_data.values())
    return dict(
        rate_hz=n_spikes / n / (duration_ms / 1000.0),
        n_events=len(bursts),
        events_per_min=len(bursts) / (duration_ms / 60000.0),
        mean_participation=float(np.mean(parts)) if parts else float("nan"),
        mean_duration_ms=float(np.mean(durs)) if durs else float("nan"),
        max_duration_ms=float(np.max(durs)) if durs else float("nan"),
        mean_ibi_ms=float(np.mean(ibis)) if ibis.size else float("nan"))


def run_point(session, arm, tau_k, duration_ms):
    import generate_one as g
    from neuron_simulation.network_builder import build_network
    from neuron_simulation.simulation import run_simulation
    from neuron_simulation.noise import reseed_noise

    sweep = g.load_sweep(os.path.join(REPO, "chtc", "sweep_config.json"))
    cfg, _prov = g.build_cfg(sweep, session)
    build_kwargs = dict(cfg["build_kwargs"])
    build_kwargs["tau_k"] = float(tau_k)
    if arm == "both":
        build_kwargs["sahp_ainc_slow"] = SEIZURE_AINC
        build_kwargs["sahp_tau_slow"] = SEIZURE_TAU_SLOW
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"], **build_kwargs)
    reseed_noise(net.noise, 0)
    spikes, _v, ko = run_simulation(
        net, duration=duration_ms, dt=cfg["dt"],
        discard_transient_ms=cfg["discard_transient_ms"],
        record_voltage=False, record_ko=True)
    ph = phenotype(spikes, net.n_neurons, duration_ms)
    ph.update(session=session, arm=arm, tau_k=float(tau_k),
              duration_ms=duration_ms)
    if ko is not None and "mean_ko" in ko:
        ph["max_ko_mM"] = float(np.max(ko["mean_ko"]))
    print("  %-6s tau_k=%6.0f: %.2f Hz | %4.1f ev/min | part %.2f | "
          "dur mean %5.0f max %6.0f ms | IBI %6.0f ms%s"
          % (arm, tau_k, ph["rate_hz"], ph["events_per_min"],
             ph["mean_participation"], ph["mean_duration_ms"],
             ph["max_duration_ms"], ph["mean_ibi_ms"],
             "" if "max_ko_mM" not in ph else " | [K+]o max %.1f mM" % ph["max_ko_mM"]),
          flush=True)
    return ph


def figure(points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT = os.path.join(DATA, "sweep_summary")
    os.makedirs(OUT, exist_ok=True)
    arms = {"kclear": ("#2b83ba", "tau_k alone (normal sAHP)"),
            "both": ("#d7191c", "tau_k + sAHP seizure state")}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    metrics = [("mean_duration_ms", "mean event duration (ms)"),
               ("max_duration_ms", "longest event (ms)"),
               ("events_per_min", "events per minute"),
               ("mean_participation", "mean participation")]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for arm, (col, lab) in arms.items():
            pts = sorted([p for p in points if p["arm"] == arm], key=lambda p: p["tau_k"])
            if pts:
                ax.plot([p["tau_k"] for p in pts], [p[key] for p in pts],
                        "o-", color=col, ms=6, lw=1.6, label=lab)
        if "duration" in key:
            ax.axhspan(LIT_ICTAL_S[0] * 1000, LIT_ICTAL_S[1] * 1000, color="#c0392b",
                       alpha=0.10, lw=0)
            ax.axhspan(LIT_INTERICTAL_S[0] * 1000, LIT_INTERICTAL_S[1] * 1000,
                       color="#2e8b57", alpha=0.12, lw=0)
            ax.text(0.02, 0.95, "4-AP slice: ictal 31-103 s (red band),\n"
                                "interictal 1.1-2.3 s (green band)",
                    transform=ax.transAxes, fontsize=7, va="top")
            ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("tau_k (ms)  --  K+ clearance time constant")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Impaired K+ clearance: does it reach the 4-AP slice ictal regime?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, "tauk_ictal_ladder.png")
    fig.savefig(out, dpi=140, facecolor="white")
    print("figure -> %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-session")
    ap.add_argument("--arm", choices=["kclear", "both"], default="both")
    ap.add_argument("--tau-k", type=float)
    ap.add_argument("--point-out")
    ap.add_argument("--duration", type=float, default=180000.0)
    ap.add_argument("--combine")
    a = ap.parse_args()

    if a.combine:
        points = []
        for p in sorted(_glob.glob(os.path.join(a.combine, "point_*.json"))):
            with open(p, encoding="utf-8") as fh:
                points.append(json.load(fh))
        out = os.path.join(results_dir(points[0]["session"], "normal", "other"),
                           "tauk_ictal_ladder.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(points, fh, indent=1)
        print("%d points -> %s" % (len(points), out))
        figure(points)
        return

    ph = run_point(a.point_session, a.arm, a.tau_k, a.duration)
    with open(a.point_out, "w", encoding="utf-8") as fh:
        json.dump(ph, fh, indent=1)


if __name__ == "__main__":
    main()
