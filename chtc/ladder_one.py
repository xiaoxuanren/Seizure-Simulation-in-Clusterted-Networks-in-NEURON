"""Run ONE point of a parameter ladder on ONE network (a single CHTC job).

Generalizes the sweep worker to arbitrary build_kwargs overrides: a ladder
config lists named points, each with its own overrides and duration, and one
job = one (session, point) pair. Outputs are phenotype JSONs (kilobytes), so
they return through normal HTCondor file transfer with no staging or tar.

    python chtc/ladder_one.py --ladder chtc/ladder_tauk.json \
        --session sweep_c50_seed01 --point-idx 3 --out results

Ladder config schema:
    {
      "name": "tauk_ictal",
      "base_sweep": "chtc/sweep_config.json",   # networks + operating point
      "duration_ms": 180000,
      "record_ko": true,
      "points": [
        {"label": "kclear_tau200", "overrides": {"tau_k": 200.0}},
        {"label": "both_tau6000",
         "overrides": {"tau_k": 6000.0, "sahp_ainc_slow": 0.004,
                       "sahp_tau_slow": 3000.0}}
      ]
    }
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
_an = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_an)


def phenotype(spike_data, n, duration_ms):
    bursts, meta = _an.detect_network_bursts_all(spike_data, n, duration_ms)
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
        mean_ibi_ms=float(np.mean(ibis)) if ibis.size else float("nan"),
        n_full=int(sum(1 for b in bursts if b["burst_class"] == "full")),
        bursts=bursts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--point-idx", type=int, required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--list-points", action="store_true")
    a = ap.parse_args()

    with open(a.ladder, encoding="utf-8") as fh:
        lad = json.load(fh)
    if a.list_points:
        for i, p in enumerate(lad["points"]):
            print("%d\t%s" % (i, p["label"]))
        return
    point = lad["points"][a.point_idx]

    import generate_one as g
    sweep = g.load_sweep(os.path.join(REPO, lad.get("base_sweep",
                                                    "chtc/sweep_config.json")))
    cfg, prov = g.build_cfg(sweep, a.session)
    build_kwargs = dict(cfg["build_kwargs"])
    build_kwargs.update(point["overrides"])

    from neuron_simulation.network_builder import build_network
    from neuron_simulation.simulation import run_simulation
    from neuron_simulation.noise import reseed_noise

    duration = float(point.get("duration_ms", lad.get("duration_ms", 60000.0)))
    net = build_network(cfg["topology"], noise_seed=cfg["noise_seed_base"], **build_kwargs)
    reseed_noise(net.noise, int(point.get("rec_idx", 0)))
    print("[%s | %s] N=%d, overrides=%s, %.0f s"
          % (a.session, point["label"], net.n_neurons, point["overrides"],
             duration / 1000.0), flush=True)
    spikes, _v, ko = run_simulation(
        net, duration=duration, dt=cfg["dt"],
        discard_transient_ms=cfg["discard_transient_ms"],
        record_voltage=False, record_ko=bool(lad.get("record_ko", False)))

    ph = phenotype(spikes, net.n_neurons, duration)
    ph.update(session=a.session, label=point["label"], overrides=point["overrides"],
              duration_ms=duration, ladder=lad.get("name", "ladder"),
              topology_seed=prov["topology_seed"], n_neurons=int(net.n_neurons))
    if ko is not None and "mean_ko" in ko:
        ph["max_ko_mM"] = float(np.max(ko["mean_ko"]))
        ph["mean_ko_mM"] = float(np.mean(ko["mean_ko"]))
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "%s__%s.json" % (a.session, point["label"]))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ph, fh, indent=1)
    print("[%s | %s] %.2f Hz | %.1f ev/min | part %.2f | dur mean %.0f max %.0f ms -> %s"
          % (a.session, point["label"], ph["rate_hz"], ph["events_per_min"],
             ph["mean_participation"], ph["mean_duration_ms"],
             ph["max_duration_ms"], out), flush=True)


if __name__ == "__main__":
    main()
