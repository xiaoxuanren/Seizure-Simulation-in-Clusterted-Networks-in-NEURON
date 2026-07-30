"""Validate a generated dataset against itself.

No external reference values. Every check compares the dataset to its own later
behaviour, so it stays meaningful whatever parameters were used:

    1. no startup burst    zero detected bursts in the opening window -- the
                           signature of every cell starting in the same state
    2. rate is stationary  firing rate early vs late; a settling transient shows
                           up as an early-vs-late gap
    3. V_rest flat from 0  opening Vm vs the recording's own late mean; a
                           shared initial condition shows as a dip or overshoot
    4. recordings differ   distinct spike counts, confirming per-recording noise
                           reseeding actually took effect

    python dataset_validate.py --config <session>/_session_config.pkl --state normal
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from neuron_simulation.analysis import detect_network_bursts  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: opening window treated as "startup" for checks 1-3 (s)
EARLY_S = 5.0
#: tolerances
RATE_TOL = 0.15      # early vs late mean rate
VM_TOL_MV = 1.0      # early Vm deviation from the late mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", required=True)
    a = ap.parse_args()

    with open(a.config, "rb") as fh:
        s = pickle.load(fh)
    cfg, session_dir = s["config"], s["session_dir"]
    d = os.path.join(session_dir, a.state)
    fs = [p for p in sorted(glob.glob(os.path.join(d, "recording*.npz")))
          if "raster" not in os.path.basename(p)]
    if not fs:
        print("no recordings in %s" % d)
        return
    gate = float(cfg["sim"]["participation_threshold"])
    print("state '%s': %d recordings | burst gate %.2f | opening window %.1f s"
          % (a.state, len(fs), gate, EARLY_S), flush=True)

    rates, counts, vm, early_b, all_b = [], [], [], 0, 0
    r_early, r_late = [], []
    for p in fs:
        z = np.load(p, allow_pickle=True)
        st = [np.atleast_1d(np.asarray(t, float)) for t in z["spike_times"]]
        n, dur = len(st), float(z["duration"])
        tot = sum(len(t) for t in st)
        counts.append(tot)
        rates.append(tot / (n * dur / 1000.0))

        allt = np.concatenate([t for t in st if t.size]) if tot else np.array([])
        r_early.append(((allt < EARLY_S * 1000).sum()) / (n * EARLY_S))
        r_late.append(((allt >= EARLY_S * 1000).sum())
                      / (n * (dur / 1000.0 - EARLY_S)))

        b = detect_network_bursts({j: st[j] for j in range(n)}, n, dur,
                                  participation_threshold=gate, burn_in_ms=0.0)
        all_b += len(b)
        early_b += sum(1 for x in b if x["start_ms"] < EARLY_S * 1000)

        if "voltage_traces" in z:
            vm.append((np.asarray(z["voltage_times"], float),
                       np.asarray(z["voltage_traces"], float).mean(axis=0)))

    rates = np.array(rates)
    r_early, r_late = np.array(r_early), np.array(r_late)
    checks = {}

    # 1 - no startup burst
    checks["1_no_startup_burst"] = dict(
        bursts_total=int(all_b), bursts_in_opening=int(early_b),
        opening_window_s=EARLY_S,
        bursts_per_recording=all_b / len(fs), passed=bool(early_b == 0))
    print("\n1. startup burst : %d burst(s) in the first %.1f s (of %d total, "
          "%.2f/rec) -> %s"
          % (early_b, EARLY_S, all_b, all_b / len(fs),
             "PASS" if early_b == 0 else "FAIL"), flush=True)

    # 2 - rate stationarity
    dev = float(np.mean(r_early) / max(np.mean(r_late), 1e-12) - 1.0)
    checks["2_rate_stationary"] = dict(
        rate_hz=float(rates.mean()), sd=float(rates.std()),
        early_hz=float(r_early.mean()), late_hz=float(r_late.mean()),
        early_vs_late=dev, tol=RATE_TOL, passed=bool(abs(dev) < RATE_TOL))
    print("2. rate          : %.4f Hz overall | first %.0f s %.4f vs rest %.4f "
          "(%+.1f%%) -> %s"
          % (rates.mean(), EARLY_S, r_early.mean(), r_late.mean(), 100 * dev,
             "PASS" if abs(dev) < RATE_TOL else "FAIL"), flush=True)

    # 3 - V_rest flat from t=0
    if vm:
        t = vm[0][0]
        g = np.stack([v for _, v in vm]).mean(axis=0)
        late = g[t >= EARLY_S * 1000].mean()
        seg = g[t <= EARLY_S * 1000]
        dmax = float(np.max(np.abs(seg - late)))
        checks["3_vrest_flat"] = dict(
            late_mean_mV=float(late), max_dev_in_opening_mV=dmax,
            tol_mV=VM_TOL_MV, passed=bool(dmax < VM_TOL_MV))
        print("3. V_rest        : late mean %.2f mV | max deviation in the first "
              "%.1f s %.2f mV -> %s"
              % (late, EARLY_S, dmax, "PASS" if dmax < VM_TOL_MV else "FAIL"),
              flush=True)
    else:
        print("3. V_rest        : skipped (voltage not recorded)", flush=True)

    # 4 - recordings actually differ
    uniq = len(set(counts))
    checks["4_recordings_differ"] = dict(
        n_recordings=len(fs), distinct_spike_counts=uniq,
        passed=bool(uniq == len(fs) or len(fs) == 1))
    print("4. independence  : %d distinct spike counts across %d recordings -> %s"
          % (uniq, len(fs), "PASS" if checks["4_recordings_differ"]["passed"]
             else "FAIL"), flush=True)

    all_pass = all(c["passed"] for c in checks.values())
    print("\n==> %s" % ("ALL CHECKS PASS" if all_pass
                        else "AT LEAST ONE CHECK FAILED"), flush=True)

    json.dump(dict(state=a.state, n_recordings=len(fs),
                   all_passed=bool(all_pass), checks=checks,
                   per_recording_rate=[float(r) for r in rates]),
              open(os.path.join(session_dir, "_validation_%s.json" % a.state), "w"),
              indent=2)

    if vm:
        fig, ax = plt.subplots(2, 1, figsize=(13, 6))
        for tt, v in vm:
            ax[0].plot(tt / 1000.0, v, lw=0.7, alpha=0.8)
        ax[0].axvspan(0, EARLY_S, color="#c0392b", alpha=0.12,
                      label="opening window")
        ax[0].set_xlim(0, min(20.0, vm[0][0].max() / 1000.0))
        ax[0].set_ylabel("population Vm (mV)")
        ax[0].set_title("%s - opening seconds (flat = no shared initial condition)"
                        % a.state)
        ax[0].legend(fontsize=8)
        for tt, v in vm:
            ax[1].plot(tt / 1000.0, v, lw=0.5, alpha=0.8)
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("population Vm (mV)")
        ax[1].set_title("full recording")
        for x in ax:
            x.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(session_dir, "_validation_%s.png" % a.state),
                    dpi=140, facecolor="white")
    print("saved -> _validation_%s.json / .png" % a.state, flush=True)


if __name__ == "__main__":
    main()
