"""Stage 3 step 3: the four gating checks on the dephased pilot.

    1. population Vm at 0-15 s   no excursion at ~4.9 s; flat from t = 0
    2. burst classification      ZERO bursts in the 4.60-5.34 s band
    3. mean firing rate          close to the flagship's 0.2789 Hz
    4. V_rest                    near -83.3 mV from t = 0, no -84.5 mV dip at ~0.5 s

Every check is pass/fail with the measured number printed. If any fails, do not
scale up the dataset.

Writes ``dephase_validation.json`` and ``dephase_validation.png``.
"""

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from neuron_simulation.analysis import detect_network_bursts  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEPHASED = os.path.join(REPO, "notebooks", "NEURON data parallel", "dephased_ic")
FLAGSHIP = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                        "20260721_163430")
FLAGSHIP_RATE = 0.2789
FLAGSHIP_VREST = -83.31
IC_BAND = (4600.0, 5340.0)
GATE = 0.35


def load(d):
    fs = [p for p in sorted(glob.glob(os.path.join(d, "recording*.npz")))
          if "raster" not in os.path.basename(p)]
    return fs


def main():
    fs = load(DEPHASED)
    if not fs:
        print("no dephased recordings found in %s" % DEPHASED, flush=True)
        return
    print("dephased recordings: %d" % len(fs), flush=True)

    rates, vm_traces, all_bursts, vrest = [], [], [], []
    for i, p in enumerate(fs):
        d = np.load(p, allow_pickle=True)
        st = [np.atleast_1d(np.asarray(t, float)) for t in d["spike_times"]]
        n = len(st)
        dur = float(d["duration"])
        rates.append(sum(len(t) for t in st) / (n * dur / 1000.0))
        b = detect_network_bursts({j: st[j] for j in range(n)}, n, dur,
                                  participation_threshold=GATE, burn_in_ms=0.0)
        for x in b:
            x["recording"] = i
        all_bursts.extend(b)
        if "voltage_traces" in d:
            tr = d["voltage_traces"]
            vt = d["voltage_times"]
            vm_traces.append((vt, tr.mean(axis=0)))
            sub = tr < -60.0
            vrest.append(float(tr[sub].mean()) if sub.any() else np.nan)

    rates = np.array(rates)
    starts = np.array([b["start_ms"] for b in all_bursts]) if all_bursts else np.array([])
    in_band = ((starts >= IC_BAND[0]) & (starts <= IC_BAND[1])).sum() if starts.size else 0

    # ---- check 1: population Vm over 0-15 s ----
    c1 = {}
    if vm_traces:
        vt = vm_traces[0][0]
        stack = np.stack([v for _, v in vm_traces])
        m = vt <= 15000.0
        grand = stack[:, m].mean(axis=0)
        base = np.median(grand)
        win = (vt[m] >= 4000.0) & (vt[m] <= 6000.0)
        exc = float(grand[win].max() - base) if win.any() else np.nan
        early = (vt[m] <= 1000.0)
        exc_early = float(grand[early].max() - base)
        c1 = dict(baseline_mV=float(base), excursion_4_6s_mV=exc,
                  excursion_0_1s_mV=exc_early,
                  passed=bool(exc < 2.0 and exc_early < 3.0))
        print("\n1. population Vm 0-15 s: baseline %.2f mV | excursion in 4-6 s "
              "%+.2f mV | excursion in 0-1 s %+.2f mV  -> %s"
              % (base, exc, exc_early, "PASS" if c1["passed"] else "FAIL"),
              flush=True)

    # ---- check 2: bursts in the IC band ----
    c2 = dict(n_bursts_total=int(len(all_bursts)), n_in_ic_band=int(in_band),
              ic_band_s=[IC_BAND[0] / 1000, IC_BAND[1] / 1000],
              bursts_per_recording=len(all_bursts) / len(fs),
              passed=bool(in_band == 0))
    print("2. bursts at the %.2f gate: %d total (%.2f/rec) | %d in the "
          "%.2f-%.2f s IC band -> %s"
          % (GATE, len(all_bursts), c2["bursts_per_recording"], in_band,
             IC_BAND[0] / 1000, IC_BAND[1] / 1000,
             "PASS" if c2["passed"] else "FAIL"), flush=True)
    if starts.size:
        print("     burst starts (s): %s"
              % ", ".join("%.2f" % (s / 1000) for s in np.sort(starts)[:20]),
              flush=True)

    # ---- check 3: mean rate ----
    dev = float(rates.mean() - FLAGSHIP_RATE)
    c3 = dict(mean_rate_hz=float(rates.mean()), sd=float(rates.std()),
              flagship_rate_hz=FLAGSHIP_RATE, difference_hz=dev,
              difference_pct=100 * dev / FLAGSHIP_RATE,
              passed=bool(abs(dev) / FLAGSHIP_RATE < 0.15))
    print("3. mean rate: %.4f Hz (sd %.4f) vs flagship %.4f -> %+.4f Hz "
          "(%+.1f%%) -> %s" % (rates.mean(), rates.std(), FLAGSHIP_RATE, dev,
                               c3["difference_pct"],
                               "PASS" if c3["passed"] else "FAIL"), flush=True)

    # ---- check 4: V_rest, and no early dip ----
    c4 = {}
    if vm_traces:
        vr = float(np.nanmean(vrest))
        vt = vm_traces[0][0]
        stack = np.stack([v for _, v in vm_traces])
        grand = stack.mean(axis=0)
        early = (vt >= 200.0) & (vt <= 1200.0)
        dip = float(grand[early].min())
        late = vt >= 20000.0
        vlate = float(grand[late].mean())
        c4 = dict(v_rest_subthreshold_mV=vr, flagship_v_rest_mV=FLAGSHIP_VREST,
                  min_in_0p2_1p2s_mV=dip, late_mean_mV=vlate,
                  dip_below_late_mV=float(dip - vlate),
                  passed=bool(abs(vr - FLAGSHIP_VREST) < 1.5 and
                              (dip - vlate) > -1.0))
        print("4. V_rest: %.2f mV (flagship %.2f) | min over 0.2-1.2 s %.2f vs "
              "late mean %.2f (dip %+.2f mV) -> %s"
              % (vr, FLAGSHIP_VREST, dip, vlate, dip - vlate,
                 "PASS" if c4["passed"] else "FAIL"), flush=True)

    checks = {"1_population_vm": c1, "2_no_ic_bursts": c2, "3_mean_rate": c3,
              "4_v_rest": c4}
    all_pass = all(c.get("passed", False) for c in checks.values() if c)
    print("\n==> %s" % ("ALL CHECKS PASS - safe to scale up"
                        if all_pass else "AT LEAST ONE CHECK FAILED - do not scale up"),
          flush=True)

    json.dump(dict(n_recordings=len(fs), all_passed=bool(all_pass),
                   checks=checks,
                   per_recording_rate=[float(r) for r in rates]),
              open(os.path.join(HERE, "dephase_validation.json"), "w"), indent=2)

    if vm_traces:
        fig, ax = plt.subplots(2, 1, figsize=(13, 6.5))
        vt = vm_traces[0][0] / 1000.0
        for i, (_, v) in enumerate(vm_traces):
            ax[0].plot(vt, v, lw=0.7, alpha=0.75, label="rec %d" % i)
        ax[0].axvspan(IC_BAND[0] / 1000, IC_BAND[1] / 1000, color="#c0392b",
                      alpha=0.15, label="flagship IC-burst band")
        ax[0].set_xlim(0, 15)
        ax[0].set_ylabel("population Vm (mV)")
        ax[0].set_title("Dephased IC: population Vm, first 15 s — no ignition at "
                        "~4.9 s")
        ax[0].legend(fontsize=7, ncol=3)
        for i, (_, v) in enumerate(vm_traces):
            ax[1].plot(vt, v, lw=0.5, alpha=0.75)
        ax[1].axhline(FLAGSHIP_VREST, color="k", ls=":", lw=1,
                      label="flagship V_rest %.2f mV" % FLAGSHIP_VREST)
        ax[1].set_xlim(0, vt.max())
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("population Vm (mV)")
        ax[1].set_title("full recording")
        ax[1].legend(fontsize=8)
        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = os.path.join(HERE, "dephase_validation.png")
        fig.savefig(p, dpi=145, facecolor="white")
        print("saved -> dephase_validation.json / .png", flush=True)


if __name__ == "__main__":
    main()
