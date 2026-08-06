"""How gate-dependent is the burst count?

The rasters show many more synchronous stripes than the 0.35 gate accepts. This
runs the SAME detector with ``participation_threshold=0.0`` so every bracketed
candidate window is returned with its participation, then sweeps the acceptance
gate offline. That separates two things the single word "burst" conflates:

  * the bracketing stage (5 ms bins over 5% active, merged <50 ms, padded +/-10 ms)
    -- this decides what counts as an EVENT at all, and
  * the acceptance gate -- which decides which events are called BURSTS.

Writes ``burst_gate_sensitivity.json`` + ``burst_gate_sensitivity.png``.
"""

import glob
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron_simulation.analysis import detect_network_bursts  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
from session_paths import results_dir  # noqa: E402
_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
RESULTS = results_dir(_S, _T, "bursts")
FIGS = results_dir(_S, _T, "figures")
GATES = np.round(np.arange(0.05, 0.91, 0.05), 2)
IC_WIN = (4500.0, 5500.0)
TOTAL_S = 200 * 60.0


def main():
    # recordings live in the DATA dir, not the results dir (they were the same
    # directory before analysis output was split out)
    paths = [p for p in sorted(glob.glob(os.path.join(SESSION, "recording*.npz")))
             if "raster" not in os.path.basename(p)]
    if not paths:
        raise SystemExit("no recordings under %s" % SESSION)
    part, dur, start, rec = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        st = d["spike_times"]
        n = len(st)
        sd = {j: np.atleast_1d(np.asarray(st[j], float)) for j in range(n)}
        ev = detect_network_bursts(sd, n, float(d["duration"]),
                                  participation_threshold=0.0, burn_in_ms=0.0)
        for b in ev:
            part.append(b["participation"])
            dur.append(b["duration_ms"])
            start.append(b["start_ms"])
            rec.append(i)
        if (i + 1) % 50 == 0:
            print("  %d/%d (%.0fs)" % (i + 1, len(paths), time.time() - t0),
                  flush=True)
    part = np.array(part)
    dur = np.array(dur)
    start = np.array(start)
    rec = np.array(rec)
    is_ic = (start >= IC_WIN[0]) & (start <= IC_WIN[1])

    print("\n=== every bracketed candidate event (gate = 0) ===", flush=True)
    print("  %d events over 200 recordings = %.2f per recording = %.4f Hz"
          % (len(part), len(part) / 200.0, len(part) / TOTAL_S), flush=True)
    print("  participation: median %.3f  mean %.3f  [5/25/75/95 %.3f/%.3f/%.3f/%.3f]"
          % (np.median(part), part.mean(),
             *np.percentile(part, [5, 25, 75, 95])), flush=True)
    print("  duration:      median %.0f ms  mean %.0f ms" % (np.median(dur), dur.mean()),
          flush=True)

    rows = []
    print("\n  gate   events   per_rec    rate_Hz    IBI_s   %IC   median_part",
          flush=True)
    for g in GATES:
        m = part > g
        k = int(m.sum())
        pic = 100.0 * is_ic[m].sum() / max(k, 1)
        rows.append(dict(gate=float(g), n_events=k, per_recording=k / 200.0,
                         rate_hz=k / TOTAL_S, ibi_s=(TOTAL_S / k) if k else None,
                         pct_ic=pic,
                         median_participation=float(np.median(part[m])) if k else None))
        print("  %.2f   %6d   %7.2f   %8.5f  %7.1f  %4.0f   %.3f"
              % (g, k, k / 200.0, k / TOTAL_S, (TOTAL_S / k) if k else np.nan,
                 pic, np.median(part[m]) if k else np.nan), flush=True)

    json.dump(dict(gates=[r["gate"] for r in rows], rows=rows,
                   n_candidates=len(part),
                   participation_percentiles={
                       str(q): float(np.percentile(part, q))
                       for q in (5, 25, 50, 75, 95)},
                   note="detector bracketing fixed (5 ms bins, 5% onset, 50 ms "
                        "merge, 10 ms pad); only the acceptance gate varies"),
              open(os.path.join(RESULTS, "burst_gate_sensitivity.json"), "w"),
              indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(part[~is_ic], bins=np.arange(0, 1.01, 0.02), color="#1f5fd0",
               label="spontaneous")
    ax[0].hist(part[is_ic], bins=np.arange(0, 1.01, 0.02), color="#c0392b",
               alpha=0.85, label="IC-locked")
    for g, c, lab in ((0.35, "k", "project gate 0.35"),
                      (0.80, "#e08020", "stored gate 0.80")):
        ax[0].axvline(g, color=c, ls="--", lw=1.4, label=lab)
    ax[0].set_xlabel("participation of bracketed event")
    ax[0].set_ylabel("count (200 recordings)")
    ax[0].set_title("(a) participation is a continuum, not two modes")
    ax[0].legend(fontsize=8)

    g = np.array([r["gate"] for r in rows])
    ax[1].semilogy(g, [r["n_events"] for r in rows], "o-", color="#2e8b57")
    for gg, c in ((0.35, "k"), (0.80, "#e08020")):
        ax[1].axvline(gg, color=c, ls="--", lw=1.2)
    ax[1].set_xlabel("acceptance gate")
    ax[1].set_ylabel("events accepted (log)")
    ax[1].set_title("(b) burst COUNT vs gate — 30x over 0.35→0.80")

    ax[2].semilogy(g, [r["rate_hz"] for r in rows], "o-", color="#2e8b57",
                   label="model")
    ax[2].axhline(0.03, color="#8e44ad", ls=":", lw=1.6,
                  label="dissociated culture ~0.03 Hz (README)")
    ax[2].axhline(1.5, color="#c0392b", ls=":", lw=1.6,
                  label="README claim 1.5 Hz")
    for gg, c in ((0.35, "k"), (0.80, "#e08020")):
        ax[2].axvline(gg, color=c, ls="--", lw=1.2)
    ax[2].set_xlabel("acceptance gate")
    ax[2].set_ylabel("burst rate (Hz, log)")
    ax[2].set_title("(c) burst RATE vs gate, against the cited references")
    ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Burst count/rate is a function of the acceptance gate — the "
                 "bracketing stage is unchanged throughout", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(FIGS, "burst_gate_sensitivity.png")
    fig.savefig(p, dpi=145, facecolor="white", bbox_inches="tight")
    print("\nsaved -> burst_gate_sensitivity.json / .png", flush=True)


if __name__ == "__main__":
    main()
