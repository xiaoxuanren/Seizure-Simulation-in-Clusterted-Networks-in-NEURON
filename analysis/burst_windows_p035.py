"""Part B: recompute burst windows at the project's 0.35 gate, and split the
initialization artifact from genuine spontaneous bursting.

The stored ``burst_windows`` in every recording were computed at
``participation_threshold=0.8``, which finds 26 bursts across 200 recordings --
almost all of them the initialization event. The project's definition is 0.35.

This recomputes at 0.35 with ``burn_in_ms=0.0`` across all 200 recordings and
writes the result to a SEPARATE file. The 0.8 windows inside each recording npz
are left untouched -- the shipped figures used them.

Classification, per recording (never across recordings -- see note below):
    first / IC-locked : start in 4.5-5.5 s (file clock)
    later             : everything else

Note on method: grand-mean participation ACROSS recordings can only detect
phase-locked events. A 50% burst at random times in 12 recordings averages down
to ~4%. So all counting here is per-recording and then aggregated.

Writes ``burstwindows_p035.npz`` + ``burstwindows_p035_summary.json`` +
``burstwindows_p035_starts.png`` into the session directory.
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
THRESH = 0.35
BURN_IN = 0.0
IC_WINDOW = (4500.0, 5500.0)   # file clock; the stored 0.8 bursts sit at 4.73-5.09 s
LATER_FROM = 6000.0


def main():
    paths = [p for p in sorted(glob.glob(os.path.join(RESULTS, "recording*.npz")))
             if "raster" not in os.path.basename(p)]
    print("recordings: %d | threshold %.2f | burn_in %.0f ms"
          % (len(paths), THRESH, BURN_IN), flush=True)

    per_rec, all_bursts = [], []
    t0 = time.time()
    for idx, p in enumerate(paths):
        d = np.load(p, allow_pickle=True)
        st = d["spike_times"]
        n = len(st)
        dur = float(d["duration"])
        spike_data = {i: np.atleast_1d(np.asarray(st[i], float)) for i in range(n)}
        bursts = detect_network_bursts(spike_data, n, dur,
                                       participation_threshold=THRESH,
                                       burn_in_ms=BURN_IN)
        stored = np.asarray(d["burst_windows"])
        n_stored = 0 if stored.size == 0 else len(stored)
        for b in bursts:
            b["recording"] = idx
            b["is_ic"] = bool(IC_WINDOW[0] <= b["start_ms"] <= IC_WINDOW[1])
        all_bursts.extend(bursts)
        per_rec.append(dict(recording=idx, n_bursts_035=len(bursts),
                            n_bursts_080_stored=int(n_stored),
                            n_ic=sum(b["is_ic"] for b in bursts),
                            n_later=sum(not b["is_ic"] for b in bursts)))
        if (idx + 1) % 25 == 0:
            print("  %3d/%d  (%.0fs)" % (idx + 1, len(paths), time.time() - t0),
                  flush=True)

    nb = np.array([r["n_bursts_035"] for r in per_rec])
    n_stored_tot = sum(r["n_bursts_080_stored"] for r in per_rec)
    ic = [b for b in all_bursts if b["is_ic"]]
    later = [b for b in all_bursts if not b["is_ic"]]

    def stats(bs, label):
        if not bs:
            print("  %-16s n=0" % label, flush=True)
            return dict(n=0)
        part = np.array([b["participation"] for b in bs])
        dur = np.array([b["duration_ms"] for b in bs])
        start = np.array([b["start_ms"] for b in bs]) / 1000.0
        print("  %-16s n=%4d | participation %.3f +/- %.3f | duration %.1f +/- %.1f ms"
              " | start %.2f-%.2f s"
              % (label, len(bs), part.mean(), part.std(), dur.mean(), dur.std(),
                 start.min(), start.max()), flush=True)
        return dict(n=len(bs), participation_mean=float(part.mean()),
                    participation_sd=float(part.std()),
                    duration_mean_ms=float(dur.mean()),
                    duration_sd_ms=float(dur.std()),
                    start_min_s=float(start.min()), start_max_s=float(start.max()))

    print("\n--- burst counts ---", flush=True)
    print("  total bursts @0.35: %d over %d recordings (%.2f per recording)"
          % (len(all_bursts), len(paths), nb.mean()), flush=True)
    print("  stored bursts @0.80: %d (%.2f per recording)"
          % (n_stored_tot, n_stored_tot / len(paths)), flush=True)
    print("  recordings with >=1 burst @0.35: %d / %d"
          % (int((nb > 0).sum()), len(paths)), flush=True)
    print("  recordings with an IC burst:     %d / %d"
          % (sum(1 for r in per_rec if r["n_ic"] > 0), len(paths)), flush=True)
    print("\n--- classes ---", flush=True)
    s_ic = stats(ic, "first/IC-locked")
    s_later = stats(later, "later")

    # Are later starts uniform across [LATER_FROM, duration]? Uniform => not
    # phase-locked => genuinely spontaneous.
    ks = None
    if later:
        from scipy.stats import kstest, uniform
        starts = np.array([b["start_ms"] for b in later])
        keep = starts >= LATER_FROM
        starts = starts[keep]
        lo, hi = LATER_FROM, 60000.0
        res = kstest(starts, uniform(loc=lo, scale=hi - lo).cdf)
        ks = dict(n=int(starts.size), statistic=float(res.statistic),
                  pvalue=float(res.pvalue), window_s=[lo / 1000, hi / 1000])
        print("\n--- uniformity of LATER burst starts on [%.0f, %.0f] s ---"
              % (lo / 1000, hi / 1000), flush=True)
        print("  KS D = %.4f  p = %.4g  (n = %d)"
              % (res.statistic, res.pvalue, starts.size), flush=True)
        print("  %s" % ("uniform: consistent with spontaneous, not phase-locked"
                        if res.pvalue > 0.05 else
                        "NOT uniform: some temporal structure remains"), flush=True)

    out = dict(threshold=THRESH, burn_in_ms=BURN_IN, n_recordings=len(paths),
               ic_window_ms=list(IC_WINDOW), later_from_ms=LATER_FROM,
               total_bursts_035=len(all_bursts),
               bursts_per_recording_035=float(nb.mean()),
               total_bursts_080_stored=int(n_stored_tot),
               recordings_with_burst_035=int((nb > 0).sum()),
               recordings_with_ic_burst=int(sum(1 for r in per_rec if r["n_ic"] > 0)),
               ic=s_ic, later=s_later, ks_uniform_later=ks, per_recording=per_rec)
    json.dump(out, open(os.path.join(RESULTS, "burstwindows_p035_summary.json"), "w"),
              indent=2)
    np.savez_compressed(
        os.path.join(RESULTS, "burstwindows_p035.npz"),
        start_ms=np.array([b["start_ms"] for b in all_bursts]),
        end_ms=np.array([b["end_ms"] for b in all_bursts]),
        duration_ms=np.array([b["duration_ms"] for b in all_bursts]),
        peak_time_ms=np.array([b["peak_time_ms"] for b in all_bursts]),
        participation=np.array([b["participation"] for b in all_bursts]),
        recording=np.array([b["recording"] for b in all_bursts]),
        is_ic=np.array([b["is_ic"] for b in all_bursts]),
        threshold=THRESH, burn_in_ms=BURN_IN)

    # Figure: cross-recording start-time histogram, IC vs later.
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    edges = np.arange(0, 61, 1.0)
    s_all = np.array([b["start_ms"] for b in all_bursts]) / 1000.0
    m_ic = np.array([b["is_ic"] for b in all_bursts])
    ax[0].hist(s_all[m_ic], bins=edges, color="#c0392b", label="first / IC-locked")
    ax[0].hist(s_all[~m_ic], bins=edges, color="#1f4e79", alpha=0.85, label="later")
    ax[0].set_xlabel("burst start (s, file clock)")
    ax[0].set_ylabel("count across 200 recordings")
    ax[0].set_title("Burst starts @ participation > %.2f" % THRESH)
    ax[0].legend(fontsize=9)
    later_s = s_all[~m_ic]
    later_s = later_s[later_s >= LATER_FROM / 1000.0]
    if later_s.size:
        ax[1].hist(later_s, bins=np.arange(6, 61, 2.0), color="#1f4e79",
                   density=True, label="later starts")
        ax[1].axhline(1.0 / (60 - 6), color="k", ls="--", lw=1,
                      label="uniform density")
        if ks:
            ax[1].set_title("Later starts vs uniform  (KS D=%.3f, p=%.3g)"
                            % (ks["statistic"], ks["pvalue"]))
        ax[1].set_xlabel("burst start (s)")
        ax[1].set_ylabel("density")
        ax[1].legend(fontsize=9)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Part B: burst windows recomputed at the 0.35 gate "
                 "(stored 0.8 windows untouched)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = os.path.join(FIGS, "burstwindows_p035_starts.png")
    fig.savefig(png, dpi=150, facecolor="white")
    print("\nsaved -> burstwindows_p035.npz / _summary.json / _starts.png",
          flush=True)
    print("TOTAL %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
