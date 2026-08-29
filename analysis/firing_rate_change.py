"""Firing-rate change across every dataset: paired normal -> seizure.

Panel A: paired slope plot -- one line per network from its normal-state to
its seizure-state mean firing rate (200 recordings each, paired noise
streams), colored by cluster-size group. This is the right display for a
paired design: every dataset is visible individually and the consistency of
the effect is read off the parallelism of the lines.

Panel B: per-network fold-change vs the published 4-AP culture value
(~3.6x, MFR 0.53 -> 1.90 Hz), so each dataset can be compared to the
literature anchor rather than only the group mean.

Panel C: the sAHP severity ladder as per-network trajectories (from
ladder_summary.csv), i.e. the dose-response with every dataset drawn.

    python analysis/firing_rate_change.py
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA, list_sessions  # noqa: E402

OUT = os.path.join(DATA, "sweep_summary")
C50, C40 = "#1f5fd0", "#c0392b"
LIT_FOLD, LIT_FROM, LIT_TO = 3.6, 0.53, 1.90


def session_rate(session, state):
    """Mean per-neuron firing rate over all recordings, and its SEM."""
    meta_p = os.path.join(DATA, session, state, "session_metadata.json")
    if not os.path.exists(meta_p):
        return None, None, None
    with open(meta_p, encoding="utf-8") as fh:
        meta = json.load(fh)
    net = np.load(glob.glob(os.path.join(DATA, session, state, "network_*.npz"))[0],
                  allow_pickle=True)
    n = int(net["neuron_positions"].shape[0])
    dur_s = float(meta["recording_duration"]) / 1000.0
    rates = np.array([r["num_spikes"] / n / dur_s for r in meta["recordings"]
                      if "num_spikes" in r], float)
    if not rates.size:
        return None, None, None
    return float(rates.mean()), float(rates.std(ddof=1) / np.sqrt(rates.size)), n


def main():
    os.makedirs(OUT, exist_ok=True)
    sessions = sorted(s for s in list_sessions() if s.startswith("sweep_"))
    data = []
    for s in sessions:
        nrm, nsem, n = session_rate(s, "normal")
        szr, ssem, _ = session_rate(s, "seizure")
        if nrm is None or szr is None:
            continue
        data.append(dict(session=s, group="c50" if "_c50_" in s else "c40",
                         n_neurons=n, normal=nrm, normal_sem=nsem,
                         seizure=szr, seizure_sem=ssem, fold=szr / nrm))
    print("%d networks with both states" % len(data))

    fig = plt.figure(figsize=(15.5, 5.2))
    axA = fig.add_subplot(1, 3, 1)
    axB = fig.add_subplot(1, 3, 2)
    axC = fig.add_subplot(1, 3, 3)

    # --- A: paired slope plot -------------------------------------------
    for d in data:
        col = C50 if d["group"] == "c50" else C40
        axA.plot([0, 1], [d["normal"], d["seizure"]], "-o", color=col, ms=5,
                 lw=1.3, alpha=0.75)
    for grp, col in (("c50", C50), ("c40", C40)):
        g = [d for d in data if d["group"] == grp]
        axA.plot([0, 1], [np.mean([d["normal"] for d in g]),
                          np.mean([d["seizure"] for d in g])],
                 "-", color=col, lw=3.5, alpha=0.95,
                 label="%s mean (n=%d)" % (grp, len(g)))
    axA.plot([0, 1], [LIT_FROM, LIT_TO], "--", color="#444444", lw=2,
             label="published 4-AP culture\n(0.53 -> 1.90 Hz, 3.6x)")
    axA.set_xticks([0, 1]); axA.set_xticklabels(["normal", "seizure"])
    axA.set_xlim(-0.25, 1.25)
    axA.set_ylabel("mean firing rate (Hz)")
    axA.set_title("Every dataset, paired\n(same network, same noise streams)")
    axA.grid(alpha=0.3, axis="y"); axA.legend(fontsize=7, loc="upper left")

    # --- B: fold change per network --------------------------------------
    order = sorted(data, key=lambda d: d["fold"])
    for i, d in enumerate(order):
        axB.plot(i, d["fold"], "o" if d["group"] == "c50" else "s",
                 color=C50 if d["group"] == "c50" else C40, ms=7)
    axB.axhline(LIT_FOLD, ls="--", color="#444444", lw=2,
                label="published 4-AP culture 3.6x")
    axB.axhline(np.mean([d["fold"] for d in data]), ls=":", color="#008837", lw=2,
                label="our mean %.1fx" % np.mean([d["fold"] for d in data]))
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels([d["session"].replace("sweep_", "").replace("_seed", "-")
                         for d in order], rotation=90, fontsize=6)
    axB.set_ylabel("seizure / normal firing-rate fold change")
    axB.set_title("Fold change per dataset")
    axB.grid(alpha=0.3, axis="y"); axB.legend(fontsize=7)

    # --- C: sAHP severity trajectories per network ------------------------
    csv_p = os.path.join(OUT, "ladder_summary.csv")
    if os.path.exists(csv_p):
        rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
        per = defaultdict(dict)
        for r in rows:
            if r["family"] == "sahp" and r["rate_hz"] not in ("", "nan"):
                per[r["session"]][float(r["x"])] = float(r["rate_hz"])
        for session, pts in sorted(per.items()):
            xs = sorted(pts)
            col = C50 if "_c50_" in session else C40
            axC.plot(xs, [pts[x] for x in xs], "-o", color=col, ms=3.5, lw=1.1,
                     alpha=0.65)
        base = np.mean([pts[0.0] for pts in per.values() if 0.0 in pts])
        axC.axhline(base * LIT_FOLD, ls="--", color="#444444", lw=2,
                    label="3.6x baseline (published 4-AP)")
        axC.set_xlabel("sAHP deficit severity s")
        axC.set_ylabel("mean firing rate (Hz)")
        axC.set_title("Dose-response, every dataset\n(180 s runs, %d networks)" % len(per))
        axC.grid(alpha=0.3); axC.legend(fontsize=7)

    fig.tight_layout()
    p = os.path.join(OUT, "firing_rate_change.png")
    fig.savefig(p, dpi=140, facecolor="white")
    print("figure -> %s" % p)

    with open(os.path.join(OUT, "firing_rate_change.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)
    print("csv    -> %s" % os.path.join(OUT, "firing_rate_change.csv"))
    folds = [d["fold"] for d in data]
    print("fold change: mean %.2fx, range %.2f-%.2f" % (np.mean(folds), min(folds), max(folds)))


if __name__ == "__main__":
    main()
