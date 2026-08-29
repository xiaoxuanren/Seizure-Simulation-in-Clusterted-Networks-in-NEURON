"""Network burst rate and participation change across every dataset.

Companion to firing_rate_change.py, from the burst_stats.json produced by
the statistical detector (full AND partial events). Paired slope plots
(normal -> seizure, same networks and noise streams) for:

    A  network burst rate (events per 60 s recording)
    B  mean burst participation
    C  mean burst duration
    D  the full/partial composition shift

Every dataset is drawn individually; group means overlaid.

    python analysis/burst_change.py
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA, list_sessions  # noqa: E402

OUT = os.path.join(DATA, "sweep_summary")
C50, C40 = "#1f5fd0", "#c0392b"


def load(session, state):
    p = os.path.join(DATA, session, "results", state, "bursts", "burst_stats.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    n_tot = max(1, d["n_bursts_total"])
    return dict(bursts_per_rec=d["bursts_per_rec"],
                participation=d["mean_participation"],
                duration_ms=d["mean_duration_ms"],
                frac_full=d["n_full"] / n_tot,
                n_full_per_rec=d["n_full"] / max(1, d["n_recordings"]),
                n_partial_per_rec=d["n_partial"] / max(1, d["n_recordings"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in sorted(x for x in list_sessions() if x.startswith("sweep_")):
        nrm, szr = load(s, "normal"), load(s, "seizure")
        if not nrm or not szr:
            continue
        r = dict(session=s, group="c50" if "_c50_" in s else "c40")
        for k in nrm:
            r["normal_" + k] = nrm[k]
            r["seizure_" + k] = szr[k]
        rows.append(r)
    print("%d networks with both states" % len(rows))

    panels = [("bursts_per_rec", "network bursts per 60 s recording", "A"),
              ("participation", "mean burst participation", "B"),
              ("duration_ms", "mean burst duration (ms)", "C"),
              ("frac_full", "fraction of events that are FULL (>0.35)", "D")]
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.9))
    for ax, (key, label, tag) in zip(axes, panels):
        for r in rows:
            col = C50 if r["group"] == "c50" else C40
            ax.plot([0, 1], [r["normal_" + key], r["seizure_" + key]], "-o",
                    color=col, ms=5, lw=1.3, alpha=0.75)
        for grp, col in (("c50", C50), ("c40", C40)):
            g = [r for r in rows if r["group"] == grp]
            if g:
                ax.plot([0, 1], [np.mean([r["normal_" + key] for r in g]),
                                 np.mean([r["seizure_" + key] for r in g])],
                        "-", color=col, lw=3.5, alpha=0.95,
                        label="%s mean (n=%d)" % (grp, len(g)))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "seizure"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylabel(label)
        ax.set_title("%s. %s" % (tag, label), fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Burst statistics across all 40 datasets (paired: same network, same noise)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, "burst_change.png")
    fig.savefig(p, dpi=140, facecolor="white"); plt.close(fig)
    print("figure -> %s" % p)

    with open(os.path.join(OUT, "burst_change.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("csv    -> %s" % os.path.join(OUT, "burst_change.csv"))
    for key, label, _ in panels:
        nv = np.array([r["normal_" + key] for r in rows], float)
        sv = np.array([r["seizure_" + key] for r in rows], float)
        print("  %-34s %6.2f -> %6.2f  (x%.2f)" % (label, nv.mean(), sv.mean(),
                                                   sv.mean() / max(nv.mean(), 1e-9)))


if __name__ == "__main__":
    main()
