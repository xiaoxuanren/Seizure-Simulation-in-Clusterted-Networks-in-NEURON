"""Summarize the CHTC mechanism ladders across all networks.

Reads the per-job JSONs produced by chtc/ladder_one.py (one per network x
ladder point) and writes:

    sweep_summary/ladder_summary.csv        one row per (network, point)
    sweep_summary/ladder_mechanisms.png     the three mechanism ladders side
                                            by side, n = 20 networks, with the
                                            4-AP slice literature bands
    sweep_summary/ladder_ictal_test.png     event duration vs tau_k against
                                            the ictal (31-103 s) band

    python analysis/ladder_summary.py --src "D:/path/to/ladder_jsons"
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA  # noqa: E402

OUT = os.path.join(DATA, "sweep_summary")
LIT_ICTAL_S = (31.0, 103.0)
LIT_INTERICTAL_S = (1.1, 2.34)
C50, C40 = "#1f5fd0", "#c0392b"

FAMILIES = {
    "fourap": ("4-AP: A-current block", "dose_mm"),
    "sahp": ("sAHP deficit severity", "severity"),
    "kclear": ("impaired K+ clearance (normal sAHP)", "tau_k"),
    "both": ("impaired K+ clearance + sAHP seizure", "tau_k"),
}


def parse_point(rec):
    """Family and x-value from a point's label/overrides."""
    label = rec["label"]
    fam = label.split("_")[0]
    ov = rec.get("overrides", {})
    if fam == "fourap":
        x = float(label.split("_")[1].replace("mM", ""))
    elif fam == "sahp":
        x = float(label.split("s")[-1])
    else:
        x = float(ov.get("tau_k", float("nan")))
    return fam, x


def load(src):
    rows = []
    for p in sorted(glob.glob(os.path.join(src, "ladder_*.json"))):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        fam, x = parse_point(rec)
        rec.pop("bursts", None)
        rec["family"] = fam
        rec["x"] = x
        rec["group"] = "c50" if "_c50_" in rec["session"] else "c40"
        rows.append(rec)
    return rows


def mean_sem(vals):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if not v.size:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def family_curve(rows, fam, key):
    pts = {}
    for r in rows:
        if r["family"] == fam and np.isfinite(r.get(key, float("nan"))):
            pts.setdefault(r["x"], []).append(r[key])
    xs = sorted(pts)
    means, sems = zip(*[mean_sem(pts[x]) for x in xs]) if xs else ((), ())
    return list(xs), list(means), list(sems), [len(pts[x]) for x in xs]


def figures(rows):
    os.makedirs(OUT, exist_ok=True)
    metrics = [("rate_hz", "firing rate (Hz)"),
               ("events_per_min", "events per minute"),
               ("mean_participation", "mean participation"),
               ("mean_duration_ms", "mean event duration (ms)")]

    # --- three mechanisms side by side -----------------------------------
    fams = ["fourap", "sahp", "both"]
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 13.0))
    for col, fam in enumerate(fams):
        for row, (key, label) in enumerate(metrics):
            ax = axes[row][col]
            xs, ms, se, ns = family_curve(rows, fam, key)
            if xs:
                ax.errorbar(xs, ms, yerr=se, fmt="o-", ms=5, lw=1.6,
                            color="#7b3294" if fam == "fourap" else
                                  ("#008837" if fam == "sahp" else "#d7191c"),
                            capsize=3)
            if fam in ("kclear", "both"):
                ax.set_xscale("log")
            if key == "mean_duration_ms":
                ax.axhspan(LIT_ICTAL_S[0] * 1000, LIT_ICTAL_S[1] * 1000,
                           color="#c0392b", alpha=0.10, lw=0)
                ax.axhspan(LIT_INTERICTAL_S[0] * 1000, LIT_INTERICTAL_S[1] * 1000,
                           color="#2e8b57", alpha=0.12, lw=0)
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel(label)
            if row == 3:
                ax.set_xlabel({"fourap": "[4-AP] (mM), IC50 1 mM",
                               "sahp": "sAHP deficit severity s",
                               "both": "tau_k (ms)"}[fam])
        axes[0][col].set_title("%s\n(mean +- SEM, n = %d networks)"
                               % (FAMILIES[fam][0], max(ns) if ns else 0), fontsize=10)
    fig.suptitle("Three routes to epileptiform activity, all %d networks\n"
                 "duration panels: red band = 4-AP slice ictal (31-103 s), "
                 "green band = interictal (1.1-2.3 s)"
                 % len({r["session"] for r in rows}), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "ladder_mechanisms.png")
    fig.savefig(p, dpi=140, facecolor="white"); plt.close(fig)
    print("figure -> %s" % p)

    # --- the ictal test ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for fam, col, lab in (("kclear", "#2b83ba", "tau_k alone"),
                          ("both", "#d7191c", "tau_k + sAHP seizure")):
        for ax, key in ((ax1, "mean_duration_ms"), (ax2, "max_duration_ms")):
            xs, ms, se, _ = family_curve(rows, fam, key)
            if xs:
                ax.errorbar(xs, ms, yerr=se, fmt="o-", color=col, ms=6, lw=1.6,
                            capsize=3, label=lab)
    for ax, title in ((ax1, "mean event duration"), (ax2, "longest event")):
        ax.axhspan(LIT_ICTAL_S[0] * 1000, LIT_ICTAL_S[1] * 1000, color="#c0392b",
                   alpha=0.10, lw=0)
        ax.axhspan(LIT_INTERICTAL_S[0] * 1000, LIT_INTERICTAL_S[1] * 1000,
                   color="#2e8b57", alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("tau_k (ms) -- K+ clearance"); ax.set_ylabel("ms")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Does impaired K+ clearance reach the 4-AP slice ictal regime?\n"
                 "red band = ictal 31-103 s, green band = interictal 1.1-2.3 s",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(OUT, "ladder_ictal_test.png")
    fig.savefig(p, dpi=140, facecolor="white"); plt.close(fig)
    print("figure -> %s" % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of ladder_*.json files")
    a = ap.parse_args()
    rows = load(a.src)
    if not rows:
        raise SystemExit("no ladder_*.json under %s" % a.src)
    os.makedirs(OUT, exist_ok=True)
    keys = ["session", "group", "family", "label", "x", "topology_seed", "n_neurons",
            "rate_hz", "n_events", "events_per_min", "mean_participation",
            "mean_duration_ms", "max_duration_ms", "mean_ibi_ms", "n_full",
            "max_ko_mM", "mean_ko_mM", "duration_ms"]
    csv_path = os.path.join(OUT, "ladder_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("%s: %d points, %d networks"
          % (csv_path, len(rows), len({r["session"] for r in rows})))
    figures(rows)

    # headline numbers
    for fam in ("fourap", "sahp", "both"):
        xs, ms, _, _ = family_curve(rows, fam, "mean_duration_ms")
        xr, mr, _, _ = family_curve(rows, fam, "rate_hz")
        if xs:
            print("%-8s duration %.0f -> %.0f ms | rate %.2f -> %.2f Hz"
                  % (fam, ms[0], ms[-1], mr[0], mr[-1]))


if __name__ == "__main__":
    main()
