"""Combine the three Part-A arms: side-by-side table, Jaccard overlap, figure.

Reads the per-arm ``burstexcl_<arm>_<tag>.json`` / ``.npz`` written by
burstexcl_glm_arm.py. Writes ``burstexcl_summary.json`` and
``burstexcl_arms.png``. Nothing pre-existing is overwritten.
"""

import itertools
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
ARMS = [("A", "full"), ("B", "drop0-6s"), ("C", "drop30-36s")]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    res, pred = {}, {}
    for arm, tag in ARMS:
        res[arm] = json.load(open(os.path.join(SESSION, "burstexcl_%s_%s.json"
                                               % (arm, tag))))
        pred[arm] = np.load(os.path.join(SESSION, "burstexcl_%s_%s.npz"
                                         % (arm, tag)))["pred"]

    rows = [
        ("spikes kept (frac)", "raw_spikes_frac_kept", "%.4f"),
        ("segments", "n_segments", "%d"),
        ("exc AUC", "exc_auc", "%.4f"),
        ("exc AP", "exc_ap", "%.4f"),
        ("inh AUC", "inh_auc", "%.4f"),
        ("inh AP", "inh_ap", "%.4f"),
        ("threshold", "thr", "%.5f"),
        ("estimated FDR", "estimated_fdr", "%.4f"),
        ("realized FDR", "realized_fdr", "%.4f"),
        ("n_pred", "n_pred", "%d"),
        ("TP", "TP", "%d"),
        ("FP", "FP", "%d"),
        ("precision", "precision", "%.4f"),
        ("recall", "recall", "%.4f"),
        ("F1", "f1", "%.4f"),
    ]
    print("%-20s %12s %12s %12s" % ("", "A (full)", "B (drop0-6)", "C (drop30-36)"),
          flush=True)
    print("-" * 60, flush=True)
    for label, key, fmt in rows:
        print("%-20s %12s %12s %12s"
              % (label, fmt % res["A"][key], fmt % res["B"][key],
                 fmt % res["C"][key]), flush=True)

    print("\n--- deltas vs arm A ---", flush=True)
    for arm in ("B", "C"):
        print("  %s: exc_AP %+.4f  F1 %+.4f  recall %+.4f  n_pred %+d"
              % (arm, res[arm]["exc_ap"] - res["A"]["exc_ap"],
                 res[arm]["f1"] - res["A"]["f1"],
                 res[arm]["recall"] - res["A"]["recall"],
                 res[arm]["n_pred"] - res["A"]["n_pred"]), flush=True)
    print("  B vs C: exc_AP %+.4f  F1 %+.4f  recall %+.4f  n_pred %+d"
          % (res["B"]["exc_ap"] - res["C"]["exc_ap"],
             res["B"]["f1"] - res["C"]["f1"],
             res["B"]["recall"] - res["C"]["recall"],
             res["B"]["n_pred"] - res["C"]["n_pred"]), flush=True)

    print("\n--- Jaccard overlap of predicted edge sets ---", flush=True)
    jac = {}
    for a, b in itertools.combinations([x[0] for x in ARMS], 2):
        inter = int((pred[a] & pred[b]).sum())
        union = int((pred[a] | pred[b]).sum())
        j = inter / max(union, 1)
        jac["%s^%s" % (a, b)] = dict(intersection=inter, union=union, jaccard=j,
                                     only_a=int((pred[a] & ~pred[b]).sum()),
                                     only_b=int((pred[b] & ~pred[a]).sum()))
        print("  %s vs %s: J = %.4f  (shared %d, only-%s %d, only-%s %d)"
              % (a, b, j, inter, a, jac["%s^%s" % (a, b)]["only_a"],
                 b, jac["%s^%s" % (a, b)]["only_b"]), flush=True)

    out = dict(arms={a: res[a] for a, _ in ARMS}, jaccard=jac)
    json.dump(out, open(os.path.join(SESSION, "burstexcl_summary.json"), "w"),
              indent=2)

    # --- figure ---
    labels = ["A\nfull 60 s", "B\ndrop 0-6 s", "C\ndrop 30-36 s"]
    colors = ["#555555", "#c0392b", "#1f4e79"]
    keys = [("exc_ap", "excitatory AP"), ("f1", "F1 (excitatory)"),
            ("recall", "recall"), ("precision", "precision"),
            ("realized_fdr", "realized FDR"), ("n_pred", "predicted edges")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
    for a, (k, title) in zip(ax.ravel(), keys):
        vals = [res[arm][k] for arm, _ in ARMS]
        a.bar(labels, vals, color=colors, width=0.6)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 1.5, hi * 0.004)
        a.set_ylim(lo - pad, hi + pad)          # zoom so tiny differences show
        for i, v in enumerate(vals):
            a.text(i, v, ("%.4f" % v) if k != "n_pred" else ("%d" % v),
                   ha="center", va="bottom", fontsize=9)
        a.set_title(title, fontsize=10)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Part A: does the IC-locked first burst drive GLM edge recovery?\n"
                 "B removes the artifact; C removes matched data elsewhere "
                 "(spikes dropped: B %.2f%%, C %.2f%%)  -- axes zoomed"
                 % (100 * (1 - res["B"]["raw_spikes_frac_kept"]),
                    100 * (1 - res["C"]["raw_spikes_frac_kept"])), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    png = os.path.join(SESSION, "burstexcl_arms.png")
    fig.savefig(png, dpi=150, facecolor="white")
    print("\nsaved -> burstexcl_summary.json / burstexcl_arms.png", flush=True)


if __name__ == "__main__":
    main()
