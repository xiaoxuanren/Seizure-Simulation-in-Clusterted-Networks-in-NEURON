"""Typed operating point (label-free sum4 @ FDR 0.70) for every session and
both states, recomputed from the saved fit npz against ground truth.

Companion to sweep_summary.py: that script records the normal-state typed
numbers inside sweep_summary.csv; this one writes both states side by side
so the seizure operating point survives in the repo as numbers.

Writes sweep_summary/typed_operating_point.csv.

    python analysis/typed_operating_point.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA, results_dir  # noqa: E402


def metrics(npz_path):
    res = np.load(npz_path, allow_pickle=True)
    pred = res["pred_adjacency"].astype(bool)
    n = pred.shape[0]
    off = ~np.eye(n, dtype=bool)
    true = (res["A_exc"].astype(bool) | res["A_inh"].astype(bool)) & off
    np.fill_diagonal(pred, False)
    tp = int((pred & true).sum())
    fp = int((pred & ~true & off).sum())
    fn = int((~pred & true).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return dict(n_neurons=n, n_true=int(true.sum()), tp=tp, fp=fp, fn=fn,
                precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-12))


def main():
    sessions = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(
        os.path.join(DATA, "sweep_*", "results")))
    rows = []
    for session in sessions:
        for state in ("normal", "seizure"):
            npz = os.path.join(results_dir(session, state, "glm"),
                               "glm_connectivity_sum4_5ms.npz")
            if not os.path.exists(npz):
                continue
            m = metrics(npz)
            m.update(session=session, state=state,
                     group="c50" if "_c50_" in session else "c40")
            rows.append(m)
    out = os.path.join(DATA, "sweep_summary", "typed_operating_point.csv")
    cols = ["session", "group", "state", "n_neurons", "n_true", "tp", "fp",
            "fn", "precision", "recall", "f1"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(out)
    for state in ("normal", "seizure"):
        rr = [r for r in rows if r["state"] == state]
        print("%s (n=%d): mean typed P %.3f  R %.3f  F1 %.3f"
              % (state, len(rr), np.mean([r["precision"] for r in rr]),
                 np.mean([r["recall"] for r in rr]),
                 np.mean([r["f1"] for r in rr])))


if __name__ == "__main__":
    main()
