"""LABEL-FREE scaling: edge recovery vs recording duration, all three layers.

Runs the ACTUAL label-free pipeline at each data size -- fit sum4, build the
spike-jitter null, pick the threshold at ``TARGET_FDR`` WITHOUT ground truth --
then evaluates against truth. Three curves per panel:

    excitatory   +W over every off-diagonal pair, scored against A_exc
    inhibitory   -W over pairs whose PREsynaptic neuron is typed inhibitory,
                 scored against A_inh. Typing uses the shipped rank rule
                 (infer_inhibitory, typing="rank", fraction=0.25) -- the A1 fix.
                 The old sign rule typed 1 of 926 neurons at n=200, which made
                 this layer essentially empty.
    all edges    edges_exc | edges_inh, scored against A_exc | A_inh

The excitatory layer does not depend on the typing rule, so those curves match
the earlier excitatory-only version of this script.

Writes ``glm_labelfree_scaling_metrics.json`` and
``glm_labelfree_scaling_vs_duration.png``.
"""

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg  # noqa: E402
from glm_connectivity import (  # noqa: E402
    infer_inhibitory, typing_score)
from session_paths import resolve, results_dir  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SESSION = resolve(_S, _T)
RESULTS = results_dir(_S, _T, "glm")
FIGS = results_dir(_S, _T, "figures")

SIZES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 140, 200]
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
TARGET_FDR, JITTER_MS, N_SURR, SEED = 0.70, 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))
TYPING_LAGS, TYPING_FRACTION = 2, 0.25
REC_MIN = 1.0

LAYERS = [("excitatory", "#2e8b57"), ("inhibitory", "#c0392b"),
          ("all edges", "#1f5fd0")]
KEY = {"excitatory": "excitatory", "inhibitory": "inhibitory",
       "all edges": "all_edges"}


def fdr_threshold(obs, null, target):
    cand = np.unique(obs)
    obs_s = np.sort(obs)
    null_s = np.sort(null.ravel())
    n_surr = null.shape[0]
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    return float(cand[ok[0]]) if len(ok) else float("inf")


def confusion(pred, truth):
    TP = int((pred & truth).sum())
    FP = int((pred & ~truth).sum())
    FN = int((truth & ~pred).sum())
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    return dict(n_pred=TP + FP, TP=TP, FP=FP, FN=FN, precision=P, recall=R,
                f1=2 * P * R / max(P + R, 1e-12),
                realized_fdr=FP / max(TP + FP, 1))


def main():
    t0 = time.time()
    print("loading %s ..." % SESSION, flush=True)
    M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
    gt = sg.load_ground_truth(SESSION)
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    A_exc, A_inh = gt["A_exc"] & off, gt["A_inh"] & off
    A_all = A_exc | A_inh
    print("N=%d | true exc %d, inh %d, all %d | %.1fs"
          % (N, A_exc.sum(), A_inh.sum(), A_all.sum(), time.time() - t0),
          flush=True)

    rows = []
    for n_rec in SIZES:
        if n_rec > len(bnd) - 1:
            continue
        tf = time.time()
        Mn = M[:, :bnd[n_rec]].tocsr()
        bndn = bnd[:n_rec + 1]

        B = sg.fit_B(Mn, bndn, max_lag=MAX_LAG, l2=L2)
        W = B[:KSUM].sum(0)
        np.fill_diagonal(W, 0.0)
        tscore = typing_score(B, k=TYPING_LAGS)

        rng = np.random.default_rng(SEED)
        null_W = []
        for _ in range(N_SURR):
            Bs = sg.fit_B(sg.jitter(Mn, bndn, JBINS, rng), bndn,
                          max_lag=MAX_LAG, l2=L2)
            Ws = Bs[:KSUM].sum(0)
            np.fill_diagonal(Ws, 0.0)
            null_W.append(Ws)

        # excitatory: every off-diagonal pair
        thr_e = fdr_threshold(W[off], np.stack([w[off] for w in null_W]),
                              TARGET_FDR)
        edges_exc = np.zeros((N, N), bool)
        edges_exc[off] = W[off] >= thr_e

        # inhibitory: only pairs whose presynaptic neuron is typed inhibitory
        mask = infer_inhibitory(W, score=tscore, typing="rank",
                                fraction=TYPING_FRACTION)
        pre_inh = np.zeros((N, N), bool)
        pre_inh[np.where(mask)[0], :] = True
        cand_i = off & pre_inh
        edges_inh = np.zeros((N, N), bool)
        thr_i = float("inf")
        if cand_i.any():
            thr_i = fdr_threshold((-W)[cand_i],
                                  np.stack([(-w)[cand_i] for w in null_W]),
                                  TARGET_FDR)
            edges_inh[cand_i] = (-W)[cand_i] >= thr_i

        row = dict(n_rec=n_rec, minutes=n_rec * REC_MIN, thr_exc=thr_e,
                   thr_inh=thr_i, n_typed_inhibitory=int(mask.sum()),
                   excitatory=confusion(edges_exc, A_exc),
                   inhibitory=confusion(edges_inh, A_inh),
                   all_edges=confusion(edges_exc | edges_inh, A_all))
        rows.append(row)
        print("  n=%3d | exc F1 %.3f (P %.3f R %.3f) | inh F1 %.3f (P %.3f "
              "R %.3f) | all F1 %.3f | %.0fs"
              % (n_rec, row["excitatory"]["f1"], row["excitatory"]["precision"],
                 row["excitatory"]["recall"], row["inhibitory"]["f1"],
                 row["inhibitory"]["precision"], row["inhibitory"]["recall"],
                 row["all_edges"]["f1"], time.time() - tf), flush=True)

    json.dump(dict(readout="sum4", target_fdr=TARGET_FDR, jitter_ms=JITTER_MS,
                   n_surrogates=N_SURR, seed=SEED, typing="rank",
                   typing_fraction=TYPING_FRACTION,
                   n_true_exc=int(A_exc.sum()), n_true_inh=int(A_inh.sum()),
                   n_true_all=int(A_all.sum()), rows=rows),
              open(os.path.join(RESULTS, "glm_labelfree_scaling_metrics.json"),
                   "w"), indent=2)

    # ---------------- figure ----------------
    x = np.array([r["minutes"] for r in rows], float)
    truth_n = {"excitatory": int(A_exc.sum()), "inhibitory": int(A_inh.sum()),
               "all edges": int(A_all.sum())}

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("TP", "true positives"), ("precision", "precision"),
              ("recall", "recall"), ("f1", "F1"),
              ("realized_fdr", "realized FDR"), ("n_pred", "predicted edges")]
    for a, (k, title) in zip(ax.ravel(), panels):
        for lname, colr in LAYERS:
            a.plot(x, [r[KEY[lname]][k] for r in rows], "-o", ms=4, lw=1.5,
                   color=colr, label=lname)
        a.set_xlabel("recording duration (min)  [= # of 60 s recordings]")
        a.set_ylabel(title)
        a.set_title("%s vs duration" % title)
        a.grid(alpha=0.25)
        a.legend(fontsize=8)
    for lname, colr in LAYERS:
        ax.ravel()[0].axhline(truth_n[lname], color=colr, ls=":", lw=1, alpha=0.6)
    ax.ravel()[4].axhline(0.10, color="gray", ls=":", lw=1)

    fig.suptitle("GLM sum4 @FDR %.2f (LABEL-FREE) - edge recovery vs recording "
                 "duration, by layer\n%s / %s   |   threshold from the "
                 "spike-jitter null (no ground truth); inhibitory layer uses the "
                 "rank typing rule (fraction=%.2f)"
                 % (TARGET_FDR, _S, _T, TYPING_FRACTION),
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGS, "glm_labelfree_scaling_vs_duration.png")
    fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
    print("\nfigure -> %s" % out, flush=True)
    print("TOTAL %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
