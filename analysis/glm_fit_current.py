"""Fit the current label-free connectivity result and save it for downstream use.

Produces ``glm_connectivity_sum4_5ms.npz`` in the schema the older dense
``infer_connectivity`` wrote, but computed with the SPARSE path (so it runs at
n=200) and with the REVISED E/I typing (the A1 fix). The previously stored file
predated that fix: it typed 4 of 926 neurons inhibitory and predicted 8
inhibitory edges, making ``pred_adjacency`` effectively excitatory-only.

Keys: W, candidates, inferred_inhibitory, edges_exc, edges_inh, pred_adjacency,
A_exc, A_inh -- so glm_distance_recovery.py and friends read it unchanged.

    python glm_fit_current.py [--n-recordings 200]
"""

import argparse
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

_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SESSION = resolve(_S, _T)
RESULTS = results_dir(_S, _T, "glm")

BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
TARGET_FDR, JITTER_MS, N_SURR, SEED = 0.70, 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))
TYPING_LAGS, TYPING_FRACTION = 2, 0.25


def fdr_threshold(obs, null, target):
    cand = np.unique(obs)
    obs_s = np.sort(obs)
    null_s = np.sort(null.ravel())
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / null.shape[0]
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    return float(cand[ok[0]]) if len(ok) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
    if a.n_recordings:
        n = min(a.n_recordings, len(bnd) - 1)
        M = M[:, :bnd[n]].tocsr()
        bnd = bnd[:n + 1]
    gt = sg.load_ground_truth(SESSION)
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    A_exc, A_inh = gt["A_exc"] & off, gt["A_inh"] & off
    print("%s/%s: %d recordings, %d neurons | true %d exc, %d inh"
          % (_S, _T, len(bnd) - 1, N, A_exc.sum(), A_inh.sum()), flush=True)

    B = sg.fit_B(M, bnd, max_lag=MAX_LAG, l2=L2)
    W = B[:KSUM].sum(0)
    np.fill_diagonal(W, 0.0)
    tscore = typing_score(B, k=TYPING_LAGS)
    print("  fit %.0fs" % (time.time() - t0), flush=True)

    rng = np.random.default_rng(SEED)
    null_W = []
    for i in range(N_SURR):
        Bs = sg.fit_B(sg.jitter(M, bnd, JBINS, rng), bnd, max_lag=MAX_LAG, l2=L2)
        Ws = Bs[:KSUM].sum(0)
        np.fill_diagonal(Ws, 0.0)
        null_W.append(Ws)
        print("    surrogate %d/%d" % (i + 1, N_SURR), flush=True)

    thr_e = fdr_threshold(W[off], np.stack([w[off] for w in null_W]), TARGET_FDR)
    edges_exc = np.zeros((N, N), bool)
    edges_exc[off] = W[off] >= thr_e

    mask = infer_inhibitory(W, score=tscore, typing="rank",
                            fraction=TYPING_FRACTION)
    pre_inh = np.zeros((N, N), bool)
    pre_inh[np.where(mask)[0], :] = True
    cand_i = off & pre_inh
    edges_inh = np.zeros((N, N), bool)
    if cand_i.any():
        thr_i = fdr_threshold((-W)[cand_i],
                              np.stack([(-w)[cand_i] for w in null_W]), TARGET_FDR)
        edges_inh[cand_i] = (-W)[cand_i] >= thr_i

    pred = edges_exc | edges_inh
    out = os.path.join(RESULTS, "glm_connectivity_sum4_5ms.npz")
    np.savez_compressed(out, W=W, candidates=off,
                        inferred_inhibitory=mask, edges_exc=edges_exc,
                        edges_inh=edges_inh, pred_adjacency=pred,
                        A_exc=gt["A_exc"], A_inh=gt["A_inh"],
                        n_recordings=len(bnd) - 1, typing="rank",
                        typing_fraction=TYPING_FRACTION, target_fdr=TARGET_FDR)
    print("\n  typed inhibitory %d/%d (correct %d)"
          % (mask.sum(), N, int((mask & gt["is_inhibitory"].astype(bool)).sum())))
    print("  edges_exc %d (TP %d) | edges_inh %d (TP %d) | all %d (TP %d)"
          % (edges_exc.sum(), (edges_exc & A_exc).sum(),
             edges_inh.sum(), (edges_inh & A_inh).sum(),
             pred.sum(), (pred & (A_exc | A_inh)).sum()), flush=True)
    print("  saved -> %s" % out, flush=True)
    print("  TOTAL %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
