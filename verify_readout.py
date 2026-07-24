#!/usr/bin/env python3
"""Verify the lag1 -> peak readout change on a saved session.

Reproduces, from spikes alone, (a) the per-lag ranking sweep and (b) the
label-free jitter-null operating points for each readout, so you can confirm
per-dataset which reduction to ship.

    python verify_readout.py --session "notebooks/NEURON data parallel/normal/20260721_163430_spikeonly"

Expected on the 926-neuron / 50-recording normal flagship (max_lag=6, l2=2.0,
bin 5 ms, 6 surrogates, seed 1):

    readout  FDR    TP     FP    FN     P      R      F1
    lag1     0.10   3724   203   9632   0.948  0.279  0.431
    peak     0.05   6253   123   7103   0.981  0.468  0.634
    peak     0.10   6994   306   6362   0.958  0.524  0.677

IMPORTANT -- the gain is data-dependent. The per-edge argmax that 'peak' relies
on needs enough spikes to be stable. On the flagship (775k spikes) peak nearly
doubles TP; on a 3-recording session (14k spikes) it is roughly neutral on TP
and merely raises precision. This script prints both so you can decide per
session rather than assuming.

Requires numpy, scipy, scikit-learn. No torch. Memory-safe: uses a sparse
backend, so it does not materialise the [max_lag*N, T] dense feature block
(13 GB at N=926, max_lag=6).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg  # noqa: E402


def fdr_threshold(obs, null_stack, target_fdr):
    """Loosest threshold whose expected surrogate exceedances / observed <= target."""
    u = np.unique(obs)
    obs_sorted = np.sort(obs)
    null_sorted = np.sort(null_stack.ravel())
    n_surr = null_stack.shape[0]
    n_obs = len(obs_sorted) - np.searchsorted(obs_sorted, u, side="left")
    n_null = (len(null_sorted) - np.searchsorted(null_sorted, u, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target_fdr)[0]
    return float(u[ok[0]]) if len(ok) else float("inf")


def confusion(pred, truth, off):
    TP = int((pred & truth).sum())
    FP = int((pred & ~truth & off).sum())
    FN = int(truth.sum()) - TP
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / int(truth.sum()) if truth.sum() else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return TP, FP, FN, P, R, F1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--bin-ms", type=float, default=5.0)
    ap.add_argument("--max-lag", type=int, default=6)
    ap.add_argument("--l2", type=float, default=2.0)
    ap.add_argument("--jitter-ms", type=float, default=25.0)
    ap.add_argument("--n-surrogates", type=int, default=6)
    ap.add_argument("--target-fdr", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--readouts", nargs="+", default=["lag1", "sum", "peak"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    M, bnd = sg.load_session(a.session, a.bin_ms)
    n = M.shape[0]
    print("[verify] %s" % a.session)
    print("[verify] %d recordings | spike matrix [%d x %d] (%.0fs) | %d spikes | load %.1fs"
          % (len(bnd) - 1, n, M.shape[1], M.shape[1] * a.bin_ms / 1000,
             int(M.sum()), time.time() - t0), flush=True)

    t0 = time.time()
    B = sg.fit_B(M, bnd, max_lag=a.max_lag, l2=a.l2)
    print("[verify] joint lag fit (max_lag=%d, l2=%.1f) in %.1fs"
          % (a.max_lag, a.l2, time.time() - t0), flush=True)

    try:
        gt = sg.load_ground_truth(a.session)
    except (IndexError, KeyError):
        gt = None

    off = ~np.eye(n, dtype=bool)

    # ---- (a) per-lag ranking sweep -------------------------------------- #
    if gt is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        ye, yi = gt["A_exc"][off], gt["A_inh"][off]
        print("\n[verify] per-lag ranking | candidates=%d  true exc=%d  inh=%d"
              % (off.sum(), ye.sum(), yi.sum()))
        print("  lag   window     exc_AUC  exc_AP   inh_AUC  inh_AP")
        for k in range(a.max_lag):
            W = B[k].copy()
            np.fill_diagonal(W, 0.0)
            se, si = W[off], -W[off]
            print("  %-3d   %2d-%2dms    %.3f    %.3f    %.3f    %.3f"
                  % (k + 1, k * a.bin_ms, (k + 1) * a.bin_ms,
                     roc_auc_score(ye, se), average_precision_score(ye, se),
                     roc_auc_score(yi, si), average_precision_score(yi, si)))
        ea = [roc_auc_score(ye, np.fill_diagonal(B[k].copy(), 0) or B[k][off])
              for k in range(a.max_lag)]
        print("  -> excitatory peaks at lag %d; inhibitory at lag %d"
              % (int(np.argmax(ea)) + 1,
                 int(np.argmax([roc_auc_score(yi, -B[k][off]) for k in range(a.max_lag)])) + 1))

    # ---- (b) jitter-null operating points per readout -------------------- #
    jitter_bins = max(1, int(round(a.jitter_ms / a.bin_ms)))
    rng = np.random.default_rng(a.seed)
    print("\n[verify] fitting %d spike-jitter surrogates (+/-%.0f ms)..."
          % (a.n_surrogates, a.jitter_ms), flush=True)
    nulls = {r: [] for r in a.readouts}
    for i in range(a.n_surrogates):
        ts = time.time()
        Bs = sg.fit_B(sg.jitter(M, bnd, jitter_bins, rng), bnd,
                      max_lag=a.max_lag, l2=a.l2)
        for r in a.readouts:
            nulls[r].append(sg.readout(Bs, r))
        del Bs
        print("  surrogate %d/%d (%.0fs)" % (i + 1, a.n_surrogates, time.time() - ts),
              flush=True)

    if gt is not None:
        A = (gt["A_exc"] | gt["A_inh"]) & off
        print("\n[verify] label-free operating points (whole-map, %d true edges)"
              % int(A.sum()))
        print("  %-8s %-6s %7s %6s %7s %7s %7s %7s  %s"
              % ("readout", "FDR", "TP", "FP", "FN", "P", "R", "F1", "realized FDR"))
    else:
        print("\n[verify] no network file -> reporting predicted edge counts only")
        print("  %-8s %-6s %10s" % ("readout", "FDR", "pred edges"))

    baseline = None
    for r in a.readouts:
        W = sg.readout(B, r)
        S = np.stack(nulls[r])
        inh = W.sum(1) < 0.0
        pre_inh = np.zeros((n, n), bool)
        pre_inh[np.where(inh)[0], :] = True
        cand_i = off & pre_inh
        for tf in a.target_fdr:
            te = fdr_threshold(W[off], np.stack([s[off] for s in S]), tf)
            ti = fdr_threshold((-W)[cand_i], np.stack([(-s)[cand_i] for s in S]), tf)
            pred = ((W >= te) & off) | (((-W) >= ti) & cand_i)
            if gt is None:
                print("  %-8s %-6.2f %10d" % (r, tf, int(pred.sum())))
                continue
            TP, FP, FN, P, R, F1 = confusion(pred, A, off)
            print("  %-8s %-6.2f %7d %6d %7d %7.3f %7.3f %7.3f  %.3f"
                  % (r, tf, TP, FP, FN, P, R, F1, 1 - P))
            if r == "lag1" and abs(tf - 0.10) < 1e-9:
                baseline = (TP, FP, FN)

    if gt is not None and baseline is not None and "peak" in a.readouts:
        bTP, bFP, bFN = baseline
        print("\n[verify] vs shipped lag1 @FDR0.10 (TP=%d FP=%d FN=%d):" % baseline)
        W = sg.readout(B, "peak")
        S = np.stack(nulls["peak"])
        inh = W.sum(1) < 0.0
        pre_inh = np.zeros((n, n), bool)
        pre_inh[np.where(inh)[0], :] = True
        cand_i = off & pre_inh
        best = None
        for tf in a.target_fdr:
            te = fdr_threshold(W[off], np.stack([s[off] for s in S]), tf)
            ti = fdr_threshold((-W)[cand_i], np.stack([(-s)[cand_i] for s in S]), tf)
            pred = ((W >= te) & off) | (((-W) >= ti) & cand_i)
            TP, FP, FN, P, R, F1 = confusion(pred, A, off)
            pareto = (TP > bTP) and (FP <= bFP)
            tag = "STRICT PARETO WIN" if pareto else ("more TP, more FP" if TP > bTP
                                                     else "no TP gain")
            print("  peak @FDR%.2f: dTP=%+d dFP=%+d dFN=%+d  [%s]"
                  % (tf, TP - bTP, FP - bFP, FN - bFN, tag))
            if pareto and (best is None or TP > best[1]):
                best = (tf, TP)
        print("  -> recommendation: %s"
              % ("ship readout='peak' at target_fdr=%.2f" % best[0] if best
                 else "peak is NOT a strict win on this session; keep lag1 "
                      "(likely too few spikes for a stable per-edge argmax)"))


if __name__ == "__main__":
    main()
