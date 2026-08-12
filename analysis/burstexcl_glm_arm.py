"""Part A: does the initialization-locked first burst drive GLM edge recovery?

Three arms over the SAME 200 recordings, identical GLM settings and identical
jitter seed. The only difference is which time window is removed:

    A  full 60 s                     baseline
    B  drop 0-6 s of every recording removes the IC-locked burst
    C  drop 30-36 s of every rec     same data loss, artifact retained

C is what makes it conclusive: without it, a B-vs-A difference cannot be
separated from "10% less data".

Trimming DROPS bins (it does not zero them), and boundaries are recomputed per
arm so lagged features cannot cross a cut. Arm C splits every recording into two
segments (0-30 s and 36-60 s), so it has 400 segments to arm A/B's 200 -- without
that split, features would leak across the excised span.

Settings are the shipped operating point from analysis/glm_labelfree_scaling.py:
bin_ms=5, max_lag=6, l2=2.0, readout sum4, jitter 25 ms, 8 surrogates, seed 1.

    python burstexcl_glm_arm.py --arm B

Writes ``burstexcl_<arm>_<tag>.json`` and ``burstexcl_<arm>_<tag>.npz`` into the
session directory. Existing GLM outputs are never touched.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg  # noqa: E402

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
from session_paths import results_dir  # noqa: E402
_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
RESULTS = results_dir(_S, _T, "ic_artifact")
FIGS = RESULTS

# --- shipped operating point (must match glm_labelfree_scaling.py exactly) ---
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
TARGET_FDR, JITTER_MS, N_SURR, SEED = 0.70, 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))

# --- arms: list of (drop_start_ms, drop_end_ms); empty = keep everything ---
ARMS = {
    "A": {"drop": None, "tag": "full"},
    "B": {"drop": (0.0, 6000.0), "tag": "drop0-6s"},
    "C": {"drop": (30000.0, 36000.0), "tag": "drop30-36s"},
}


def keep_intervals(rec_bins, drop):
    """Bin intervals to KEEP within one recording of ``rec_bins`` bins."""
    if drop is None:
        return [(0, rec_bins)]
    lo = int(round(drop[0] / BIN_MS))
    hi = int(round(drop[1] / BIN_MS))
    lo, hi = max(0, lo), min(rec_bins, hi)
    out = []
    if lo > 0:
        out.append((0, lo))
    if hi < rec_bins:
        out.append((hi, rec_bins))
    return out


def trim(M, bnd, drop):
    """Drop a window from every recording; return (M_trimmed, boundaries).

    Every kept interval becomes its OWN segment, so a recording cut in the
    middle contributes two boundaries and lagged features cannot bridge the cut.
    """
    blocks, new_bnd = [], [0]
    for r in range(len(bnd) - 1):
        s, e = bnd[r], bnd[r + 1]
        for (a, b) in keep_intervals(e - s, drop):
            blocks.append(M[:, s + a:s + b])
            new_bnd.append(new_bnd[-1] + (b - a))
    return sp.hstack(blocks, format="csr"), new_bnd


def verify_no_leak(M, bnd, max_lag):
    """Assert lagged features are zero in the first ``lag`` bins of every segment."""
    bad = 0
    for k in range(max_lag):
        lag = k + 1
        S = sg._shift(M, bnd, lag)
        Sc = S.tocsc()
        for s in bnd[:-1]:
            if Sc[:, s:s + lag].nnz != 0:
                bad += 1
    return bad


def fdr_threshold(obs, null_scores, target):
    """Shipped label-free threshold rule; also returns the estimated FDR there."""
    cand = np.unique(obs)
    obs_s = np.sort(obs)
    null_s = np.sort(null_scores.ravel())
    n_surr = null_scores.shape[0]
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    if not len(ok):
        return float("inf"), float("nan")
    return float(cand[ok[0]]), float(fdr[ok[0]])


def sum4_W(Mn, bndn):
    B = sg.fit_B(Mn, bndn, max_lag=MAX_LAG, l2=L2)
    W = B[:KSUM].sum(0)
    np.fill_diagonal(W, 0.0)
    return W


def raw_spike_counts(drop):
    """True spike counts (not binarised) retained vs dropped, from the npz files."""
    paths = [p for p in sorted(glob.glob(os.path.join(SESSION, "recording*.npz")))
             if "raster" not in os.path.basename(p)]
    total = kept = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        for t in d["spike_times"]:
            t = np.atleast_1d(np.asarray(t, float))
            total += t.size
            if drop is None:
                kept += t.size
            else:
                kept += int(((t < drop[0]) | (t >= drop[1])).sum())
    return total, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()

    arm = ARMS[a.arm]
    drop, tag = arm["drop"], arm["tag"]
    t0 = time.time()

    print("=== arm %s (%s) ===" % (a.arm, tag), flush=True)
    M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
    N = M.shape[0]
    nnz_full = M.nnz
    print("  loaded [%d x %d] %d recordings, %d occupied bins | %.1fs"
          % (N, M.shape[1], len(bnd) - 1, nnz_full, time.time() - t0), flush=True)

    Mn, bndn = trim(M, bnd, drop)
    del M
    print("  trimmed -> [%d x %d], %d segments, %d occupied bins (%.2f%% of full)"
          % (N, Mn.shape[1], len(bndn) - 1, Mn.nnz, 100.0 * Mn.nnz / nnz_full),
          flush=True)

    if a.verify:
        bad = verify_no_leak(Mn, bndn, MAX_LAG)
        print("  leakage check: %d segment-starts with non-zero lagged features "
              "(must be 0)" % bad, flush=True)

    total_sp, kept_sp = raw_spike_counts(drop)
    print("  raw spikes: kept %d / %d = %.4f (dropped %.4f)"
          % (kept_sp, total_sp, kept_sp / total_sp, 1 - kept_sp / total_sp),
          flush=True)

    gt = sg.load_ground_truth(SESSION)
    cand = ~np.eye(N, dtype=bool)
    ye = gt["A_exc"][cand].astype(bool)
    yi = gt["A_inh"][cand].astype(bool)

    tf = time.time()
    W = sum4_W(Mn, bndn)
    se = W[cand]
    print("  observed fit in %.1fs" % (time.time() - tf), flush=True)

    rng = np.random.default_rng(SEED)          # identical seed across arms
    null = []
    for i in range(N_SURR):
        ts = time.time()
        null.append(sum4_W(sg.jitter(Mn, bndn, JBINS, rng), bndn)[cand])
        print("    surrogate %d/%d  %.1fs" % (i + 1, N_SURR, time.time() - ts),
              flush=True)
    null = np.stack(null)

    thr, est_fdr = fdr_threshold(se, null, TARGET_FDR)
    pred = se >= thr if np.isfinite(thr) else np.zeros_like(se, bool)
    TP = int((pred & ye).sum())
    FP = int((pred & ~ye).sum())
    FN = int((~pred & ye).sum())
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    realized = FP / max(TP + FP, 1)

    from sklearn.metrics import roc_auc_score, average_precision_score
    # Threshold-free, per layer, using the SIGNED score in the layer's direction.
    exc_auc = float(roc_auc_score(ye, se))
    exc_ap = float(average_precision_score(ye, se))
    inh_auc = float(roc_auc_score(yi, -se))
    inh_ap = float(average_precision_score(yi, -se))

    res = dict(
        arm=a.arm, tag=tag, drop_ms=drop,
        n_segments=len(bndn) - 1, n_bins=int(Mn.shape[1]),
        occupied_bins=int(Mn.nnz), occupied_bins_full=int(nnz_full),
        occupied_bins_frac=float(Mn.nnz / nnz_full),
        raw_spikes_total=int(total_sp), raw_spikes_kept=int(kept_sp),
        raw_spikes_frac_kept=float(kept_sp / total_sp),
        bin_ms=BIN_MS, max_lag=MAX_LAG, l2=L2, ksum=KSUM,
        target_fdr=TARGET_FDR, jitter_ms=JITTER_MS, n_surrogates=N_SURR,
        seed=SEED, leak_violations=int(bad) if a.verify else None,
        exc_auc=exc_auc, exc_ap=exc_ap, inh_auc=inh_auc, inh_ap=inh_ap,
        n_true_exc=int(ye.sum()), n_true_inh=int(yi.sum()),
        thr=thr, estimated_fdr=est_fdr, n_pred=int(TP + FP),
        TP=TP, FP=FP, FN=FN, precision=P, recall=R, f1=F1,
        realized_fdr=realized,
        elapsed_s=time.time() - t0,
    )

    out_json = os.path.join(RESULTS, "burstexcl_%s_%s.json" % (a.arm, tag))
    json.dump(res, open(out_json, "w"), indent=2)
    out_npz = os.path.join(RESULTS, "burstexcl_%s_%s.npz" % (a.arm, tag))
    np.savez_compressed(out_npz, W=W.astype(np.float32), pred=pred, thr=thr,
                        se=se.astype(np.float32))

    print("\n--- arm %s (%s) ---" % (a.arm, tag), flush=True)
    print("  exc AUC %.4f  AP %.4f   |   inh AUC %.4f  AP %.4f"
          % (exc_auc, exc_ap, inh_auc, inh_ap), flush=True)
    print("  thr %.5f  est_FDR %.4f  n_pred %d  TP %d  FP %d"
          % (thr, est_fdr, TP + FP, TP, FP), flush=True)
    print("  P %.4f  R %.4f  F1 %.4f  realized_FDR %.4f"
          % (P, R, F1, realized), flush=True)
    print("  -> %s" % out_json, flush=True)
    print("  TOTAL %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
