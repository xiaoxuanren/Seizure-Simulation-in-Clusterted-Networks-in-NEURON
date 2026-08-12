"""A1: E/I typing collapse -- measure the fix at n = 200.

The old rule ``W.sum(1) < 0`` fires on 1 of 926 neurons at n=200, which collapses
the inhibitory candidate set. The ranking was never the problem (inhibitory
AUC 0.960 / AP 0.459); the zero cut discards it.

This measures the shipped ``infer_inhibitory`` (imported, not reimplemented) on
the flagship session under all three typing rules, at the shipped operating
point: bin_ms=5, max_lag=6, l2=2.0, readout sum4, target_fdr=0.70, 8 jitter
surrogates, seed 1, sparse path (dense would need ~80 GB at n=200).

One GLM fit serves everything: B is readout-independent, so the edge score
(sum4) and the typing score (lag1+lag2 row sums) come from the same fit, and one
surrogate pass yields both nulls.

Reports four tables -- {combined all-edge denominator, separated} x {before,
after} -- plus fraction sensitivity and the rank-vs-null comparison.

Writes ``a1_typing_fix_results.json`` and ``a1_typing_fix_tables.csv``.
"""

import argparse
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

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
from session_paths import results_dir  # noqa: E402
_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
RESULTS = results_dir(_S, _T, "glm")
FIGS = RESULTS
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
TARGET_FDR, JITTER_MS, N_SURR, SEED = 0.70, 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))
TYPING_LAGS = 2
FRACTIONS = (0.15, 0.20, 0.25, 0.30)
DEFAULT_FRACTION = 0.25
NULL_Q = 0.70


def fdr_threshold(obs, null, target):
    cand = np.unique(obs)
    obs_s = np.sort(obs)
    null_s = np.sort(null.ravel())
    n_surr = null.shape[0]
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    if not len(ok):
        return float("inf"), float("nan")
    return float(cand[ok[0]]), float(fdr[ok[0]])


def confusion(pred, truth, thr, est_fdr):
    TP = int((pred & truth).sum())
    FP = int((pred & ~truth).sum())
    FN = int((truth & ~pred).sum())
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    return dict(thr=thr, estimated_fdr=est_fdr, n_pred=TP + FP, TP=TP, FP=FP,
                FN=FN, precision=P, recall=R,
                f1=2 * P * R / max(P + R, 1e-12),
                realized_fdr=FP / max(TP + FP, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=None,
                    help="cap the number of recordings used (default: all). "
                         "Results are NOT comparable across different caps.")
    a = ap.parse_args()

    t0 = time.time()
    M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
    if a.n_recordings is not None:
        n_rec = min(a.n_recordings, len(bnd) - 1)
        M = M[:, :bnd[n_rec]].tocsr()
        bnd = bnd[:n_rec + 1]
        print("[cap] using %d of the available recordings" % n_rec, flush=True)
    gt = sg.load_ground_truth(SESSION)
    N = M.shape[0]
    off = ~np.eye(N, dtype=bool)
    A_exc, A_inh = gt["A_exc"] & off, gt["A_inh"] & off
    A_all = (gt["A_exc"] | gt["A_inh"]) & off
    print("loaded [%d x %d] %d rec | true exc %d, inh %d, all %d | %.1fs"
          % (N, M.shape[1], len(bnd) - 1, A_exc.sum(), A_inh.sum(), A_all.sum(),
             time.time() - t0), flush=True)

    # --- one fit serves every variant -------------------------------------- #
    tf = time.time()
    B = sg.fit_B(M, bnd, max_lag=MAX_LAG, l2=L2)
    W = B[:KSUM].sum(0)
    np.fill_diagonal(W, 0.0)
    tscore = typing_score(B, k=TYPING_LAGS)
    print("  observed fit %.1fs" % (time.time() - tf), flush=True)

    rng = np.random.default_rng(SEED)
    null_W, null_t = [], []
    for i in range(N_SURR):
        ts = time.time()
        Bs = sg.fit_B(sg.jitter(M, bnd, JBINS, rng), bnd, max_lag=MAX_LAG, l2=L2)
        Ws = Bs[:KSUM].sum(0)
        np.fill_diagonal(Ws, 0.0)
        null_W.append(Ws)
        null_t.append(typing_score(Bs, k=TYPING_LAGS))
        print("    surrogate %d/%d %.1fs" % (i + 1, N_SURR, time.time() - ts),
              flush=True)
    null_t = np.stack(null_t)

    # sanity: the sign rule's reach, and the row-sum medians that break it
    row = W.sum(1)
    is_inh = gt["is_inhibitory"].astype(bool)
    print("\n  row-sum medians: exc %+.4f  inh %+.4f  | W.sum(1)<0 fires on %d/%d"
          % (np.median(row[~is_inh]), np.median(row[is_inh]),
             int((row < 0).sum()), N), flush=True)

    from sklearn.metrics import roc_auc_score, average_precision_score
    thresh_free = dict(
        exc_auc=float(roc_auc_score(A_exc[off], W[off])),
        exc_ap=float(average_precision_score(A_exc[off], W[off])),
        inh_auc=float(roc_auc_score(A_inh[off], (-W)[off])),
        inh_ap=float(average_precision_score(A_inh[off], (-W)[off])),
        typing_auc=float(roc_auc_score(is_inh, tscore)),
    )
    print("  threshold-free: exc AUC %.4f AP %.4f | inh AUC %.4f AP %.4f | "
          "typing AUC %.4f" % (thresh_free["exc_auc"], thresh_free["exc_ap"],
                               thresh_free["inh_auc"], thresh_free["inh_ap"],
                               thresh_free["typing_auc"]), flush=True)

    # --- excitatory layer: independent of typing, computed once ------------ #
    obs_e = W[off]
    nulls_e = np.stack([w[off] for w in null_W])
    thr_e, est_e = fdr_threshold(obs_e, nulls_e, TARGET_FDR)
    edges_exc = np.zeros((N, N), bool)
    edges_exc[off] = obs_e >= thr_e
    exc_row = confusion(edges_exc, A_exc, thr_e, est_e)
    print("\n  excitatory layer (typing-independent): thr %.5f n_pred %d TP %d "
          "P %.4f R %.4f F1 %.4f" % (thr_e, exc_row["n_pred"], exc_row["TP"],
                                     exc_row["precision"], exc_row["recall"],
                                     exc_row["f1"]), flush=True)

    # --- inhibitory layer per typing variant ------------------------------- #
    variants = [("sign", dict(typing="sign"))]
    for f in FRACTIONS:
        variants.append(("rank@%.2f" % f, dict(typing="rank", fraction=f)))
    variants.append(("null@q%.2f" % NULL_Q, dict(typing="null", q=NULL_Q)))

    results, exc_masks = {}, {}
    for label, kw in variants:
        mask = infer_inhibitory(W, score=tscore, null_scores=null_t, **kw)
        pre_inh = np.zeros((N, N), bool)
        pre_inh[np.where(mask)[0], :] = True
        cand_i = off & pre_inh
        if cand_i.any():
            obs_i = (-W)[cand_i]
            nulls_i = np.stack([(-w)[cand_i] for w in null_W])
            thr_i, est_i = fdr_threshold(obs_i, nulls_i, TARGET_FDR)
            edges_inh = np.zeros((N, N), bool)
            edges_inh[cand_i] = obs_i >= thr_i
        else:
            thr_i, est_i = float("inf"), float("nan")
            edges_inh = np.zeros((N, N), bool)

        pred_all = edges_exc | edges_inh
        results[label] = dict(
            typing=label, n_typed_inhibitory=int(mask.sum()),
            typed_correct=int((mask & is_inh).sum()),
            typing_precision=float((mask & is_inh).sum() / max(mask.sum(), 1)),
            typing_recall=float((mask & is_inh).sum() / max(is_inh.sum(), 1)),
            separated_exc=exc_row,
            separated_inh=confusion(edges_inh, A_inh, thr_i, est_i),
            combined_all_edge=confusion(pred_all, A_all, thr_e, est_e),
            **thresh_free)
        exc_masks[label] = edges_exc.copy()

        r = results[label]
        print("\n  === typing = %s ===" % label, flush=True)
        print("    typed inhibitory %d/%d (correct %d, precision %.3f, recall %.3f)"
              % (r["n_typed_inhibitory"], N, r["typed_correct"],
                 r["typing_precision"], r["typing_recall"]), flush=True)
        i_, c_ = r["separated_inh"], r["combined_all_edge"]
        print("    inh layer : thr %.5f n_pred %5d TP %4d FP %4d P %.4f R %.4f "
              "F1 %.4f realFDR %.4f"
              % (i_["thr"], i_["n_pred"], i_["TP"], i_["FP"], i_["precision"],
                 i_["recall"], i_["f1"], i_["realized_fdr"]), flush=True)
        print("    combined  : n_pred %5d TP %5d FP %4d P %.4f R %.4f F1 %.4f "
              "realFDR %.4f"
              % (c_["n_pred"], c_["TP"], c_["FP"], c_["precision"], c_["recall"],
                 c_["f1"], c_["realized_fdr"]), flush=True)

    # --- the required verification ----------------------------------------- #
    base = exc_masks["sign"]
    identical = {k: bool(np.array_equal(base, v)) for k, v in exc_masks.items()}
    print("\n  edges_exc bit-identical across every typing variant: %s"
          % all(identical.values()), flush=True)
    for k, v in identical.items():
        print("      %-14s %s" % (k, "identical" if v else "*** DIFFERS ***"),
              flush=True)

    out = dict(config=dict(n_rec=len(bnd) - 1, bin_ms=BIN_MS, max_lag=MAX_LAG,
                           l2=L2, ksum=KSUM, target_fdr=TARGET_FDR,
                           jitter_ms=JITTER_MS, n_surrogates=N_SURR, seed=SEED,
                           typing_lags=TYPING_LAGS, null_q=NULL_Q),
               truth=dict(n_true_exc=int(A_exc.sum()), n_true_inh=int(A_inh.sum()),
                          n_true_all=int(A_all.sum())),
               row_sum_median_exc=float(np.median(row[~is_inh])),
               row_sum_median_inh=float(np.median(row[is_inh])),
               sign_rule_fires_on=int((row < 0).sum()),
               edges_exc_identical=identical,
               before="sign", after="rank@%.2f" % DEFAULT_FRACTION,
               variants=results)
    suffix = "" if a.n_recordings is None else "_n%d" % (len(bnd) - 1)
    p = os.path.join(RESULTS, "a1_typing_fix_results%s.json" % suffix)
    json.dump(out, open(p, "w"), indent=2)

    cols = ["variant", "table", "thr", "n_pred", "TP", "FP", "FN", "precision",
            "recall", "f1", "estimated_fdr", "realized_fdr", "exc_auc", "exc_ap",
            "inh_auc", "inh_ap"]
    lines = [",".join(cols)]
    for label, r in results.items():
        for tname, key in (("separated_exc", "separated_exc"),
                           ("separated_inh", "separated_inh"),
                           ("combined_all_edge", "combined_all_edge")):
            d = r[key]
            lines.append(",".join([label, tname] + [
                "%.6g" % d[c] if isinstance(d[c], float) else str(d[c])
                for c in ("thr", "n_pred", "TP", "FP", "FN", "precision",
                          "recall", "f1", "estimated_fdr", "realized_fdr")] + [
                "%.6g" % r[c] for c in ("exc_auc", "exc_ap", "inh_auc", "inh_ap")]))
    csvp = os.path.join(RESULTS, "a1_typing_fix_tables%s.csv" % suffix)
    open(csvp, "w").write("\n".join(lines) + "\n")
    print("\nsaved -> a1_typing_fix_results.json / a1_typing_fix_tables.csv",
          flush=True)
    print("TOTAL %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
