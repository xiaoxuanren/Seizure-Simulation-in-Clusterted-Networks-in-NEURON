"""Run ONE duration-grid analysis point on ONE dataset (a single CHTC job).

Reproduces the flagship deck's duration-resolved protocol for a sweep
session: fit the sum4 GLM on the first n recordings, then report, per
layer (excitatory / inhibitory / typed):

  * threshold-free ranking quality: AUC and average precision (numpy-only
    implementations -- the worker container has no scikit-learn)
  * the label-free operating curve: for each nominal FDR target in a grid,
    the jitter-null threshold, estimated FDR, and REALIZED precision/recall
    (the calibration surface of deck slide 24)
  * oracle ceilings: best-F1 threshold and recall at true FDR 0.10
    (deck slide 26)

Inputs: the session's spike-only recordings, shipped per job as
spikes_<session>.tar.gz (extracted to data/<session>/), plus the repo code.

    python chtc/analysis_one.py --session sweep_c50_seed01 --n-recordings 50 \
        --data data --out out
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, os.path.join(REPO, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sparse_glm as sg  # noqa: E402
from glm_connectivity import infer_inhibitory, typing_score  # noqa: E402

BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
JITTER_MS, N_SURR, SEED = 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))
TARGETS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TYPING_LAGS, TYPING_FRACTION = 2, 0.25


def auc_ap(y, score):
    """Rank AUC + average precision, numpy only."""
    order = np.argsort(score)[::-1]
    ys = y[order].astype(float)
    n_pos, n_neg = ys.sum(), len(ys) - ys.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)   # ties: fine for dense scores
    auc = (ranks[y.astype(bool)].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    ap = (prec * ys).sum() / n_pos
    return float(auc), float(ap)


def fdr_threshold(obs, null_scores, target):
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


def prf(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, precision=P, recall=R,
                f1=2 * P * R / max(P + R, 1e-12), n_pred=int(pred.sum()))


def oracle(score_flat, true_flat):
    """Best-F1 over all thresholds + recall at true FDR 0.10 (vectorized)."""
    order = np.argsort(score_flat)[::-1]
    ys = true_flat[order].astype(float)
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec = tp / k
    rec = tp / max(ys.sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    ok = np.where(1 - prec <= 0.10)[0]
    r10 = float(rec[ok[-1]]) if len(ok) else 0.0
    return dict(best_f1=float(f1[i]), best_f1_precision=float(prec[i]),
                best_f1_recall=float(rec[i]), recall_at_true_fdr10=r10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--n-recordings", type=int, required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    sd = os.path.join(a.data, a.session)
    if not glob.glob(os.path.join(sd, "recording*.npz")):
        sd = os.path.join(sd, "normal")          # repo layout fallback
    M, bnd = sg.load_session(sd, bin_ms=BIN_MS)
    n = M.shape[0]
    if a.n_recordings < len(bnd) - 1:
        end = bnd[a.n_recordings]
        M, bnd = M[:, :end], bnd[:a.n_recordings + 1]
    gt = sg.load_ground_truth(sd)
    off = ~np.eye(n, dtype=bool)
    A_exc, A_inh = gt["A_exc"] & off, gt["A_inh"] & off
    A_all = A_exc | A_inh

    B = sg.fit_B(M, bnd, max_lag=MAX_LAG, l2=L2)
    W = B[:KSUM].sum(0)
    np.fill_diagonal(W, 0.0)

    rng = np.random.default_rng(SEED)
    nullW = []
    for _ in range(N_SURR):
        Bj = sg.fit_B(sg.jitter(M, bnd, JBINS, rng), bnd, max_lag=MAX_LAG, l2=L2)
        Wj = Bj[:KSUM].sum(0)
        np.fill_diagonal(Wj, 0.0)
        nullW.append(Wj)

    is_inh_pred = np.asarray(infer_inhibitory(
        W, score=typing_score(B, k=TYPING_LAGS), fraction=TYPING_FRACTION), bool)

    res = dict(session=a.session, n_recordings=a.n_recordings,
               n_neurons=int(n), layers={})

    layers = {
        "exc": (W, np.array([Wj for Wj in nullW]), A_exc, None),
        "inh": (-W, np.array([-Wj for Wj in nullW]), A_inh, None),
        "all_absW": (np.abs(W), np.array([np.abs(Wj) for Wj in nullW]), A_all, None),
    }
    for name, (score, nulls, truth, _mask) in layers.items():
        sflat, tflat = score[off], truth[off]
        auc, apv = auc_ap(tflat, sflat)
        layer = dict(auc=auc, ap=apv, n_true=int(tflat.sum()),
                     oracle=oracle(sflat, tflat), targets=[])
        nflat = np.array([nl[off] for nl in nulls])
        for t in TARGETS:
            thr, est = fdr_threshold(sflat, nflat, t)
            pred = score > thr
            np.fill_diagonal(pred, False)
            m = prf(pred & off, truth)
            m.update(target=t, threshold=thr, est_fdr=est,
                     realized_fdr=1 - m["precision"])
            layer["targets"].append(m)
        res["layers"][name] = layer

    # typed layer at the standard 0.70 target: per-class scores restricted by
    # predicted presynaptic type
    typed_pred = np.zeros((n, n), bool)
    exc_score = W.copy(); exc_score[is_inh_pred, :] = -np.inf
    inh_score = -W.copy(); inh_score[~is_inh_pred, :] = -np.inf
    for score, nulls in ((exc_score, [Wj for Wj in nullW]),
                         (inh_score, [-Wj for Wj in nullW])):
        valid = np.isfinite(score) & off
        sflat = score[valid]
        nflat = np.array([nl[valid] for nl in nulls])
        thr, _ = fdr_threshold(sflat, nflat, 0.70)
        typed_pred |= (score > thr) & valid
    res["typed_at_070"] = prf(typed_pred, A_all)
    res["typing_n_inh_pred"] = int(is_inh_pred.sum())
    res["typing_n_inh_true"] = int(np.asarray(gt["is_inhibitory"]).sum())

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "durgrid_%s_n%03d.json" % (a.session, a.n_recordings))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    e = res["layers"]["exc"]
    print("[%s n=%d] exc AUC %.3f AP %.3f | typed@.70 P %.3f R %.3f -> %s"
          % (a.session, a.n_recordings, e["auc"], e["ap"],
             res["typed_at_070"]["precision"], res["typed_at_070"]["recall"], out),
          flush=True)


if __name__ == "__main__":
    main()
