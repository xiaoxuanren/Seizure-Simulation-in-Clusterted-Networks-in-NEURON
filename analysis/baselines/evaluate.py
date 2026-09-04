"""Evaluation against the wired ground truth -- the ONLY place labels are read.

Reuses the existing harness rather than re-implementing it:
``auc_ap``, ``oracle``, ``prf`` come from ``chtc/analysis_one.py`` (the
duration-grid worker, numpy-only, validated against scikit-learn), and the
E/I rank typing from ``glm_connectivity.infer_inhibitory``.

Read-outs of a signed score matrix ``S[pre, post]`` (positive = excitatory
evidence, negative = inhibitory):

    exc        +S         vs A_exc          (denominator: excitatory edges)
    inh        -S         vs A_inh          (denominator: inhibitory edges)
    all_absS   |S|        vs A_exc | A_inh  (untyped protocol)
    typed      +S on rows typed excitatory, -S on rows typed inhibitory
                          vs A_exc | A_inh  (typed protocol, rank rule 0.25)

Every predicted mask is scored under TWO denominators, explicitly labelled:
``all_edges`` (truth = A_exc | A_inh) and ``exc_only`` (truth = A_exc; a
prediction landing on an inhibitory edge counts as a false positive, the
convention of ``glm_connectivity.evaluate``'s ``confusion_excitatory``).

Candidate set: every ordered off-diagonal pair, identical for all methods.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import sparse_glm as _sg  # noqa: E402
from baselines.labelfree import rank_typing, _a1  # noqa: E402

auc_ap = _a1.auc_ap
oracle = _a1.oracle
prf = _a1.prf

#: conduction-delay model of the simulator (network_builder.py defaults)
SYN_DELAY_MS, DELAY_PER_DISTANCE = 1.5, 2.0
DISTANCE_GRID = np.arange(0.0, 16.01, 0.5)


class GroundTruth:
    """Exact wired graph of a dataset, evaluation only."""

    def __init__(self, session_dir):
        gt = _sg.load_ground_truth(session_dir)
        net = np.load(sorted(glob.glob(os.path.join(session_dir, "network_*.npz")))[0],
                      allow_pickle=True)
        self.n = n = len(gt["is_inhibitory"])
        self.off = ~np.eye(n, dtype=bool)
        self.is_inhibitory = np.asarray(gt["is_inhibitory"], bool)
        self.A_exc = gt["A_exc"] & self.off
        self.A_inh = gt["A_inh"] & self.off
        self.A_all = self.A_exc | self.A_inh
        self.positions = np.asarray(net["neuron_positions"], float)
        self.clusters = np.asarray(net["cluster_assignments"], int)
        self.distance = np.sqrt(((self.positions[:, None, :] - self.positions[None, :, :]) ** 2).sum(-1))
        self.weight = np.zeros((n, n))
        for row in net["connections"]:
            self.weight[int(row[0]), int(row[1])] = abs(float(row[2]))
        self.delay_ms = SYN_DELAY_MS + DELAY_PER_DISTANCE * self.distance   # meaningful on edges
        self.n_exc, self.n_inh, self.n_all = int(self.A_exc.sum()), int(self.A_inh.sum()), int(self.A_all.sum())


# ------------------------------------------------------------------------- #
# read-outs
# ------------------------------------------------------------------------- #
def typed_score(S, inh_mask):
    T = np.array(S, float, copy=True)
    T[np.asarray(inh_mask, bool), :] *= -1.0
    return T


def readouts(S, gt, inh_mask=None, typing_fraction=0.25):
    """{name: (score, truth)} for the four read-outs."""
    S = np.asarray(S, float)
    if inh_mask is None:
        inh_mask = rank_typing(-S.sum(1), typing_fraction)
    return {
        "exc": (S, gt.A_exc),
        "inh": (-S, gt.A_inh),
        "all_absS": (np.abs(S), gt.A_all),
        "typed": (typed_score(S, inh_mask), gt.A_all),
    }, inh_mask


def confusions(pred, gt):
    """Confusion under both denominators.  Returns {denominator: prf-dict}."""
    pred = np.asarray(pred, bool) & gt.off
    out = {}
    for denom, truth in (("all_edges", gt.A_all), ("exc_only", gt.A_exc)):
        m = prf(pred, truth)
        m["realized_fdr"] = 1.0 - m["precision"] if m["n_pred"] else float("nan")
        m["n_true"] = int(truth.sum())
        out[denom] = m
    return out


def threshold_free(S, gt, inh_mask=None, typing_fraction=0.25):
    """AUC/AP + oracle ceilings per read-out, plus the oracle best-F1 masks."""
    ro, inh_mask = readouts(S, gt, inh_mask, typing_fraction)
    res, masks = {}, {}
    for name, (score, truth) in ro.items():
        s, t = score[gt.off], truth[gt.off]
        auc, ap = auc_ap(t, s)
        orc = oracle(s, t)
        # the oracle best-F1 mask: top-k by score with k from the oracle sweep
        order = np.argsort(s)[::-1]
        ys = t[order].astype(float)
        tp = np.cumsum(ys)
        k = np.arange(1, len(ys) + 1)
        f1 = 2 * tp / np.maximum(k + ys.sum(), 1e-12)
        kbest = int(np.argmax(f1)) + 1
        thr = float(s[order[kbest - 1]])
        mask = np.zeros((gt.n, gt.n), bool)
        mask[gt.off] = score[gt.off] >= thr
        masks[name] = mask
        res[name] = dict(auc=auc, ap=ap, n_true=int(t.sum()), oracle=orc,
                         oracle_threshold=thr, oracle_n_pred=int(mask.sum()))
    return res, masks, inh_mask


def rows_threshold_free(res):
    """Flatten :func:`threshold_free` output into long-format rows."""
    rows = []
    for ro, r in res.items():
        denom = {"exc": "exc_only", "inh": "inh_only"}.get(ro, "all_edges")
        for k in ("auc", "ap", "n_true"):
            rows.append(dict(readout=ro, operating_point="threshold_free", denominator=denom,
                             metric=k, value=float(r[k])))
        o = r["oracle"]
        # precision / recall / F1 / n_pred of the oracle mask come from
        # rows_confusion (both denominators); only the threshold lives here
        rows.append(dict(readout=ro, operating_point="oracle_best_f1", denominator=denom,
                         metric="threshold", value=float(r["oracle_threshold"])))
        rows.append(dict(readout=ro, operating_point="oracle_true_fdr10", denominator=denom,
                         metric="recall", value=float(o["recall_at_true_fdr10"])))
    return rows


def rows_confusion(pred, gt, readout, operating_point, extra=None):
    rows = []
    for denom, m in confusions(pred, gt).items():
        for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_pred", "realized_fdr", "n_true"):
            rows.append(dict(readout=readout, operating_point=operating_point, denominator=denom,
                             metric=k, value=float(m[k])))
    for k, v in (extra or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            rows.append(dict(readout=readout, operating_point=operating_point, denominator="n/a",
                             metric=k, value=float(v)))
    return rows


# ------------------------------------------------------------------------- #
# structure-resolved recovery
# ------------------------------------------------------------------------- #
def distance_recovery(pred, gt, grid=DISTANCE_GRID, min_count=15):
    """Recall / precision of ``pred`` binned by inter-somatic distance, plus
    the near-half / far-half recall split at the median true-edge distance."""
    pred = np.asarray(pred, bool) & gt.off
    d_true = gt.distance[gt.A_all]
    d_tp = gt.distance[pred & gt.A_all]
    d_pred = gt.distance[pred]
    n_true, _ = np.histogram(d_true, grid)
    n_tp, _ = np.histogram(d_tp, grid)
    n_pred, _ = np.histogram(d_pred, grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_true >= min_count, n_tp / np.maximum(n_true, 1), np.nan)
        precision = np.where(n_pred >= min_count, n_tp / np.maximum(n_pred, 1), np.nan)
    med = float(np.median(d_true))
    near = float((d_tp < med).sum() / max((d_true < med).sum(), 1))
    far = float((d_tp >= med).sum() / max((d_true >= med).sum(), 1))
    missed = gt.distance[gt.A_all & ~pred]
    return dict(bin_edges=grid.tolist(), recall=recall.tolist(), precision=precision.tolist(),
                n_true=n_true.tolist(), n_tp=n_tp.tolist(), n_pred=n_pred.tolist(),
                median_true_distance=med, near_half_recall=near, far_half_recall=far,
                median_tp_distance=float(np.median(d_tp)) if d_tp.size else float("nan"),
                median_missed_distance=float(np.median(missed)) if missed.size else float("nan"))


def weight_quantile_recovery(pred, gt, q=5):
    """Detection probability by true |weight| quantile, WITHIN class."""
    pred = np.asarray(pred, bool) & gt.off
    out = {}
    for name, A in (("exc", gt.A_exc), ("inh", gt.A_inh)):
        w = gt.weight[A]
        det = pred[A]
        if w.size == 0:
            out[name] = [float("nan")] * q
            continue
        edges = np.quantile(w, np.linspace(0, 1, q + 1))
        vals = []
        for i in range(q):
            m = (w >= edges[i]) & ((w <= edges[i + 1]) if i == q - 1 else (w < edges[i + 1]))
            vals.append(float(det[m].mean()) if m.any() else float("nan"))
        out[name] = vals
    return out


def delay_recovery(profile, lags_ms, gt, signed=True):
    """Delay-recovery gate: for true excitatory edges, correlate the lag at
    which the method's evidence peaks with the true conduction delay.

    ``profile``: ``[L, N, N]`` evidence per lag (``[l, pre, post]``);
    ``lags_ms``: the lag centres in ms.  Also reports the same statistic on
    the TRANSPOSED profile: a transposed method shows up as r_transposed >>
    r.  Returns dict(r, r_transposed, n_edges, ...)."""
    P = np.asarray(profile, float)
    L = P.shape[0]
    lags_ms = np.asarray(lags_ms, float)
    assert len(lags_ms) == L
    pre, post = np.where(gt.A_exc)
    true_delay = gt.delay_ms[pre, post]

    def peak_lag(E):
        prof = E[:, pre, post]                       # [L, n_edges]
        idx = np.argmax(prof if signed else np.abs(prof), 0)
        return lags_ms[idx]

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    pk = peak_lag(P)
    pk_t = peak_lag(np.transpose(P, (0, 2, 1)))
    return dict(r=corr(pk, true_delay), r_transposed=corr(pk_t, true_delay),
                n_edges=int(len(pre)), lag_resolution_ms=float(np.min(np.diff(lags_ms))) if L > 1 else float("nan"),
                mean_abs_error_ms=float(np.mean(np.abs(pk - true_delay))),
                true_delay_range_ms=[float(true_delay.min()), float(true_delay.max())])
