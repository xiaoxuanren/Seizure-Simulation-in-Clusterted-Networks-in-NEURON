"""Label-free operating-point helpers (NO ground truth anywhere in here).

The jitter-null FDR rule is the shipped one -- ``fdr_threshold`` is imported
from ``chtc/analysis_one.py`` rather than re-implemented -- and the typed
protocol follows the same script: the excitatory layer is thresholded over
pairs whose presynaptic neuron is NOT typed inhibitory, the inhibitory layer
over pairs whose presynaptic neuron IS typed inhibitory, each at the same
nominal FDR target, with E/I typing by the rank rule
(``glm_connectivity.infer_inhibitory(typing="rank", fraction=0.25)``).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from glm_connectivity import infer_inhibitory  # noqa: E402


def _load_analysis_one():
    spec = importlib.util.spec_from_file_location(
        "_chtc_analysis_one", os.path.join(REPO, "chtc", "analysis_one.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_a1 = _load_analysis_one()
fdr_threshold = _a1.fdr_threshold      # (obs, null[n_surr, n], target) -> (thr, est_fdr)


def rank_typing(typing_score, fraction=0.25):
    """Boolean [N] inferred-inhibitory mask by the shipped rank rule.
    ``typing_score``: larger = more inhibitory (for a signed score matrix S the
    natural choice is ``-S.sum(1)``; the GLM uses its own lag-1..2 score)."""
    return np.asarray(infer_inhibitory(None, score=typing_score, typing="rank",
                                       fraction=fraction), bool)


def fdr_pred_untyped(S_abs, null_abs, target):
    """Untyped |S| protocol: threshold |S| over all off-diagonal pairs.
    ``null_abs``: list/array of [N, N] surrogate |S| matrices.  Returns
    (pred [N, N] bool, info dict).  Strict ``>`` as in the shipped scripts."""
    n = S_abs.shape[0]
    off = ~np.eye(n, dtype=bool)
    obs = S_abs[off]
    null = np.stack([np.asarray(x)[off] for x in null_abs])
    thr, est = fdr_threshold(obs, null, target)
    pred = (S_abs > thr) & off if np.isfinite(thr) else np.zeros((n, n), bool)
    return pred, dict(threshold=float(thr), est_fdr=float(est), target=float(target),
                      n_pred=int(pred.sum()))


def fdr_pred_typed(S, null_S, inh_mask, target):
    """Typed protocol (chtc/analysis_one.py): +S over pairs with a non-inhibitory
    presynaptic neuron, -S over pairs with an inhibitory one, each thresholded
    at ``target`` against the matching surrogate scores."""
    n = S.shape[0]
    off = ~np.eye(n, dtype=bool)
    inh = np.asarray(inh_mask, bool)
    pred = np.zeros((n, n), bool)
    info = {}
    for name, sign, rows in (("exc", +1.0, ~inh), ("inh", -1.0, inh)):
        valid = off & rows[:, None]
        if not valid.any():
            info[name] = dict(threshold=float("inf"), est_fdr=float("nan"), n_pred=0)
            continue
        obs = (sign * S)[valid]
        null = np.stack([(sign * np.asarray(x))[valid] for x in null_S])
        thr, est = fdr_threshold(obs, null, target)
        layer = ((sign * S) > thr) & valid if np.isfinite(thr) else np.zeros((n, n), bool)
        pred |= layer
        info[name] = dict(threshold=float(thr), est_fdr=float(est), n_pred=int(layer.sum()))
    info["n_inh_typed"] = int(inh.sum())
    info["target"] = float(target)
    info["n_pred"] = int(pred.sum())
    return pred, info


def bonferroni_alpha(alpha, n_tests):
    return float(alpha) / max(int(n_tests), 1)
