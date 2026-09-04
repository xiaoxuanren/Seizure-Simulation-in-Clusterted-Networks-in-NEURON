"""The shipped lag-resolved ridge GLM (sum4 readout, jitter-null FDR 0.70).

Not a baseline -- the reference row of the table.  It is NOT re-fit by a
second implementation: ``score`` reads the ridge solve of the shared
accumulator, which ``test_accumulate.py`` shows to be bit-identical to
``sparse_glm.fit_B`` and to the shipped ``glm_connectivity_sum4_5ms.npz``.
The label-free operating point reproduces ``analysis/glm_fit_current.py``
(untyped |W| protocol of ``sweep_fdr_calibration.py``; typed protocol of
``chtc/analysis_one.py``) with the same surrogate stream (seed 1, 8 draws).

Source: this repository, ``glm_connectivity.py`` / ``sparse_glm.py``.
Orientation: ``B[lag, pre, post]``; ``W[pre, post] = sum_{lag<=4} B``.
"""
from __future__ import annotations

import numpy as np

from baselines import accumulate as acc
from baselines import common as cm
from baselines.labelfree import fdr_pred_typed, fdr_pred_untyped, rank_typing
from glm_connectivity import typing_score

NAME = "glm_ridge"
TIER = 1
DIRECTED = True
CITATION = ("This repository: sparse lag-resolved ridge GLM, glm_connectivity.py; "
            "jitter null after Fujisawa et al. 2008 / Amarasingham et al. 2012.")
PARAMS = dict(bin_ms=cm.GLM_BIN_MS, max_lag=cm.GLM_MAX_LAG, l2=cm.GLM_L2, ksum=cm.GLM_KSUM,
              jitter_ms=cm.JITTER_MS, n_surr=cm.N_SURR, seed=cm.SEED,
              target_fdr=cm.TARGET_FDR, typing_lags=cm.TYPING_LAGS,
              typing_fraction=cm.TYPING_FRACTION)


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


def score(spikes, n_neurons, **params):
    p = _p(params)
    sd = spikes.at(p["bin_ms"])
    B = sd.ridge_B(p["max_lag"], p["l2"])
    return acc.readout_sum_k(B, p["ksum"]).astype(np.float64)


def lag_profile(spikes, n_neurons, **params):
    p = _p(params)
    sd = spikes.at(p["bin_ms"])
    B = sd.ridge_B(p["max_lag"], p["l2"]).copy()
    for k in range(B.shape[0]):
        np.fill_diagonal(B[k], 0.0)
    lags_ms = (np.arange(p["max_lag"]) + 1) * p["bin_ms"]
    return lags_ms, B


def typing_inhibitory(spikes, n_neurons, **params):
    p = _p(params)
    B = spikes.at(p["bin_ms"]).ridge_B(p["max_lag"], p["l2"])
    return rank_typing(typing_score(B, k=p["typing_lags"]), p["typing_fraction"])


def label_free(spikes, n_neurons, S, **params):
    p = _p(params)
    nulls = []
    for sj in spikes.surrogates(p["bin_ms"], p["jitter_ms"], p["n_surr"], p["seed"]):
        nulls.append(acc.readout_sum_k(sj.ridge_B(p["max_lag"], p["l2"]), p["ksum"]))
        sj.release()
    untyped = fdr_pred_untyped(np.abs(S), [np.abs(w) for w in nulls], p["target_fdr"])
    inh = typing_inhibitory(spikes, n_neurons, **params)
    typed = fdr_pred_typed(S, nulls, inh, p["target_fdr"])
    return dict(untyped=untyped, typed=typed, typing_inh=inh,
                rule="jitter-null FDR %.2f, %d surrogates, +/-%g ms" % (
                    p["target_fdr"], p["n_surr"], p["jitter_ms"]))
