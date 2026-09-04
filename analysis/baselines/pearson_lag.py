"""Lagged Pearson correlation: max-|r| over lags 1..6 at 5 ms, signed.

Rate normalisation of the SAME lag products the GLM whitens
(``accumulate.pearson_from_counts``): ``r_l[i, j]`` correlates i's spikes with
j's spikes l bins later.  No conditioning on other neurons, so shared drive
is not explained away -- the point of including it.  The score is the r at
the lag of largest |r| (sign kept: positive = excitatory, negative =
inhibitory).  No label-free threshold rule is defined by the method.

Source: textbook cross-correlation coefficient (e.g. Perkel, Gerstein &
Moore 1967, Biophys J 7:419 for the lagged form); no dedicated paper.
"""
from __future__ import annotations

import numpy as np

from baselines import accumulate as acc
from baselines import common as cm

NAME = "pearson_lag"
TIER = 1
DIRECTED = True
CITATION = "Lagged Pearson correlation coefficient of binned spike trains (no dedicated source)."
PARAMS = dict(bin_ms=cm.GLM_BIN_MS, max_lag=cm.GLM_MAX_LAG)


def _profile(spikes, params):
    p = dict(PARAMS)
    p.update(params)
    sd = spikes.at(p["bin_ms"])
    lags = list(range(1, p["max_lag"] + 1))
    C = sd.lag_products(p["max_lag"])
    T_l = sd.valid_pairs(lags)
    E = np.empty((len(lags), sd.n_neurons, sd.n_neurons))
    for k, l in enumerate(lags):
        E[k] = acc.pearson_from_counts(C[l], sd.n_spikes, T_l[k])
        np.fill_diagonal(E[k], 0.0)
    return np.asarray(lags, float) * p["bin_ms"], E


def score(spikes, n_neurons, **params):
    _, E = _profile(spikes, params)
    idx = np.abs(E).argmax(0)
    S = np.take_along_axis(E, idx[None], 0)[0]
    np.fill_diagonal(S, 0.0)
    return S


def lag_profile(spikes, n_neurons, **params):
    return _profile(spikes, params)
