"""Zero-lag Pearson correlation at 5 ms -- the naive floor.

Undirected and symmetric by construction (``S == S.T``): it can say two
neurons co-fire, never which one drives the other, so on the directed
ground truth it is scored as an undirected method (every true edge i -> j
also credits j -> i).  Same-bin coincidences at 5 ms do capture part of the
1.5-13.5 ms conduction-delay range, which is why it is not at chance.
Expected to fail badly on a bursting network -- that is the point.

Source: Pearson product-moment correlation of binned spike trains.
"""
from __future__ import annotations

import numpy as np

from baselines import accumulate as acc
from baselines import common as cm

NAME = "pearson0"
TIER = 1
DIRECTED = False
CITATION = "Zero-lag Pearson correlation coefficient of binned spike trains (no dedicated source)."
PARAMS = dict(bin_ms=cm.GLM_BIN_MS)


def score(spikes, n_neurons, **params):
    p = dict(PARAMS)
    p.update(params)
    sd = spikes.at(p["bin_ms"])
    C0 = sd.lag_products(0)[0]
    S = acc.pearson_from_counts(C0, sd.n_spikes, sd.T)
    np.fill_diagonal(S, 0.0)
    return S
