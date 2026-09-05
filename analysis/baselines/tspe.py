"""Total Spiking Probability Edges (TSPE): an edge-filtered normalised
cross-correlogram read out at its largest absolute value over delays.

Source
------
De Blasi S, Ciba M, Bahmer A, Thielemann C (2019).  Total spiking probability
edges: A cross-correlation based method for effective connectivity estimation
of cortical spiking neurons.  J Neurosci Methods 312:169-181.
doi:10.1016/j.jneumeth.2018.11.013.
Reference code: the MATLAB original ``TSPE.m`` (github.com/biomemsLAB/TSPE)
and the Python port in Elephant
(``elephant.functional_connectivity_src.total_spiking_probability_edges``).
Re-implemented here in numpy/scipy from those sources; neither is imported.

The method was built for multi-electrode-array cultures: it was validated on
simulated networks of ~1000 Izhikevich neurons with low connection density
and on cultured cortical neurons, at 1 ms bins with delays up to 25 bins.
Its published defaults are used unchanged (PARAMS); nothing is tuned here.

Algorithm (defaults in PARAMS)
------------------------------
1. Normalised cross-correlation per delay d at ``bin_ms`` = 1 ms (the shared
   fine-timing pass)::

       NCC(d)[i, j] = (C[d][i, j] / T_d - p_i p_j) / (sigma_i sigma_j)

   with ``C[d]`` the shared lag products, ``T_d`` the number of valid
   (t - d, t) bin pairs, ``p`` the spike probability per bin and
   ``sigma = sqrt(p (1 - p))`` (``accumulate.pearson_from_counts``).  The
   delay axis is padded so that every filter fits:
   ``d = -P .. max_delay + P - 1`` with ``P = max(surrounding) + max(crossover)``
   (Elephant: ``delay_times = range(-max_padding, max_delay + max_padding)``).
2. For every (a, b, c) in surrounding x observed x crossover window sizes the
   edge filter ``[-1/a]*a + [0]*c + [2/b]*b + [0]*c + [-1/a]*a`` is convolved
   along the delay axis (mode 'valid') and the result with the running-total
   filter ``ones(b)`` (mode 'full'); the outputs of all combinations are
   summed -> ``tspe[n, i, j]``, n = 0 .. max_delay - 1.
3. ``S[i, j] = tspe[n*, i, j]`` with ``n* = argmax_n |tspe[n, i, j]|``; the
   delay estimate is ``n*`` bins.  Positive = excitatory (a peak in the
   causal window), negative = inhibitory (a trough).  No threshold: the
   paper chose its operating points by ROC against ground truth, so no
   ``label_free`` rule is defined.

ORIENTATION (this package's convention, NOT the MATLAB storage order)
---------------------------------------------------------------------
``S[i, j]`` is evidence for the directed edge i -> j (row = presynaptic,
column = postsynaptic).  The shared products ``C[d][i, j]`` count bins in
which neuron i fired d bins BEFORE neuron j, so ``NCC(d)[i, j]`` pairs i's
spikes with j's spikes d bins LATER and is evidence for i -> j at delay d; a
negative delay is the transpose, ``NCC(-d)[i, j] = NCC(d)[j, i]``.  The
returned ``S`` and lag profile ``E[n, i, j]`` follow that convention.

Delay bookkeeping
-----------------
For one (a, b, c) the NCC window covers delays ``-(a+c) .. max_delay+(a+c)-1``
and the (symmetric) edge filter has length ``2a + 2c + b``, so 'valid' output
k (0-based) places the observed window at delays ``k .. k+b-1`` (flanks at
``k-c-a .. k-c-1`` and ``k+b+c .. k+b+c+a-1``).  The 'full' running total
then sums, at output n, the b placements whose observed window contains
delay n.  Hence output index n IS the delay in bins (n = 0 .. max_delay - 1),
``lags_ms = arange(max_delay) * bin_ms``, and a unit impulse at delay n in
NCC peaks at output n (asserted by ``python tspe.py --selftest``).  Elephant
reports this 0-based index as the delay; the MATLAB original reports
``index - 2`` on its own 1-based, differently padded axis.  The mapping here
is derived from the convolution bookkeeping rather than copied from either.

Deviations from the published method
------------------------------------
* NCC normalisation.  The paper's formula subtracts the means; Elephant omits
  the subtraction and divides by the total bin count with a ddof=1 standard
  deviation.  We use the shared ``pearson_from_counts`` (mean-subtracted,
  per-delay valid-pair count ``T_d``, Bernoulli sigma).  The edge filter sums
  to zero, so the delay-independent ``p_i p_j`` offset cancels; what remains
  is a per-pair rescaling by ``T / T_d = 1 + O(d / T_segment)`` that does not
  move the peak within a pair's delay profile.
* Recording boundaries.  The shared accumulator never lets a correlogram
  count bridge two recordings; the original assumes one continuous recording.
* Evaluation order.  The cascade is linear in NCC, so it is evaluated as ONE
  kernel ``K[max_delay, n_delays]`` obtained by pushing unit impulses through
  the exact Elephant cascade (``oaconvolve`` 'valid' then 'full', summed over
  filter pairs) and applied by a single matrix product.  This equals running
  the cascade on the full ``[n_delays, N, N]`` stack up to floating-point
  rounding (asserted by the self-test) and avoids 2 x len(filter_pairs) FFT
  passes over that stack.
* The optional spike-count normalisation (MATLAB ``flag`` / Elephant
  ``normalize``; off by default in both) is not implemented.
* The diagonal (autocorrelation) is zeroed; every ordered off-diagonal pair
  is scored, there is no shortlist.

Memory: the padded NCC stack is ``(max_delay + 2P) x N x N`` float64
(41 x 926 x 926 = 281 MB) plus the ``max_delay x N x N`` output; the shared
float32 lag products are only sliced, never upcast as a whole.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import oaconvolve

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ANALYSIS not in sys.path:
    sys.path.insert(0, _ANALYSIS)
from baselines import accumulate as acc  # noqa: E402
from baselines import common as cm  # noqa: E402

NAME = "tspe"
TIER = 1
DIRECTED = True
CITATION = ("De Blasi S, Ciba M, Bahmer A, Thielemann C (2019) Total spiking probability edges: "
            "a cross-correlation based method for effective connectivity estimation of cortical "
            "spiking neurons. J Neurosci Methods 312:169-181 (MATLAB github.com/biomemsLAB/TSPE; "
            "Python port in Elephant).")
#: published defaults (TSPE.m / Elephant): 1 ms bins, delays 0..24 bins,
#: surrounding (a), observed (b) and crossover (c) window sizes in bins
PARAMS = dict(bin_ms=cm.FINE_BIN_MS, max_delay=25,
              surrounding=(3, 4, 5, 6, 7, 8), observed=(2, 3, 4, 5, 6), crossover=(0,))


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


# ------------------------------------------------------------------------- #
# filters (De Blasi 2019 Sec. 2; Elephant generate_edge_filter /
# generate_running_total_filter / generate_filter_pairs)
# ------------------------------------------------------------------------- #
def edge_filter(a, b, c):
    """``[-1/a]*a + [0]*c + [2/b]*b + [0]*c + [-1/a]*a`` (sums to zero)."""
    f = np.zeros(2 * a + 2 * c + b, np.float64)
    f[:a] = -1.0 / a
    f[a + c:a + c + b] = 2.0 / b
    f[a + 2 * c + b:] = -1.0 / a
    return f


def running_total_filter(b):
    return np.ones(b, np.float64)


def filter_pairs(surrounding, observed, crossover):
    return [(int(a), int(b), int(c)) for a in surrounding for b in observed for c in crossover]


def cascade(ncc, pairs, max_padding, max_delay):
    """Elephant's filter loop along axis 0 of ``ncc`` ``[n_delays, ...]`` (any
    trailing shape): per (a, b, c) select the window the filter needs, 'valid'
    edge convolution, 'full' running-total convolution, sum.  Returns
    ``[max_delay, ...]`` float64."""
    ncc = np.asarray(ncc, np.float64)
    trail = (1,) * (ncc.ndim - 1)
    out = np.zeros((max_delay,) + ncc.shape[1:], np.float64)
    for a, b, c in pairs:
        pad = a + c
        win = ncc[max_padding - pad:max_delay + max_padding + pad]
        x1 = oaconvolve(win, edge_filter(a, b, c).reshape((-1,) + trail), mode="valid", axes=0)
        x2 = oaconvolve(x1, running_total_filter(b).reshape((-1,) + trail), mode="full", axes=0)
        out += x2
    return out


def kernel(surrounding, observed, crossover, max_delay):
    """The cascade as one linear map: ``(K [max_delay, n_delays], delays [n_delays])``
    with ``K[:, m]`` the cascade response to a unit impulse at ``delays[m]``
    (bins; ``delays = -P .. max_delay + P - 1``)."""
    max_padding = int(max(surrounding)) + int(max(crossover))
    delays = np.arange(-max_padding, int(max_delay) + max_padding)
    K = cascade(np.eye(len(delays)), filter_pairs(surrounding, observed, crossover),
                max_padding, int(max_delay))
    return K, delays


# ------------------------------------------------------------------------- #
# statistics
# ------------------------------------------------------------------------- #
def _ncc_stack(sd, delays):
    """``NCC[k][i, j]``: correlation of i's spikes with j's spikes ``delays[k]``
    bins later, from the shared lag products (negative delays = transpose)."""
    n = sd.n_neurons
    dmax = int(np.max(np.abs(delays)))
    # the shared 1 ms pass is lags 0..FINE_MAX_LAG; slice it, never recompute
    max_lag = cm.FINE_MAX_LAG if sd.bin_ms == cm.FINE_BIN_MS else dmax
    if dmax > max_lag:
        raise ValueError("tspe needs lags up to %d, shared pass has %d" % (dmax, max_lag))
    C = sd.lag_products(max_lag)                      # float32 [max_lag + 1, N, N]
    T = sd.valid_pairs(range(dmax + 1))
    ncc = np.empty((len(delays), n, n), np.float64)
    for k, d in enumerate(delays):
        d = int(d)
        r = acc.pearson_from_counts(C[abs(d)], sd.n_spikes, T[abs(d)])
        ncc[k] = r if d >= 0 else r.T
    return ncc


def _profile(spikes, params):
    p = _p(params)
    max_delay = int(p["max_delay"])
    sd = spikes.at(p["bin_ms"])
    K, delays = kernel(p["surrounding"], p["observed"], p["crossover"], max_delay)
    ncc = _ncc_stack(sd, delays)
    n = sd.n_neurons
    E = (K @ ncc.reshape(len(delays), n * n)).reshape(max_delay, n, n)
    del ncc
    for k in range(max_delay):
        np.fill_diagonal(E[k], 0.0)
    lags_ms = np.arange(max_delay, dtype=np.float64) * float(p["bin_ms"])
    return lags_ms, E


def readout(E):
    """MATLAB ``[CM, DM]``: ``S`` = tspe at the largest |tspe| over delays,
    ``delay_bins`` = that delay index (0-based = bins)."""
    idx = np.abs(E).argmax(0)
    S = np.take_along_axis(E, idx[None], 0)[0].astype(np.float64)
    np.fill_diagonal(S, 0.0)
    return S, idx


# ------------------------------------------------------------------------- #
# contract
# ------------------------------------------------------------------------- #
def score(spikes, n_neurons, **params):
    _, E = _profile(spikes, params)
    return readout(E)[0]


def lag_profile(spikes, n_neurons, **params):
    return _profile(spikes, params)


# ------------------------------------------------------------------------- #
# self-test (no data): kernel == cascade, impulse bookkeeping, orientation
# ------------------------------------------------------------------------- #
def _selftest():
    import scipy.sparse as sp

    p = PARAMS
    pairs = filter_pairs(p["surrounding"], p["observed"], p["crossover"])
    P = int(max(p["surrounding"])) + int(max(p["crossover"]))
    K, delays = kernel(p["surrounding"], p["observed"], p["crossover"], p["max_delay"])
    assert K.shape == (p["max_delay"], p["max_delay"] + 2 * P) and delays[0] == -P

    # 1. one-kernel evaluation == Elephant's cascade on a random stack
    rng = np.random.default_rng(0)
    ncc = rng.standard_normal((len(delays), 5, 5))
    direct = cascade(ncc, pairs, P, p["max_delay"])
    via_k = (K @ ncc.reshape(len(delays), -1)).reshape(direct.shape)
    assert np.allclose(direct, via_k, atol=1e-9), np.abs(direct - via_k).max()

    # 2. a unit NCC impulse at delay n peaks at output n, for every n
    for n in range(p["max_delay"]):
        assert int(K[:, n + P].argmax()) == n, (n, int(K[:, n + P].argmax()))

    # 3. orientation + sign: 0 -> 1 excitatory at 3 bins, 0 -| 2 inhibitory
    T = 60000
    m0 = rng.random(T) < 0.03
    m1 = rng.random(T) < 0.01
    m1[3:] |= m0[:-3]                       # neuron 1 fires 3 bins AFTER neuron 0
    m2 = rng.random(T) < 0.2
    sup = np.zeros(T, bool)
    for d in range(2, 6):
        sup[d:] |= m0[:-d]
    m2 &= ~sup                              # neuron 2 silenced 2-5 bins after neuron 0
    M = sp.csr_matrix(np.stack([m0, m1, m2]).astype(np.float32))
    sd = cm.SpikeData(M, [0, T], 1.0)

    class _Tiny:
        def at(self, bin_ms):
            assert float(bin_ms) == 1.0
            return sd

    S = score(_Tiny(), 3)
    lags_ms, E = lag_profile(_Tiny(), 3)
    S2, D = readout(E)
    assert np.isfinite(S).all() and np.allclose(S, S2) and np.diag(S).tolist() == [0, 0, 0]
    assert S[0, 1] > 0 and S[0, 1] > S[1, 0], (S[0, 1], S[1, 0])
    assert int(E[:, 0, 1].argmax()) == 3 and int(D[0, 1]) == 3, int(D[0, 1])
    assert S[0, 2] < 0 and S[0, 2] < S[2, 0], (S[0, 2], S[2, 0])
    assert lags_ms[3] == 3.0
    print("tspe self-test OK: S[0,1]=%.3f S[1,0]=%.3f delay=%d bins | S[0,2]=%.3f S[2,0]=%.3f"
          % (S[0, 1], S[1, 0], D[0, 1], S[0, 2], S[2, 0]))
    return True


if __name__ == "__main__":
    _selftest()
