"""Classical cross-correlogram with peak / trough detection against a flank
baseline and a Poisson significance test (Perkel, Gerstein & Moore 1967).

Source
------
Perkel DH, Gerstein GL, Moore GP (1967) Neuronal spike trains and stochastic
point processes. II. Simultaneous spike trains. Biophys J 7:419-440.
The paper introduces the cross-correlation histogram (cross-correlogram) of
two simultaneously recorded spike trains, reads a short-latency PEAK as
evidence of an excitatory influence of the reference train on the target
train, a short-latency TROUGH as inhibitory evidence, and assesses either
against the count expected if the trains were independent, treating the
per-bin count as a Poisson variate.

Orientation (the package invariant, see ``accumulate.py``)
----------------------------------------------------------
``S[i, j]`` is evidence for the directed edge ``i -> j`` (row = presynaptic,
column = postsynaptic).  The shared 1 ms lag products are
``C = spikes.at(1.0).lag_products(50)`` with ``C[l][i, j]`` = number of bins
in which neuron ``i`` fired ``l`` bins BEFORE neuron ``j``.  So the
correlogram of the ordered pair (i, j) at lag ``+l`` is ``C[l][i, j]`` and at
lag ``-l`` it is the transpose, ``C[l][j, i]``.  A peak at a POSITIVE lag
(i leads j) supports ``i -> j``.

Statistics (all constants in ``PARAMS``)
-----------------------------------------
* baseline ``b[i, j]``: mean count over the FLANK lags ``|l|`` in
  ``flank_lags = (30, 50)`` on BOTH sides of the correlogram, i.e. the 21
  positive-lag bins ``C[30..50][i, j]`` and the 21 negative-lag bins
  ``C[30..50][j, i]`` -- 42 bins per pair.
* causal window: lags ``causal_lags = (1, 15)`` ms (the true conduction
  delays are 1.5-13.5 ms; +1.5 ms margin).
* ``z_peak   = (max_{l in causal} C[l][i, j] - b) / sqrt(max(b, b_floor))``
  ``z_trough = (min_{l in causal} C[l][i, j] - b) / sqrt(max(b, b_floor))``
  ``S[i, j]  = z_peak  if  z_peak >= -z_trough  else  z_trough``
  -- the larger-magnitude deviation from baseline, sign kept: positive =
  excitatory (peak), negative = inhibitory (trough).  ``b_floor = 0.5``
  counts guards the Poisson standard error for pairs whose flanks are nearly
  empty (at ~0.3 Hz per neuron and 1 ms bins, a quarter of the pairs have
  ``b < 0.5`` on 50 recordings); without it a single coincidence over an
  empty flank would be an unbounded z.
* ``lag_profile``: ``E[l] = (C[l] - b) / sqrt(max(b, b_floor))`` for the
  causal lags, signed, diagonal 0 -- ``argmax`` over lags is the excitatory
  peak latency used by the delay gate.
* ``label_free`` (the method's own operating point): a two-sided Poisson
  test of the extreme causal bin against the flank baseline,
  ``alpha = 0.001`` Bonferroni-corrected over the 15 causal bins.  A pair is
  excitatory if ``P(Poisson(mu) >= c_max) < alpha / 15`` and inhibitory if
  ``P(Poisson(mu) <= c_min) < alpha / 15`` with ``mu = max(b, b_floor)``.
  ``untyped`` = either test significant; ``typed`` = the same mask (its sign
  already types the edge); ``typing_inh = rank_typing(-S.sum(1))`` is
  recorded for the neuron-typing row only.

Known behaviour (expected, reported, not corrected)
---------------------------------------------------
The 1967 method conditions on nothing: in a densely interconnected network
common input and polysynaptic chains produce broad, positive-lag-heavy
correlogram peaks that a distant flank baseline does not remove, so the
method makes many false inferences (Perkel et al. already noted that a peak
cannot by itself distinguish a direct connection from shared input).  Slow
network co-modulation (bursts) also inflates the flank baseline relative to
the rate product, which is precisely why the flank rather than the rate
product is used here (see Deviations).  We expect a high excitatory AUC with
a low AP and a label-free operating point with many false positives on the
dense sweep networks; this is the reference behaviour the table records.
At the low rates of these datasets the inhibitory branch of the Poisson
test can only fire when ``b`` exceeds ~10 counts (``P(X = 0 | b) < alpha/15``),
which few pairs reach, so inhibitory edges are found almost exclusively by
the signed score, not by the label-free test.

Deviations from the published method
------------------------------------
1. Baseline: the paper estimates the count expected under independence from
   the firing rates (``n_i n_j Delta / T``, a flat baseline).  Here the mean
   of the +/-30..50 ms flanks is used (the common later practice, e.g.
   Csicsvari et al. 1998; Fujisawa et al. 2008), because it absorbs part of
   the slow co-modulation that the rate product ignores.  On these data the
   flank baseline is ~3-5x the rate product (median 3.4x, count-weighted 5x,
   measured on sweep_c50_seed01 normal, n=50).
2. ``b_floor``: the Poisson standard error and the Poisson test mean are
   floored at 0.5 counts (not in the paper; a guard for near-empty flanks).
   For the test this is conservative: a pair with ``b < 0.5`` needs at
   least 6 coincidences in one causal bin to be declared excitatory.
3. Significance: the paper argues significance qualitatively / by eye with
   Poisson confidence bands; here the extreme causal bin is tested with a
   Bonferroni-corrected two-sided Poisson test (``alpha / n_causal``) so the
   rule is explicit and reproducible.
4. Discretisation: the correlogram is the binned 1 ms lag product of the
   shared accumulator (binary bins, ``T = int(duration / bin_ms)`` per
   recording, no count bridges two recordings).  Counts at lag ``l`` use
   ``T_seg - l`` bin pairs per recording (< 0.1 % fewer than lag 0 for 60 s
   recordings); they are NOT rescaled by the valid-pair count.
5. Candidate set: every ordered off-diagonal pair is scored (the paper
   inspects individual pairs); the diagonal (autocorrelogram) is set to 0.

Nothing in this module reads ground truth or any ``network_*.npz``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from baselines import common as cm
from baselines.labelfree import bonferroni_alpha, rank_typing

NAME = "ccg_classical"
TIER = 1
DIRECTED = True
CITATION = ("Perkel DH, Gerstein GL, Moore GP (1967) Neuronal spike trains and stochastic "
            "point processes. II. Simultaneous spike trains. Biophys J 7:419-440.")
PARAMS = dict(
    bin_ms=cm.FINE_BIN_MS,        # 1 ms correlogram bins (the shared fine pass)
    max_lag=cm.FINE_MAX_LAG,      # lags 0..50 of the shared lag products
    causal_lags=(1, 15),          # inclusive lag window (bins) searched for the peak / trough
    flank_lags=(30, 50),          # inclusive flank (bins), taken on BOTH sides -> 42 bins
    b_floor=0.5,                  # counts; floor of the Poisson SE and of the test mean
    alpha=0.001,                  # two-sided Poisson test level, Bonferroni / n_causal
    typing_fraction=cm.TYPING_FRACTION,  # rank rule for the recorded E/I typing
)


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


def _lag_range(pair):
    lo, hi = int(pair[0]), int(pair[1])
    if lo < 1 or hi < lo:
        raise ValueError("bad lag window %r" % (pair,))
    return lo, hi


def _stats(spikes, n_neurons, p):
    """Shared sufficient statistics for every entry point.

    Returns ``(sd, C, b, denom)``: the 1 ms pass, its lag-product stack
    (float32, [max_lag + 1, N, N], NOT upcast), the flank baseline ``b``
    [N, N] float64 and ``denom = sqrt(max(b, b_floor))``.
    """
    sd = spikes.at(p["bin_ms"])
    if sd.n_neurons != int(n_neurons):
        raise ValueError("n_neurons %d != spike matrix rows %d" % (n_neurons, sd.n_neurons))
    c_lo, c_hi = _lag_range(p["causal_lags"])
    f_lo, f_hi = _lag_range(p["flank_lags"])
    if c_hi >= f_lo:
        raise ValueError("causal window %r overlaps flank %r" % (p["causal_lags"], p["flank_lags"]))
    C = sd.lag_products(max(int(p["max_lag"]), f_hi, c_hi))
    # positive-lag flank (i before j), summed lag-wise in float64 without
    # copying the float32 stack; the negative-lag flank is its transpose.
    F_pos = C[f_lo:f_hi + 1].sum(0, dtype=np.float64)
    n_flank = 2 * (f_hi - f_lo + 1)
    b = (F_pos + F_pos.T) / float(n_flank)
    denom = np.sqrt(np.maximum(b, float(p["b_floor"])))
    return sd, C, b, denom


def _extremes(C, p):
    """Max and min causal-window count per ordered pair, float64 [N, N]."""
    c_lo, c_hi = _lag_range(p["causal_lags"])
    block = C[c_lo:c_hi + 1]                       # view of the float32 stack
    return block.max(0).astype(np.float64), block.min(0).astype(np.float64)


def score(spikes, n_neurons, **params):
    """Signed z of the larger-magnitude causal deviation from the flank baseline."""
    p = _p(params)
    _, C, b, denom = _stats(spikes, n_neurons, p)
    c_max, c_min = _extremes(C, p)
    z_peak = (c_max - b) / denom
    z_trough = (c_min - b) / denom
    S = np.where(z_peak >= -z_trough, z_peak, z_trough)
    np.fill_diagonal(S, 0.0)
    return S


def lag_profile(spikes, n_neurons, **params):
    """Signed z per causal lag: ``E[k][i, j] = (C[l_k][i, j] - b) / denom``."""
    p = _p(params)
    sd, C, b, denom = _stats(spikes, n_neurons, p)
    c_lo, c_hi = _lag_range(p["causal_lags"])
    lags = np.arange(c_lo, c_hi + 1)
    E = np.empty((len(lags), sd.n_neurons, sd.n_neurons), np.float64)
    for k, l in enumerate(lags):
        E[k] = (C[l] - b) / denom
        np.fill_diagonal(E[k], 0.0)
    return lags.astype(np.float64) * p["bin_ms"], E


def label_free(spikes, n_neurons, S, **params):
    """Two-sided Poisson test of the extreme causal bin against the flank
    baseline, ``alpha`` Bonferroni-corrected over the causal bins."""
    p = _p(params)
    sd, C, b, _ = _stats(spikes, n_neurons, p)
    c_max, c_min = _extremes(C, p)
    c_lo, c_hi = _lag_range(p["causal_lags"])
    f_lo, f_hi = _lag_range(p["flank_lags"])
    n_tests = c_hi - c_lo + 1
    a = bonferroni_alpha(p["alpha"], n_tests)
    mu = np.maximum(b, float(p["b_floor"]))
    off = cm.offdiag(sd.n_neurons)
    # P(X >= c_max) = sf(c_max - 1);  P(X <= c_min) = cdf(c_min)
    p_exc = poisson.sf(c_max - 1.0, mu)
    p_inh = poisson.cdf(c_min, mu)
    exc = (p_exc < a) & off
    inh = (p_inh < a) & off
    pred = exc | inh
    sig_counts = c_max[exc]
    info = dict(alpha=float(p["alpha"]), alpha_bonferroni=float(a), n_tests=int(n_tests),
                b_floor=float(p["b_floor"]),
                n_exc_sig=int(exc.sum()), n_inh_sig=int(inh.sum()),
                n_both_sig=int((exc & inh).sum()),
                min_sig_peak_count=float(sig_counts.min()) if sig_counts.size else float("nan"),
                n_pred=int(pred.sum()))
    typed_info = dict(info)
    typing_inh = rank_typing(-np.asarray(S, np.float64).sum(1), p["typing_fraction"])
    typed_info["n_inh_typed"] = int(typing_inh.sum())
    return dict(untyped=(pred, info), typed=(pred.copy(), typed_info),
                typing_inh=typing_inh,
                rule="Poisson flank test, alpha %g Bonferroni/%d, flank +/-%d..%d ms, b_floor %g"
                     % (p["alpha"], n_tests, f_lo, f_hi, p["b_floor"]))
