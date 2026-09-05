"""Jitter-corrected cross-correlogram: the shipped spike-jitter null applied to
the RAW correlogram instead of to a GLM fit.

Source
------
Fujisawa S, Amarasingham A, Harrison MT, Buzsaki G (2008) Behavior-dependent
short-term assembly dynamics in the medial prefrontal cortex.  Nat Neurosci
11:823-833 (jitter-corrected CCG for monosynaptic E/I detection).
Amarasingham A, Harrison MT, Hatsopoulos NG, Geman S (2012) Conditional
modeling and the jitter method of spike resampling.  J Neurophysiol
107:517-531 (the conditional-inference justification of the jitter null).

Why this baseline exists
------------------------
The shipped ridge GLM (``glm_ridge``) and this method read the SAME sufficient
statistic and are tested against the SAME null:

* statistic  -- the 5 ms lag products ``C[l] = shift(M, l) M^T``, l = 1..6.
  In the GLM these are the right-hand side ``F X^T`` of the ridge normal
  equations (``accumulate.gram``); the ``sum4`` readout sums the whitened
  coefficients over lags 1..4.  Here the raw counts are summed over the same
  lags 1..4 (5-20 ms) with NO whitening: no Gram matrix, no conditioning on
  the other neurons, no ridge.
* null       -- ``Session.surrogates(5.0)``: 8 spike-jitter surrogates,
  +/-25 ms (= +/-5 bins), ``default_rng(1)``, produced by ``sparse_glm.jitter``
  in the same generator order as the shipped pipeline, so the surrogate
  matrices are bit-identical to those behind the GLM's jitter null.

The difference between the ``glm_ridge`` row and this row is therefore what
the joint (whitened) fit contributes; the difference between this row and
``ccg_classical`` / ``pearson_lag`` is what the jitter null contributes.

Statistic
---------
For every ordered pair (i, j) with observed count ``obs = sum_{l=1..4} C[l][i, j]``
and surrogate counts ``nl_k`` (k = 1..8):

    z[i, j] = (obs - mu) / max(sigma, sqrt(mu), sigma_floor)
    mu      = mean_k nl_k          sigma = std_k nl_k   (ddof = 0)

``S = z`` with the diagonal set to 0.  Positive = excess short-latency
coincidences beyond what the jitter null explains (excitatory evidence);
negative = deficit (inhibitory evidence).  The sign therefore comes from
peak-versus-trough of the causal window 5-20 ms relative to the jitter
expectation, as in Fujisawa et al. (2008).

Eight surrogate draws give a noisy standard deviation (a pair whose 8 null
counts happen to coincide would get sigma = 0), so the denominator is floored
by the Poisson dispersion ``sqrt(mu)`` -- the dispersion a jitter null of
independent coincidences would have -- and by ``sigma_floor`` for pairs with
essentially no coincidences (``mu`` ~ 0), which keeps ``S`` finite everywhere.

Orientation (package invariant, see ``accumulate.py``)
-------------------------------------------------------
``S[i, j]`` is evidence for the directed edge i -> j: row = presynaptic,
column = postsynaptic.  ``C[l][i, j]`` counts the bins in which neuron i
fired l bins BEFORE neuron j; a negative lag is the transpose,
``ccg_ij(-l) = C[l][j, i]``.  Nothing in this module transposes.

Lag profile
-----------
``lag_profile`` returns the per-lag z, ``E[l] = (C[l] - mu_l) / max(sd_l,
sqrt(mu_l), sigma_floor)`` for l = 1..6 (5..30 ms), with ``mu_l`` / ``sd_l``
the per-lag mean / sd over the same 8 surrogates.  Observed and surrogate
counts are gathered in one pass so every surrogate is processed once.

Label-free operating point
--------------------------
The shipped jitter-null FDR rule (``labelfree.fdr_pred_untyped`` /
``fdr_pred_typed``, i.e. ``chtc/analysis_one.fdr_threshold``) on the z
statistic.  The null z's are leave-one-out: surrogate k is z-scored against
the mean / sd of the OTHER 7 surrogates with the same denominator floor,
``z_k = (nl_k - mean_{j!=k} nl_j) / max(std_{j!=k} nl_j, sqrt(mean_{j!=k} nl_j),
sigma_floor)``, so each null draw is treated like the observation (scored
against draws it was not part of).  Untyped: ``|z|`` over all off-diagonal
pairs at target FDR 0.70.  Typed: E/I typing by the shipped rank rule on
``-z.sum(1)`` (fraction 0.25), then +z over pairs with a non-inhibitory
presynaptic neuron and -z over pairs with an inhibitory one, each at FDR 0.70.

Deviations from the published method
------------------------------------
1. Bin width, jitter window and surrogate count are those of the shipped GLM
   null (5 ms bins, +/-25 ms, 8 surrogates, seed 1) rather than Fujisawa et
   al.'s 1 ms bins, +/-5 ms jitter and ~1000 surrogates.  This is deliberate:
   the row is meant to be the un-whitened twin of the GLM row, not a re-tuned
   correlogram method.  (A 25 ms window destroys structure up to ~25 ms, so
   slow comodulation on that scale is NOT part of the null; the shipped GLM
   has the same property.)
2. Fujisawa et al. tested the observed correlogram against pointwise and
   global (simultaneous) acceptance bands taken from surrogate quantiles.
   Eight draws cannot support a 99% quantile, so the statistic is a
   surrogate-mean / surrogate-sd z-score with a Poisson floor on the
   denominator (``sigma_floor``, ``sqrt(mu)``), and the threshold is the
   shipped jitter-null FDR rule instead of a significance band.
3. The evidence is summed over lags 1..4 (5-20 ms) -- the raw analogue of the
   GLM's ``sum4`` readout -- rather than tested bin by bin at 1 ms in a
   1-4 ms window.  The bin-by-bin z is exposed through ``lag_profile``.
4. The surrogates come from ``sparse_glm.jitter``: every spike is displaced
   by an independent uniform draw in [-5, +5] bins and clipped to its own
   recording (the per-spike "basic" jitter of Fujisawa et al. 2008).
   Amarasingham et al. (2012) recommend INTERVAL jitter (uniform resampling
   within fixed, non-overlapping windows) for exact conditional guarantees;
   it is not used here so that the null stays identical to the GLM's.  Bins
   are binary, so two spikes jittered into the same bin merge (inherited from
   the shipped jitter).
5. No refractory / burst handling, no rate-matching beyond the jitter, and
   no candidate shortlist: every ordered off-diagonal pair is scored.

Published critique
------------------
Kobayashi R, Kurita S, Kurth A, Kitano K, Mizuseki K, Diesmann M, Richmond
BJ, Shinomoto S (2019) Reconstructing neuronal circuitry from parallel spike
trains.  Nat Commun 10:4468: on simulated networks with ground truth the
jittering method produced few false positives but missed many connections,
inhibitory ones in particular, compared with their GLMCC.  The
``ccg_classical`` and ``glm_ridge`` rows bracket that trade-off here.

Every constant is in ``PARAMS`` (the runner records them).  Nothing here
reads ground truth.
"""
from __future__ import annotations

import numpy as np

from baselines import common as cm
from baselines.labelfree import fdr_pred_typed, fdr_pred_untyped, rank_typing

NAME = "ccg_jitter"
TIER = 1
DIRECTED = True
CITATION = ("Fujisawa S, Amarasingham A, Harrison MT, Buzsaki G (2008) Nat Neurosci 11:823-833; "
            "Amarasingham A, Harrison MT, Hatsopoulos NG, Geman S (2012) J Neurophysiol 107:517-531 "
            "(jitter-corrected cross-correlogram; here with the shipped GLM's surrogates).")
PARAMS = dict(
    bin_ms=cm.GLM_BIN_MS,            # 5.0  -- the GLM pass
    ksum=cm.GLM_KSUM,                # 4    -- lags 1..4 summed (5-20 ms), the raw sum4
    max_lag=cm.GLM_MAX_LAG,          # 6    -- lag products computed 1..6 (lag profile)
    jitter_ms=cm.JITTER_MS,          # 25.0 -- +/- jitter half-width (= 5 bins)
    n_surr=cm.N_SURR,                # 8    -- surrogate draws
    seed=cm.SEED,                    # 1    -- default_rng seed of the surrogate stream
    sigma_floor=0.5,                 # absolute floor on the null sd (counts)
    target_fdr=cm.TARGET_FDR,        # 0.70 -- shipped jitter-null FDR target
    typing_fraction=cm.TYPING_FRACTION,  # 0.25 -- rank rule E/I typing
)

#: single-entry cache of the surrogate pass, so ``score`` -> ``label_free`` in
#: one runner process processes the 8 surrogates once.  Keyed on the dataset
#: identity (path, recordings, arm, padding) AND the Session object id.
_CACHE = {"key": None, "stats": None}


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


def _cache_key(spikes, p):
    sdir = getattr(spikes, "session_dir", None)
    if sdir is None:
        return None
    return (id(spikes), sdir, getattr(spikes, "n_recordings", None), getattr(spikes, "arm", None),
            getattr(spikes, "pad_ms", None), float(p["bin_ms"]), int(p["ksum"]), int(p["max_lag"]),
            float(p["jitter_ms"]), int(p["n_surr"]), int(p["seed"]))


def _stats(spikes, p):
    """One pass over the observed lag products and the shared surrogates.

    Returns dict(obs_lag [L, N, N], obs_sum [N, N], lag_sum [L, N, N],
    lag_sq [L, N, N], null_sum [n_surr, N, N]) -- all float64, lags 1..L in
    position 0..L-1 -- where ``*_sum`` are the ``ksum``-lag sums and
    ``lag_sum`` / ``lag_sq`` are running sums over surrogates per lag.
    """
    key = _cache_key(spikes, p)
    if key is not None and _CACHE["key"] == key:
        return _CACHE["stats"]
    L, k = int(p["max_lag"]), int(p["ksum"])
    n_surr = int(p["n_surr"])
    if not 1 <= k <= L:
        raise ValueError("ksum must satisfy 1 <= ksum <= max_lag (got %d, %d)" % (k, L))
    if n_surr < 3:
        raise ValueError("n_surr must be >= 3 for the leave-one-out null (got %d)" % n_surr)

    sd = spikes.at(p["bin_ms"])
    n = sd.n_neurons
    C = sd.lag_products(L)                                   # float32 [L+1, N, N] (shared)
    obs_lag = np.empty((L, n, n), np.float64)
    for l in range(1, L + 1):                                # one lag at a time: no f64 stack copy
        obs_lag[l - 1] = C[l]
    obs_sum = obs_lag[:k].sum(0)

    lag_sum = np.zeros((L, n, n), np.float64)
    lag_sq = np.zeros((L, n, n), np.float64)
    null_sum = np.empty((n_surr, n, n), np.float64)
    surr = spikes.surrogates(p["bin_ms"], p["jitter_ms"], n_surr, p["seed"])
    for s, sj in enumerate(surr):
        Cs = sj.lag_products(L)                              # float32 [L+1, N, N]
        acc_k = np.zeros((n, n), np.float64)
        for l in range(1, L + 1):
            c = np.asarray(Cs[l], np.float64)
            lag_sum[l - 1] += c
            lag_sq[l - 1] += c * c
            if l <= k:
                acc_k += c
        null_sum[s] = acc_k
        sj.release()                                         # drop the surrogate's dense products
    stats = dict(obs_lag=obs_lag, obs_sum=obs_sum, lag_sum=lag_sum, lag_sq=lag_sq,
                 null_sum=null_sum)
    if key is not None:
        _CACHE["key"], _CACHE["stats"] = key, stats
    return stats


def _z(obs, mu, sigma, floor):
    """``(obs - mu) / max(sigma, sqrt(mu), floor)`` with a zero diagonal."""
    denom = np.maximum(np.maximum(sigma, np.sqrt(np.maximum(mu, 0.0))), float(floor))
    z = (obs - mu) / denom
    np.fill_diagonal(z, 0.0)
    return z


def score(spikes, n_neurons, **params):
    p = _p(params)
    st = _stats(spikes, p)
    mu = st["null_sum"].mean(0)
    sigma = st["null_sum"].std(0)                            # ddof = 0
    return _z(st["obs_sum"], mu, sigma, p["sigma_floor"])


def lag_profile(spikes, n_neurons, **params):
    p = _p(params)
    st = _stats(spikes, p)
    L, n_surr = int(p["max_lag"]), int(p["n_surr"])
    n = st["obs_sum"].shape[0]
    E = np.empty((L, n, n), np.float64)
    for l in range(L):
        mu = st["lag_sum"][l] / n_surr
        var = np.maximum(st["lag_sq"][l] / n_surr - mu * mu, 0.0)
        E[l] = _z(st["obs_lag"][l], mu, np.sqrt(var), p["sigma_floor"])
    lags_ms = (np.arange(L) + 1) * float(p["bin_ms"])
    return lags_ms, E


def _loo_null_z(null_sum, floor):
    """Leave-one-out null z's: surrogate k against the mean / sd of the others."""
    n_surr = null_sum.shape[0]
    tot = null_sum.sum(0)
    tot_sq = (null_sum * null_sum).sum(0)
    out = []
    for k in range(n_surr):
        nk = null_sum[k]
        mu_o = (tot - nk) / (n_surr - 1)
        var_o = np.maximum((tot_sq - nk * nk) / (n_surr - 1) - mu_o * mu_o, 0.0)
        out.append(_z(nk, mu_o, np.sqrt(var_o), floor))
    return out


def label_free(spikes, n_neurons, S, **params):
    p = _p(params)
    st = _stats(spikes, p)
    S = np.asarray(S, np.float64)
    null_z = _loo_null_z(st["null_sum"], p["sigma_floor"])   # 8 x [N, N], live only here
    untyped = fdr_pred_untyped(np.abs(S), [np.abs(z) for z in null_z], p["target_fdr"])
    inh = rank_typing(-S.sum(1), p["typing_fraction"])
    typed = fdr_pred_typed(S, null_z, inh, p["target_fdr"])
    del null_z
    return dict(untyped=untyped, typed=typed, typing_inh=inh,
                rule="jitter-null FDR %.2f on jitter z, %d surrogates +/-%g ms (shipped null, raw CCG)"
                     % (p["target_fdr"], p["n_surr"], p["jitter_ms"]))
