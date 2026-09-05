"""Filtered and Normalised Cross-Correlation Histogram (FNCCH), Pastore et al. 2018.

Source
------
Pastore VP, Massobrio P, Godjoski A, Martinoia S (2018) Identification of
excitatory-inhibitory links and network topology in large-scale neuronal
assemblies from multi-electrode recordings.  PLoS Comput Biol 14(8):e1006381.

Provenance: the method was developed for multi-electrode-array recordings of
dissociated cortical cultures (MEA-60 and the 4096-electrode CMOS
high-density MEA) and validated in silico on random networks of 1000
Izhikevich neurons (1 h recordings, 1 ms bins, correlation window
W = 25 ms).  It is a correlogram method: no conditioning on other neurons,
so shared drive is not explained away -- part of why it is a baseline here.

The statistic
-------------
For an ordered pair (i, j) with spike counts n_i, n_j and raw
cross-correlogram counts C_ij(tau):

    NCCH_ij(tau)  = C_ij(tau) / sqrt(n_i n_j)                 tau in [-W, W]
    FNCCH_ij(tau) = NCCH_ij(tau) - mean_{tau' in [-W, W]} NCCH_ij(tau')

The normalisation removes the firing-rate dependence of the raw count; the
"filter" (subtracting the window mean) removes the slow common component
that the culture's population bursts put under every correlogram.  The
connectivity value is the FNCCH bin of largest magnitude with its sign
preserved: positive = excitatory link, negative = inhibitory link.  The paper
then applies a hard threshold to the distribution of connectivity values:
mu + 2 sigma for excitatory links and mu + 1 sigma (in magnitude) for
inhibitory links.

Orientation convention (the package invariant, see ``accumulate.py``)
--------------------------------------------------------------------
``S[i, j]`` is evidence for the directed edge ``i -> j`` (row = presynaptic,
column = postsynaptic).  The shared lag products are
``C = spikes.at(1.0).lag_products(50)`` with ``C[l][i, j]`` = number of bins
in which neuron i fired ``l`` bins BEFORE neuron j; a negative lag is the
transpose, ``ccg_ij(-l) = C[l][j, i]``.  Hence for the ordered pair (i, j)

    NCCH_ij(+l) = C[l][i, j] / sqrt(n_i n_j)      l = 0..W
    NCCH_ij(-l) = C[l][j, i] / sqrt(n_i n_j)      l = 1..W

and the directed score ``S[i, j]`` is read from the CAUSAL half l = 1..W
(i before j).  ``lag_profile`` returns the signed causal FNCCH per lag so
``argmax`` over lags finds the excitatory peak.

Deviations from the published method
------------------------------------
* Directed read-out.  The paper searches the peak over the full window
  [-W, W] for each electrode pair and reads the direction of the link from
  the sign of the lag at which the peak sits (positive lag: x -> y).  Our
  candidate set is every ORDERED off-diagonal pair, so the equivalent
  read-out is the peak over the causal half l = 1..W of the ordered pair's
  own FNCCH: the paper's peak at lag +tau for the pair {i, j} is exactly our
  causal peak for (i, j), and its peak at -tau is our causal peak for
  (j, i).  The one difference is that the paper keeps at most one direction
  per unordered pair (the larger), whereas we score both directions
  independently and let the threshold decide -- this can only add candidate
  links, never move them.  The zero-lag bin carries no direction and is
  excluded from the peak search (it is still part of the window mean).
* Binned counts, recording boundaries.  The correlograms come from the
  shared binary-binned accumulator (a 1 ms bin with more than one spike
  counts once; lagged pairs never bridge two recordings), and n_i is the
  number of occupied bins over the concatenated recordings.  The paper
  works on time stamps of continuous 1 h recordings.
* Threshold statistics.  mu/sigma for the excitatory threshold are the mean
  and standard deviation of the POSITIVE connectivity values over all
  off-diagonal ordered pairs, and for the inhibitory threshold of the
  magnitudes of the NEGATIVE values; a link is excitatory if
  S > mu_pos + k_exc sigma_pos and inhibitory if
  S < -(mu_neg + k_inh sigma_neg).  The paper describes the two thresholds
  on "the distribution of connectivity values" without spelling out the
  sign split; treating the excitatory and inhibitory populations
  separately is the reading under which mu + 2 sigma and mu + 1 sigma are
  both meaningful.  No constant is tuned against labels.
* E/I typing of neurons is not part of the paper (it types links); the
  ``typing_inh`` reported for the typed read-out is the package's shared
  rank rule on ``-S.sum(1)``.

Memory: the shared 1 ms stack ``[51, N, N]`` float32 is sliced, never
upcast as a whole; per-chunk float64 work is ``[W, chunk_rows, N]``.
"""
from __future__ import annotations

import numpy as np

from baselines import common as cm
from baselines.labelfree import rank_typing

NAME = "fncch"
TIER = 1
DIRECTED = True
CITATION = ("Pastore VP, Massobrio P, Godjoski A, Martinoia S (2018) Identification of "
            "excitatory-inhibitory links and network topology in large-scale neuronal "
            "assemblies from multi-electrode recordings. PLoS Comput Biol 14(8):e1006381.")
PARAMS = dict(
    bin_ms=cm.FINE_BIN_MS,   # 1 ms bins (the paper's in silico setting)
    window_ms=25.0,          # correlation window W: tau in [-W, W]
    k_exc=2.0,               # excitatory hard threshold  mu_pos + k_exc sigma_pos
    k_inh=1.0,               # inhibitory hard threshold  mu_neg + k_inh sigma_neg (magnitude)
    typing_fraction=cm.TYPING_FRACTION,  # shared rank rule for neuron E/I typing
    chunk_rows=128,          # presynaptic rows per dense float64 chunk (memory bound only)
)


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


def _setup(spikes, p):
    """Shared lag products, the 1/sqrt(n_i n_j) normaliser and the window
    mean of the NCCH over tau in [-W, W] for every ordered pair."""
    sd = spikes.at(p["bin_ms"])
    W = int(round(p["window_ms"] / p["bin_ms"]))
    if W < 1:
        raise ValueError("window_ms must cover at least one bin")
    # request the shared fine-timing stack so every 1 ms method reuses one cache
    C = sd.lag_products(max(W, cm.FINE_MAX_LAG))
    n = np.asarray(sd.n_spikes, np.float64)
    inv = np.where(n > 0, 1.0 / np.sqrt(np.where(n > 0, n, 1.0)), 0.0)
    inv_norm = np.outer(inv, inv)                       # [N, N] 1/sqrt(n_i n_j), 0 if silent
    N = inv_norm.shape[0]
    tot = np.zeros((N, N), np.float64)                  # sum_{l=0..W} C[l][i, j]
    for l in range(W + 1):
        tot += C[l]
    # window [-W, W]: positive lags = tot, negative lags = tot.T minus the
    # zero-lag matrix (C[0] = M M^T is symmetric and must be counted once)
    mean_ncch = (tot + tot.T - np.asarray(C[0], np.float64)) * inv_norm / float(2 * W + 1)
    return sd, W, C, inv_norm, mean_ncch


def _causal_chunks(C, W, inv_norm, mean_ncch, chunk_rows):
    """Yield ``(a, b, F)`` with ``F[l - 1, i - a, j] = FNCCH_ij(+l)`` for
    l = 1..W over presynaptic rows a..b-1 (float64, diagonal zeroed)."""
    N = inv_norm.shape[0]
    step = max(int(chunk_rows), 1)
    for a in range(0, N, step):
        b = min(a + step, N)
        F = np.asarray(C[1:W + 1, a:b, :], np.float64)   # slice, then upcast the chunk
        F *= inv_norm[a:b][None]
        F -= mean_ncch[a:b][None]
        rows = np.arange(a, b)
        F[:, rows - a, rows] = 0.0
        yield a, b, F


def score(spikes, n_neurons, **params):
    """Signed FNCCH peak over the causal lags 1..W, ``[pre, post]``."""
    p = _p(params)
    sd, W, C, inv_norm, mean_ncch = _setup(spikes, p)
    N = sd.n_neurons
    S = np.zeros((N, N), np.float64)
    for a, b, F in _causal_chunks(C, W, inv_norm, mean_ncch, p["chunk_rows"]):
        idx = np.abs(F).argmax(0)                        # [nb, N] lag of largest |FNCCH|
        S[a:b] = np.take_along_axis(F, idx[None], 0)[0]
    np.fill_diagonal(S, 0.0)
    return S


def lag_profile(spikes, n_neurons, **params):
    """``(lags_ms [W], E [W, N, N])``: signed causal FNCCH per lag."""
    p = _p(params)
    sd, W, C, inv_norm, mean_ncch = _setup(spikes, p)
    N = sd.n_neurons
    E = np.empty((W, N, N), np.float64)
    for a, b, F in _causal_chunks(C, W, inv_norm, mean_ncch, p["chunk_rows"]):
        E[:, a:b, :] = F
    lags_ms = np.arange(1, W + 1, dtype=np.float64) * p["bin_ms"]
    return lags_ms, E


def _hard_threshold(S, k_exc, k_inh):
    """The paper's hard threshold on the distribution of connectivity values:
    mu + k_exc sigma over the positive values (excitatory), mu + k_inh sigma
    over the magnitudes of the negative values (inhibitory)."""
    N = S.shape[0]
    off = cm.offdiag(N)
    v = S[off]
    pos = v[v > 0]
    neg = -v[v < 0]
    if pos.size:
        mu_pos, sd_pos = float(pos.mean()), float(pos.std())
        thr_exc = mu_pos + k_exc * sd_pos
    else:
        mu_pos = sd_pos = float("nan")
        thr_exc = float("inf")
    if neg.size:
        mu_neg, sd_neg = float(neg.mean()), float(neg.std())
        thr_inh = mu_neg + k_inh * sd_neg
    else:
        mu_neg = sd_neg = float("nan")
        thr_inh = float("inf")
    exc = (S > thr_exc) & off
    inh = (S < -thr_inh) & off
    info = dict(threshold_exc=float(thr_exc), threshold_inh=float(thr_inh),
                mu_pos=mu_pos, sigma_pos=sd_pos, mu_neg=mu_neg, sigma_neg=sd_neg,
                k_exc=float(k_exc), k_inh=float(k_inh),
                n_exc=int(exc.sum()), n_inh=int(inh.sum()))
    return exc, inh, info


def label_free(spikes, n_neurons, S, **params):
    """The method's own operating point (spikes only, via ``S``)."""
    p = _p(params)
    S = np.asarray(S, np.float64)
    exc, inh, info = _hard_threshold(S, p["k_exc"], p["k_inh"])
    pred = exc | inh
    info["n_pred"] = int(pred.sum())
    typing_inh = rank_typing(-S.sum(1), p["typing_fraction"])
    # the mask is sign-typed already (exc from S > 0, inh from S < 0), so the
    # typed read-out uses the same prediction
    typed_info = dict(info)
    return dict(untyped=(pred, info), typed=(pred.copy(), typed_info),
                typing_inh=typing_inh,
                rule="hard threshold mu+%gsigma (exc) / mu+%gsigma (inh) on the FNCCH peak "
                     "distribution" % (p["k_exc"], p["k_inh"]))
