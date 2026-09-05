"""Delayed transfer entropy (Schreiber 2000; Ito et al. 2011): k = l = 1,
peak over delays 1..30 ms ("TEPk"), run at FOUR bin widths (1, 2, 5, 10 ms).

Transfer entropy from a SOURCE train y to a TARGET train x at delay d is the
information that the source's bin d steps back carries about the target's
current bin beyond the target's own one-bin history (Schreiber 2000, eq. 4;
Ito et al. 2011, eq. 2, "delayed TE", history orders k = l = 1):

    TE_{y->x}(d) = sum_{x_t, x_{t-1}, y_{t-d}}  p(x_t, x_{t-1}, y_{t-d})
                   log2 [ p(x_t | x_{t-1}, y_{t-d}) / p(x_t | x_{t-1}) ]

with x_t, x_{t-1}, y_{t-d} in {0, 1}.  The plug-in estimate is the 2x2x2
contingency table of (x_t, x_{t-1}, y_{t-d}) over every bin t at which all
three lie inside one recording.  Ito et al. score a pair by the PEAK of
TE(d) over delays d = 1..30 bins at 1 ms ("TEPk"); that is what ``score``
returns, with a sign added (see Deviations).

ORIENTATION (the package invariant, ``accumulate.py``)
------------------------------------------------------
``S[i, j] = TE_{i->j}``: row i is the SOURCE (presynaptic) train y, column j
the TARGET (postsynaptic) train x.  The shared lag products give
``C[d][i, j] = #{t : M[i, t-d] = 1 and M[j, t] = 1}`` ("i fires d bins before
j"), which is exactly the (x=1, y=1) cell of the table for the pair (i, j)
at delay d, so ``C[d]`` enters here unchanged.  The other two pairwise counts
are the same kind of product with the shifted / history-masked matrices:
with ``Sd = shift(M, d)``, ``S1 = shift(M, 1)`` and ``H = M * S1`` (bins in
which the target fired at t AND at t-1),

    n_xhy[i, j] = (Sd @ H.T)[i, j]    x=1, h=1, y=1
    n_xy[i, j]  = (Sd @ M.T)[i, j]  = C[d][i, j]
    n_hy[i, j]  = (Sd @ S1.T)[i, j]   h=1, y=1
    n_y[i] = row sums of Sd; n_x[j], n_h[j], n_xh[j] = row sums of M, S1, H
    restricted to the valid bins; T_d = number of valid bins.

The remaining cells follow by inclusion-exclusion.  ``shift`` drops spikes
that would cross a recording boundary, so no count bridges two recordings.

BIN-WIDTH DEPENDENCE -- why four widths are run
-----------------------------------------------
With one-bin histories the estimate is a 2x2x2 table of binary bins, so the
bin width sets everything at once: the resolution of the delay axis, the
fraction of a synapse's latency spread (here 1.5-13.5 ms conduction delay
plus the postsynaptic integration time) that lands in one bin, and how much
of the network's shared drive counts as "same bin".  At 1 ms a monosynaptic
effect is diluted over several delay bins and each cell is sparse; at 10 ms
one bin gathers most of it but only three delays remain and coincident
common input is absorbed into the table.  Ito et al. (2011) report that
their TE measures are sensitive to the bin size (their primary analysis
used 1 ms bins with 30 delays).  Rather than picking one width, the registry
runs this module as ``te_1ms``, ``te_2ms``, ``te_5ms`` and ``te_10ms`` by
overriding ``bin_ms`` (``registry.VARIANTS``) and the sensitivity is reported
alongside the results.  The delay window is held fixed in milliseconds
(``max_delay_ms``), so the number of delays is round(30 / bin_ms) =
30, 15, 6, 3.

Deviations from the published method
------------------------------------
* History orders are fixed at k = l = 1 (Ito's delayed TE).  Their
  higher-order extension HOTE (k = 3, l = 2) and the coincidence-index
  variants (TE-CI, HOTE-CI) are not implemented; ``k`` / ``l`` other than 1
  raise ``NotImplementedError``.
* Ito et al. used 1 ms bins and delays 1..30 bins.  Here the 30 ms window
  is kept in ms and the bin width is swept (see above).
* SIGN.  Ito's TE is unsigned.  To meet the package contract (positive =
  excitatory, negative = inhibitory) the peak TE is multiplied by +1 when,
  at the peak delay, a source spike RAISES the target's firing probability,
  p(x=1 | h=0, y=1) > p(x=1 | h=0, y=0) (h = the target's previous bin),
  and by -1 when it lowers it.  Ties, and the undefined case p(y=1) = 0
  (source never fires in a valid bin, so TE = 0 anyway), default to +.
* No bias correction, surrogate normalisation or shuffling is applied; the
  raw plug-in TE is used, as in Ito's TEPk.
* Ito et al. thresholded by ROC against the known network, which needs the
  labels, so this module defines NO label-free operating point
  (``label_free`` is absent; the runner reports the oracle best-F1 and the
  rank-rule typing).
* Every neuron and every recording is used (Ito subsampled 100 of 1000
  neurons over 30 min); the table is pooled over recordings and, per delay,
  restricted to bins whose lagged partners lie in the same recording.
* At the two shared bin widths the accumulator's canonical lag range is
  requested (1 ms: lags 0..50, the fine-timing pass; 5 ms: lags 0..6, the
  GLM pass) so the cached lag products are shared with the other methods;
  only the delays 1..D are read.

Sources
-------
Schreiber T (2000) Measuring information transfer. Phys Rev Lett 85:461-464.
Ito S, Hansen ME, Heiland R, Lumsdaine A, Litke AM, Beggs JM (2011)
Extending transfer entropy improves identification of effective
connectivity in a spiking cortical network model. PLoS ONE 6(11):e27431.

Uses spikes only; nothing here reads ground truth.
"""
from __future__ import annotations

import numpy as np

from baselines import accumulate as acc
from baselines import common as cm

NAME = "transfer_entropy"
TIER = 2
DIRECTED = True
CITATION = ("Schreiber T (2000) Measuring information transfer. Phys Rev Lett 85:461-464; "
            "Ito S, Hansen ME, Heiland R, Lumsdaine A, Litke AM, Beggs JM (2011) Extending "
            "transfer entropy improves identification of effective connectivity in a spiking "
            "cortical network model. PLoS ONE 6(11):e27431.")
#: bin_ms is the swept hyperparameter (registry.VARIANTS te_1ms .. te_10ms);
#: max_delay_ms fixes the delay window in ms (D = round(max_delay_ms / bin_ms)
#: delays); k, l are the target / source history orders (only 1 implemented);
#: log_base 2 gives TE in bits as in Ito et al.
PARAMS = dict(bin_ms=1.0, max_delay_ms=30.0, k=1, l=1, log_base=2)


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    if int(q["k"]) != 1 or int(q["l"]) != 1:
        raise NotImplementedError("transfer_entropy implements history orders k = l = 1 only")
    return q


def _n_delays(p):
    return max(1, int(round(float(p["max_delay_ms"]) / float(p["bin_ms"]))))


def _shared_max_lag(bin_ms, n_delays):
    """Lag range to request from the accumulator: the canonical range of a
    shared pass (so its cache is reused), otherwise just the delays."""
    if float(bin_ms) == cm.FINE_BIN_MS:
        return max(n_delays, cm.FINE_MAX_LAG)
    if float(bin_ms) == cm.GLM_BIN_MS:
        return max(n_delays, cm.GLM_MAX_LAG)
    return n_delays


def _valid_mask(bnd, d):
    """1.0 at every bin t whose lagged partner t - d lies in the same recording
    (t - 1 then does too, since d >= 1); 0.0 elsewhere."""
    v = np.zeros(int(bnd[-1]), np.float64)
    for s, e in zip(bnd[:-1], bnd[1:]):
        if e - s > d:
            v[s + d:e] = 1.0
    return v


def _plogp_terms(c, m, q, r):
    """``c * [ln c - ln m - ln q + ln r]`` with the convention 0 ln 0 = 0:
    cells with c = 0 contribute nothing (their marginals may be 0 too)."""
    out = np.zeros_like(c)
    pos = c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        val = c * (np.log(c) - np.log(m) - np.log(q) + np.log(r))
    out[pos] = val[pos]
    return out


def _te_from_counts(n_xhy, n_xy, n_hy, n_xh, n_x, n_h, n_y, T_d, log_base):
    """Transfer entropy and its sign for every ordered pair from the counts.

    Pairwise counts ``n_xhy, n_xy, n_hy`` are ``[N, N]`` indexed
    ``[source i, target j]``; ``n_xh, n_x, n_h`` are per TARGET (``[N]``,
    broadcast along columns); ``n_y`` is per SOURCE (broadcast along rows);
    ``T_d`` is the number of valid bins.  Returns ``(te [N, N] >= 0,
    sign [N, N] in {-1, +1})``.
    """
    xh, x, h = n_xh[None, :], n_x[None, :], n_h[None, :]
    y = n_y[:, None]
    # the 2x2x2 table c[x, h, y] by inclusion-exclusion (exact integer counts;
    # the clip only guards the arithmetic)
    c = {
        (1, 1, 1): n_xhy,
        (1, 0, 1): n_xy - n_xhy,
        (0, 1, 1): n_hy - n_xhy,
        (0, 0, 1): y - n_xy - n_hy + n_xhy,
        (1, 1, 0): xh - n_xhy,
        (1, 0, 0): x - xh - n_xy + n_xhy,
        (0, 1, 0): h - xh - n_hy + n_xhy,
        (0, 0, 0): T_d - x - h - y + xh + n_xy + n_hy - n_xhy,
    }
    c = {k: np.maximum(v, 0.0) for k, v in c.items()}
    # marginals rebuilt from the cells so they are consistent by construction
    m = {(hh, yy): c[0, hh, yy] + c[1, hh, yy] for hh in (0, 1) for yy in (0, 1)}   # (h, y)
    q = {(xx, hh): c[xx, hh, 0] + c[xx, hh, 1] for xx in (0, 1) for hh in (0, 1)}   # (x, h)
    r = {hh: m[hh, 0] + m[hh, 1] for hh in (0, 1)}                                  # h
    te = np.zeros_like(n_xhy)
    for (xx, hh, yy), cell in c.items():
        te += _plogp_terms(cell, m[hh, yy], q[xx, hh], r[hh])
    te /= float(T_d) * np.log(float(log_base))
    te = np.maximum(te, 0.0)
    # sign: does a source spike raise the target's firing (h = 0 stratum)?
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = c[1, 0, 1] / m[0, 1]          # p(x=1 | h=0, y=1)
        p0 = c[1, 0, 0] / m[0, 0]          # p(x=1 | h=0, y=0)
    sign = np.where(p1 < p0, -1.0, 1.0)   # ties and NaN (p(y=1) = 0) -> +
    return te, sign


def _per_delay(sd, n_delays, log_base):
    """Yield ``(d, te [N, N], sign [N, N])`` for d = 1..n_delays, diagonal 0."""
    M, bnd = sd.M, sd.bnd
    n = sd.n_neurons
    C = sd.lag_products(_shared_max_lag(sd.bin_ms, n_delays))
    S1 = acc.shift(M, bnd, 1)
    H = M.multiply(S1).tocsr()
    Ht, S1t = H.T.tocsc(), S1.T.tocsc()
    for d in range(1, n_delays + 1):
        v = _valid_mask(bnd, d)
        T_d = float(v.sum())
        if T_d <= 0:
            z = np.zeros((n, n))
            yield d, z, np.ones((n, n))
            continue
        Sd = S1 if d == 1 else acc.shift(M, bnd, d)
        n_xhy = (Sd @ Ht).toarray().astype(np.float64)
        n_hy = (Sd @ S1t).toarray().astype(np.float64)
        n_xy = np.asarray(C[d], np.float64)              # slice only; never the stack
        n_y = np.asarray(Sd @ v, np.float64).ravel()
        n_x = np.asarray(M @ v, np.float64).ravel()
        n_h = np.asarray(S1 @ v, np.float64).ravel()
        n_xh = np.asarray(H @ v, np.float64).ravel()
        te, sign = _te_from_counts(n_xhy, n_xy, n_hy, n_xh, n_x, n_h, n_y, T_d, log_base)
        np.fill_diagonal(te, 0.0)
        yield d, te, sign


def score(spikes, n_neurons, **params):
    """``S[i, j] = sign * max_d TE_{i->j}(d)`` (Ito's TEPk, signed)."""
    p = _p(params)
    sd = spikes.at(p["bin_ms"])
    n = sd.n_neurons
    best = np.zeros((n, n))
    best_sign = np.ones((n, n))
    for _, te, sign in _per_delay(sd, _n_delays(p), p["log_base"]):
        better = te > best
        best[better] = te[better]
        best_sign[better] = sign[better]
    S = best * best_sign
    np.fill_diagonal(S, 0.0)
    return S


def lag_profile(spikes, n_neurons, **params):
    """``(lags_ms [D], E [D, N, N])`` with ``E[d-1] = TE(d)`` (unsigned, >= 0,
    so argmax over the first axis is the TE peak)."""
    p = _p(params)
    sd = spikes.at(p["bin_ms"])
    D = _n_delays(p)
    E = np.empty((D, sd.n_neurons, sd.n_neurons))
    for d, te, _ in _per_delay(sd, D, p["log_base"]):
        E[d - 1] = te
    return np.arange(1, D + 1, dtype=np.float64) * float(p["bin_ms"]), E
