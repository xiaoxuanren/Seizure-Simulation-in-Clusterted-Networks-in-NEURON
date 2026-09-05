"""Deconvolution-based spike transmission gain (STG) -- Spivak et al. 2022.

Source
------
Spivak L, Levi A, Sloin HE, Someck S, Stark E (2022) Deconvolution improves
the detection and quantification of spike transmission gain from spike
trains.  Commun Biol 5:520.  doi:10.1038/s42003-022-03450-5

The published method
--------------------
For a reference unit 1 and a target unit 2, at bin size B = 1 ms over a
histogram range of 2M + 1 = 61 bins (lags -30..30 ms):

1. CCH_12 and the two auto-correlograms ACH_1, ACH_2.
2. Each ACH is normalised: the zero-lag bin is set to zero, the mean is
   subtracted, every bin is divided by the unit's total spike count, and
   the zero-lag bin is then set so that the ACH sums to one.  A unit with
   no burst structure therefore has a delta ACH and is left untouched.
3. Deconvolution  dcCCH_12 = F^-1( F(CCH_12) / (F(ACH_1) F(ACH_2)) ),
   which removes the multiplicative imprint of the reference unit's own
   bursting (and the target's) from the transmission estimate.
4. Predictor: hollowed median filter on the dcCCH,
   pred(m) = median{dcCCH(m - delta .. m - 1), dcCCH(m + 1 .. m + delta)},
   delta = 5 ms.
5. Spike transmission curve  STC(m) = (dcCCH(m) - pred(m)) / n_ref  (excess
   target spikes per reference spike) over the region of interest
   ROI = 0 < tau <= 5 ms; the STG is the sum of STC over the ROI bins bounded
   by the zero crossings around the extremum.  Troughs give negative STG
   (inhibition) by the same procedure.
6. Significance: Poisson test of the dcCCH against the predictor inside the
   ROI at alpha = 0.001.

Validated in the paper on point-process simulations, a 100-neuron CA1 model
and 1041 PYR / 215 INT recorded in mouse CA1.  The method is explicitly
designed to remove presynaptic burst-firing effects from the transmission
estimate, which makes it the strongest correlation-based competitor on a
burst-dominated network such as this one.

ORIENTATION (package invariant, see accumulate.py)
--------------------------------------------------
``S[i, j]`` is evidence for the directed edge i -> j: row = presynaptic
(reference) neuron, column = postsynaptic (target).  With the shared lag
products ``C = spikes.at(1.0).lag_products(50)``, ``C[l][i, j]`` counts the
bins in which i fired l bins BEFORE j, so the CCH of the ordered pair (i, j)
at lag +l is ``C[l][i, j]`` and at lag -l it is ``C[l][j, i]`` (the
transpose).  ``ACH_i(l) = C[|l|][i, i]``.  A causal peak of the pair (i, j)
therefore lies at positive lags, and ``E[l - 1, i, j]`` of
:func:`lag_profile` is STC_ij(+l ms).

Implementation on the shared statistics
---------------------------------------
Everything is read from the 1 ms pass ``C[0..30]`` (a slice of the shared
lags 0..50 stack; the stack itself is never upcast).  Pairs are processed in
row chunks: a ``[chunk, N, 61]`` float64 CCH block is deconvolved with
``numpy.fft.rfft/irfft`` along the lag axis, the hollowed median predictor is
evaluated only at the lags needed (ROI bins, and lags 1..30 for the profile),
and the ROI extremum / zero-crossing support / sum is vectorised over the
chunk.  Peak memory is a few hundred MB on top of the shared stack.

The candidate set is every ordered off-diagonal pair; the diagonal is 0.
Nothing here reads ground truth; every constant is in ``PARAMS``.

Deviations from the published method
------------------------------------
* FFT alignment.  The deconvolution is the circular FFT over the 61-bin
  window (the paper's histogram range); the ACHs are ``ifftshift``-ed so
  that their zero lag sits at index 0 and the deconvolved CCH keeps the
  CCH's own lag alignment (zero lag at index M).  Without this the quotient
  is circularly shifted by one bin.
* Spectral floor.  ``|F(ACH)|`` is floored at ``fach_floor`` (1e-6) before
  dividing, to bound the amplification for strongly bursting units whose
  normalised-ACH spectrum passes near zero.  The paper applies no floor.
* Normalisation detail.  The mean subtracted from the ACH is taken over all
  2M + 1 bins including the (already zeroed) zero-lag bin; a unit with zero
  spikes gets a delta ACH (no deconvolution) instead of a division by zero.
* Edge bins.  For lags within ``delta`` of the window edge (|lag| > 25 ms)
  the hollowed median uses the truncated window.  This only affects the lag
  profile at lags 26..30 ms, never the score (ROI <= 15 ms).
* Poisson test on real-valued counts.  The deconvolved CCH is not integer,
  so the tails are evaluated through the regularised incomplete gamma
  function, the continuous extension of the Poisson CDF:
  P(X >= k) = gammainc(k, lambda) (upper tail, excitatory) and
  P(X <= k) = gammaincc(k + 1, lambda) (lower tail, inhibitory).  A
  deconvolved count <= 0 gives an upper-tail p of 1 and is treated as 0 in
  the lower tail.
* Rate floor for the Poisson test (``pred_floor``).  At this network's
  ~0.3 Hz mean rate the expected coincidence count per 1 ms bin is ~0.2, so
  the hollowed median of ten neighbouring bins is 0 for most pairs and the
  paper's test is undefined (lambda = 0; any two deconvolved coincidences
  would be "significant").  The paper's hours-long multi-Hz CA1 recordings
  never meet this.  The rate is therefore floored at the independence
  expectation of the pair, n_i n_j / T_l (``pred_floor="chance"``, derived
  from spikes only); a float sets a constant floor instead.  The floor
  enters the significance test only, never the score.
* Multiple comparisons.  The ROI bins are tested at
  ``alpha / n_roi_bins`` (Bonferroni), a conservative reading of the
  paper's per-bin Poisson test at alpha = 0.001; a pair is called at either
  tail (untyped) or at the tail matching its rank-rule E/I type (typed).
* Unit inclusion.  No quality / minimum-rate criteria: every ordered pair
  of the network is scored, as the harness requires.
* E/I typing.  The paper does not type units from the STG; the typed
  read-out uses the package's rank rule on ``-S.sum(1)``.

Mismatch with this network (why ``stg_roi15`` exists)
-----------------------------------------------------
The faithful ROI (0 < tau <= 5 ms) covers only the shortest of this
network's conduction delays (1.5 + 2.0 x distance ms, 1.5-13.5 ms); a
synapse with a 9 ms delay contributes nothing to the score under the
published ROI.  The registry variant ``stg_roi15`` (``roi_ms=(0, 15)``)
widens the ROI to the network's delay range; it is a documented probe of
that mismatch, not a tuned parameter.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaincc

from baselines import common as cm
from baselines.labelfree import bonferroni_alpha, rank_typing

NAME = "stg"
TIER = 2
DIRECTED = True
CITATION = ("Spivak L, Levi A, Sloin HE, Someck S, Stark E (2022) Deconvolution improves the "
            "detection and quantification of spike transmission gain from spike trains. "
            "Commun Biol 5:520. doi:10.1038/s42003-022-03450-5")
PARAMS = dict(
    bin_ms=cm.FINE_BIN_MS,      # 1 ms bins (paper: B = 1 ms)
    half_window_ms=30.0,        # histogram range 2M+1 = 61 ms (lags -30..30)
    delta_ms=5.0,               # hollowed-median half width
    roi_ms=(0.0, 5.0),          # region of interest 0 < tau <= 5 ms
    alpha=0.001,                # Poisson test level
    fach_floor=1e-6,            # floor on |F(ACH)| before dividing (deviation)
    pred_floor="chance",        # Poisson-rate floor: n_i n_j / T_l, or a float (deviation)
    chunk_rows=64,              # presynaptic rows per dense block (memory only)
    typing_fraction=cm.TYPING_FRACTION,  # rank-rule E/I typing for the typed read-out
)


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


# ------------------------------------------------------------------------- #
# pieces
# ------------------------------------------------------------------------- #
def _geometry(p):
    """(bin_ms, M, delta, roi) with M / delta in bins and roi the positive
    lag bins l with roi_lo < l * bin_ms <= roi_hi."""
    bin_ms = float(p["bin_ms"])
    M = int(round(float(p["half_window_ms"]) / bin_ms))
    delta = int(round(float(p["delta_ms"]) / bin_ms))
    lo, hi = (float(x) for x in p["roi_ms"])
    roi = np.array([l for l in range(1, M + 1) if lo < l * bin_ms <= hi], int)
    if M < 1 or delta < 1 or not len(roi):
        raise ValueError("stg: empty ROI or degenerate window (%r)" % (p,))
    return bin_ms, M, delta, roi


def _normalised_ach(C, n_spikes, M):
    """Paper's ACH normalisation -> ``[N, 2M+1]`` float64, zero lag at index M.

    Zero-lag bin := 0; subtract the mean; divide by the spike count; set the
    zero-lag bin so the ACH sums to one.  Units without spikes get a delta."""
    N = C.shape[1]
    a = np.empty((N, 2 * M + 1), np.float64)
    for l in range(M + 1):
        d = np.diagonal(C[l]).astype(np.float64)
        a[:, M + l] = d
        a[:, M - l] = d
    a[:, M] = 0.0
    a -= a.mean(1, keepdims=True)
    ok = np.asarray(n_spikes, np.float64) > 0
    a[ok] /= np.asarray(n_spikes, np.float64)[ok, None]
    a[~ok] = 0.0
    a[:, M] = 1.0 - (a.sum(1) - a[:, M])
    return a


def _ach_spectrum(a, floor):
    """rfft of the zero-lag-at-index-0 ACH, magnitude floored at ``floor``."""
    F = np.fft.rfft(np.fft.ifftshift(a, axes=1), axis=1)
    mag = np.abs(F)
    scale = np.where(mag < floor, floor / np.maximum(mag, 1e-300), 1.0)
    F = F * scale
    F[mag == 0.0] = floor
    return F


def _cch_block(C, a, b, M):
    """CCH of the ordered pairs (i in a..b, j) over lags -M..M -> [b-a, N, 2M+1].
    Positive lags come from ``C[l][i, j]``, negative ones from ``C[l][j, i]``."""
    N = C.shape[1]
    x = np.empty((b - a, N, 2 * M + 1), np.float64)
    for l in range(M + 1):
        x[:, :, M + l] = C[l][a:b, :]
        if l:
            x[:, :, M - l] = C[l][:, a:b].T
    return x


def _deconvolve(cch, F_rows, F_all, L):
    """dcCCH = F^-1( F(CCH_ij) / (F(ACH_i) F(ACH_j)) ), real part."""
    F = np.fft.rfft(cch, axis=-1)
    F /= F_rows[:, None, :] * F_all[None, :, :]
    return np.fft.irfft(F, n=L, axis=-1)


def _hollow_median(x, delta, idx):
    """Hollowed median predictor of ``x[..., L]`` at the lag indices ``idx``
    (window m-delta..m-1, m+1..m+delta, truncated at the array edge)."""
    L = x.shape[-1]
    offs = np.concatenate([np.arange(-delta, 0), np.arange(1, delta + 1)])
    out = np.empty(x.shape[:-1] + (len(idx),), np.float64)
    for k, m in enumerate(idx):
        cols = m + offs
        cols = cols[(cols >= 0) & (cols < L)]
        out[..., k] = np.median(x[..., cols], axis=-1)
    return out


def _stg_sum(R):
    """Signed STG from the ROI STC ``R[..., K]``: sum over the run of bins
    sharing the sign of the ROI extremum and contiguous with it (bounded by
    the zero crossings on both sides, clipped to the ROI)."""
    ext = np.argmax(np.abs(R), axis=-1)
    sgn = np.sign(np.take_along_axis(R, ext[..., None], -1))
    same = (R * sgn) > 0.0
    run = np.cumsum(~same, axis=-1)
    support = same & (run == np.take_along_axis(run, ext[..., None], -1))
    return (R * support).sum(-1)


def _poisson_tails(k, lam):
    """Continuous Poisson tails of the deconvolved count ``k`` at rate ``lam``:
    upper P(X >= k) = gammainc(k, lam) (p = 1 for k <= 0) and
    lower P(X <= k) = gammaincc(k + 1, lam) with k clipped at 0."""
    pos = k > 0.0
    p_exc = np.where(pos, gammainc(np.where(pos, k, 1.0), lam), 1.0)
    p_inh = gammaincc(np.maximum(k, 0.0) + 1.0, lam)
    return p_exc, p_inh


def _core(sd, p, want_profile=False, want_pvals=False):
    """One chunked pass over all ordered pairs.  Returns dict(S, lags_ms, E,
    p_exc, p_inh, n_roi)."""
    bin_ms, M, delta, roi = _geometry(p)
    N = sd.n_neurons
    L = 2 * M + 1
    # the shared fine-timing stack (lags 0..50); slice, never upcast the whole
    C = sd.lag_products(max(M, cm.FINE_MAX_LAG))
    n = np.asarray(sd.n_spikes, np.float64)
    inv_n = np.where(n > 0, 1.0 / np.maximum(n, 1.0), 0.0)
    F_all = _ach_spectrum(_normalised_ach(C, n, M), float(p["fach_floor"]))

    prof_lags = np.arange(1, M + 1) if want_profile else np.empty(0, int)
    need = np.unique(np.concatenate([roi, prof_lags]))     # positive lag bins
    idx = M + need                                          # index on the 61-bin axis
    roi_pos = np.searchsorted(need, roi)
    prof_pos = np.searchsorted(need, prof_lags)

    S = np.zeros((N, N), np.float64)
    E = np.zeros((len(prof_lags), N, N), np.float64) if want_profile else None
    p_exc = np.ones((N, N), np.float64) if want_pvals else None
    p_inh = np.ones((N, N), np.float64) if want_pvals else None
    floor_mode = p["pred_floor"]
    if want_pvals and floor_mode == "chance":
        T_roi = np.asarray(sd.valid_pairs(roi.tolist()), np.float64)     # [K]

    chunk = max(1, int(p["chunk_rows"]))
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        dc = _deconvolve(_cch_block(C, a, b, M), F_all[a:b], F_all, L)  # [nb, N, L]
        pred = _hollow_median(dc, delta, idx)                             # [nb, N, len(need)]
        dcn = dc[..., idx]
        del dc
        stc = (dcn - pred) * inv_n[a:b, None, None]
        S[a:b] = _stg_sum(stc[..., roi_pos])
        if want_profile:
            E[:, a:b, :] = np.transpose(stc[..., prof_pos], (2, 0, 1))
        if want_pvals:
            lam = pred[..., roi_pos]
            if floor_mode == "chance":
                chance = n[a:b, None, None] * n[None, :, None] / np.maximum(T_roi, 1.0)[None, None, :]
                lam = np.maximum(lam, chance)
            else:
                lam = np.maximum(lam, float(floor_mode))
            pe, pi = _poisson_tails(dcn[..., roi_pos], lam)
            p_exc[a:b] = pe.min(-1)
            p_inh[a:b] = pi.min(-1)

    np.fill_diagonal(S, 0.0)
    if want_profile:
        for k in range(E.shape[0]):
            np.fill_diagonal(E[k], 0.0)
    if want_pvals:
        np.fill_diagonal(p_exc, 1.0)
        np.fill_diagonal(p_inh, 1.0)
    return dict(S=S, lags_ms=prof_lags.astype(np.float64) * bin_ms, E=E,
                p_exc=p_exc, p_inh=p_inh, n_roi=int(len(roi)))


# ------------------------------------------------------------------------- #
# contract
# ------------------------------------------------------------------------- #
def score(spikes, n_neurons, **params):
    """Signed STG ``S[pre, post]`` (excess target spikes per reference spike
    summed over the ROI support of the extremum)."""
    p = _p(params)
    return _core(spikes.at(p["bin_ms"]), p)["S"]


def lag_profile(spikes, n_neurons, **params):
    """(lags_ms 1..M, E[l, pre, post] = STC at lag +l), signed."""
    p = _p(params)
    r = _core(spikes.at(p["bin_ms"]), p, want_profile=True)
    return r["lags_ms"], r["E"]


def label_free(spikes, n_neurons, S, **params):
    """The paper's operating point: Poisson test of the deconvolved counts
    against the hollowed-median predictor inside the ROI, alpha = 0.001,
    Bonferroni over the ROI bins.  Untyped: either tail.  Typed: the
    excitatory tail on rows typed excitatory, the inhibitory tail on rows
    typed inhibitory (rank rule on ``-S.sum(1)``)."""
    p = _p(params)
    S = np.asarray(S, np.float64)
    N = S.shape[0]
    r = _core(spikes.at(p["bin_ms"]), p, want_pvals=True)
    a_bin = bonferroni_alpha(p["alpha"], r["n_roi"])
    off = cm.offdiag(N)
    exc_sig = (r["p_exc"] < a_bin) & off
    inh_sig = (r["p_inh"] < a_bin) & off
    untyped = exc_sig | inh_sig
    inh = rank_typing(-S.sum(1), p["typing_fraction"])
    typed = (exc_sig & ~inh[:, None]) | (inh_sig & inh[:, None])
    base = dict(alpha=float(p["alpha"]), alpha_per_bin=float(a_bin), n_roi_bins=int(r["n_roi"]),
                n_exc_sig=int(exc_sig.sum()), n_inh_sig=int(inh_sig.sum()))
    info_u = dict(base, n_pred=int(untyped.sum()))
    info_t = dict(base, n_inh_typed=int(inh.sum()), n_pred=int(typed.sum()),
                  n_pred_exc_layer=int((exc_sig & ~inh[:, None]).sum()),
                  n_pred_inh_layer=int((inh_sig & inh[:, None]).sum()))
    return dict(untyped=(untyped, info_u), typed=(typed, info_t), typing_inh=inh,
                rule="Poisson test dcCCH vs hollowed-median predictor in ROI %g-%g ms, "
                     "alpha %g Bonferroni over %d bins" % (
                         float(p["roi_ms"][0]), float(p["roi_ms"][1]), float(p["alpha"]),
                         int(r["n_roi"])))
