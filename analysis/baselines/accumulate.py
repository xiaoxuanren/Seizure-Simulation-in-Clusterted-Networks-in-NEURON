"""Shared accumulator: ONE pass over per-recording sparse products yields the
sufficient statistics of every method in this package, including the shipped
ridge GLM.

ORIENTATION CONVENTION -- the package invariant (asserted in test_accumulate.py)
-------------------------------------------------------------------------------
Every (N, N) array produced anywhere in ``analysis/baselines`` is indexed
``[pre, post]``: ``X[i, j]`` is evidence for a directed edge ``i -> j``.
For the lag products returned by :func:`lag_products`,

    C[l][i, j] = number of bins t with  M[i, t - l] == 1  and  M[j, t] == 1

i.e. the raw cross-correlogram count of the ORDERED pair (i, j) at lag +l:
"neuron i fires l bins BEFORE neuron j".  A negative lag is the transpose:
``ccg_ij(-l) = C[l][j, i]``.  This is exactly the shipped GLM's convention:
its design row-block for lag l is the shifted matrix ``S_l = shift(M, l)``
(a spike at bin t moved to bin t + l), so the right-hand side of the ridge
normal equations, ``F X^T``, at lag l IS ``C[l]`` -- the same sufficient
statistic before whitening -- and the solved coefficient tensor is
``B[l, pre, post]``.

Three normalisations of the same statistics
-------------------------------------------
* raw counts          ``C[l]``                                  -> classical / jittered CCG
* rate-normalised     ``(C[l]/T_l - p_i p_j) / sqrt(v_i v_j)``   -> Pearson (zero-lag, lagged)
* precision-whitened  ``(F F^T + lambda I)^-1 F X^T``            -> the shipped ridge GLM

Recording boundaries are respected exactly as in ``sparse_glm``: a shifted
spike that would cross into the next recording (segment) is dropped, so no
predictor and no correlogram count ever bridges two independent trials.

Two passes are used by the runner: a 5 ms pass with ``max_lag=6`` that
reproduces the shipped ``glm_connectivity_sum4_5ms`` fit bit-for-bit, and a
1 ms pass with lags 0..50 for the fine-timing correlogram methods.  Nothing
here reads ground truth.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import sparse_glm as _sg  # noqa: E402  (the shipped path; never modified here)

# re-exported so every method module uses the ONE shift/jitter implementation
shift = _sg._shift
jitter = _sg.jitter


def load_session(session_dir, bin_ms, n_recordings=None):
    """Sparse binary [N, T] spike matrix + recording boundaries.

    Thin wrapper over ``sparse_glm.load_session`` (identical binning: bins are
    binary, ``T = int(duration / bin_ms)`` per recording) that optionally keeps
    only the first ``n_recordings`` recordings, as ``glm_fit_current.py`` and
    ``chtc/analysis_one.py`` do.
    """
    M, bnd = _sg.load_session(session_dir, bin_ms=bin_ms)
    if n_recordings is not None and n_recordings < len(bnd) - 1:
        M = M[:, :bnd[n_recordings]].tocsr()
        bnd = list(bnd[:n_recordings + 1])
    return M, list(bnd)


def segment_stats(M, bnd):
    """Per-neuron spike counts, total bins and segment count."""
    n_spikes = np.asarray(M.sum(1)).ravel().astype(np.float64)
    return dict(n_spikes=n_spikes, T=int(bnd[-1]), n_segments=len(bnd) - 1,
                seg_len=np.diff(np.asarray(bnd, np.int64)))


def lag_products(M, bnd, max_lag, lags=None, dtype=np.float32, progress=None):
    """Raw cross-correlogram counts ``C[l][pre, post]`` for l = 0..max_lag.

    ``C[0] = M M^T`` (same-bin coincidences; its diagonal is the spike count),
    ``C[l] = shift(M, l) M^T`` for l >= 1.  Returned as an array of shape
    ``[max_lag + 1, N, N]`` (``lags`` selects a subset, still indexed by the
    position in ``lags``).  Counts are exact in float32 below 2**24.
    """
    lag_list = list(range(max_lag + 1)) if lags is None else list(lags)
    n = M.shape[0]
    Mt = M.T.tocsc()
    out = np.empty((len(lag_list), n, n), dtype)
    for k, l in enumerate(lag_list):
        S = M if l == 0 else shift(M, bnd, l)
        out[k] = (S @ Mt).toarray()
        if progress is not None:
            progress(k, len(lag_list))
    return out


def valid_pairs_per_lag(bnd, lags):
    """Number of (t - l, t) bin pairs that lie inside one segment, per lag."""
    seg = np.diff(np.asarray(bnd, np.int64))
    return np.array([int(np.clip(seg - l, 0, None).sum()) for l in lags], np.float64)


def pearson_from_counts(C_l, n_spikes, T_l):
    """Lag-l Pearson correlation of binary trains from the raw count matrix.

    ``r_ij = (C_l[i, j] / T_l - p_i p_j) / sqrt(p_i (1 - p_i) p_j (1 - p_j))``
    with ``p = n_spikes / T_l``.  Orientation follows ``C_l``: ``r[i, j]`` is
    the correlation of i's spikes with j's spikes l bins LATER.  Rate
    normalisation only -- no whitening against other neurons.
    """
    p = np.asarray(n_spikes, np.float64) / float(T_l)
    v = p * (1.0 - p)
    denom = np.sqrt(np.outer(v, v))
    cov = np.asarray(C_l, np.float64) / float(T_l) - np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, cov / denom, 0.0)
    return r


def gram(M, bnd, max_lag):
    """The ridge normal equations of the shipped GLM, assembled exactly as
    ``sparse_glm.fit_B`` does (same products, same dtypes, same order).

    Returns ``(G, RHS)`` with ``G`` the ``[max_lag N, max_lag N]`` Gram matrix
    ``F F^T`` (no ridge term yet) and ``RHS = F M^T`` of shape
    ``[max_lag N, N]`` whose row block ``a`` is ``C[a + 1]`` -- the raw
    cross-correlogram at lag a + 1.
    """
    n = M.shape[0]
    S = [shift(M, bnd, k + 1) for k in range(max_lag)]
    G = np.zeros((max_lag * n, max_lag * n), np.float64)
    RHS = np.zeros((max_lag * n, n), np.float64)
    Mt = M.T.tocsc()
    for a in range(max_lag):
        RHS[a * n:(a + 1) * n] = (S[a] @ Mt).toarray()
        for b in range(a, max_lag):
            blk = (S[a] @ S[b].T).toarray().astype(np.float64)
            G[a * n:(a + 1) * n, b * n:(b + 1) * n] = blk
            if b != a:
                G[b * n:(b + 1) * n, a * n:(a + 1) * n] = blk.T
    return G, RHS


def solve_ridge(G, RHS, l2, max_lag):
    """``B[lag, pre, post] = (G + l2 I)^-1 RHS`` -- the shipped GLM solve.

    ``G`` is not modified.  With ``(G, RHS)`` from :func:`gram` this is
    bit-identical to ``sparse_glm.fit_B(M, bnd, max_lag, l2)``.
    """
    n = RHS.shape[1]
    Gl = G.copy()
    Gl[np.diag_indices_from(Gl)] += l2
    return np.linalg.solve(Gl, RHS).reshape(max_lag, n, n)


def readout_sum_k(B, k):
    """The shipped ``sum_k`` readout: sum of coefficients over lags 1..k,
    diagonal zeroed.  ``k=4`` is the headline ``sum4``."""
    W = B[:k].sum(0)
    np.fill_diagonal(W, 0.0)
    return W


def trim_windows(M, bnd, windows_bins):
    """Drop bins inside per-segment windows; every retained run becomes its
    own segment so lagged features never bridge a cut (the mechanics of
    ``analysis/burstexcl_arm_compare.py``).

    ``windows_bins[r]`` is an ``(k, 2)`` array of ``[lo, hi)`` bin offsets
    relative to segment r's start.  Returns ``(M_trimmed, new_bnd, n_dropped)``.
    """
    blocks, new_bnd, dropped = [], [0], 0
    for r in range(len(bnd) - 1):
        s, e = bnd[r], bnd[r + 1]
        rec_bins = e - s
        keep = np.ones(rec_bins, bool)
        for lo, hi in np.asarray(windows_bins[r], np.int64).reshape(-1, 2):
            lo, hi = max(0, int(lo)), min(rec_bins, int(hi))
            if hi > lo:
                keep[lo:hi] = False
        dropped += int((~keep).sum())
        idx = np.flatnonzero(keep)
        if not len(idx):
            continue
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            blocks.append(M[:, s + run[0]:s + run[-1] + 1])
            new_bnd.append(new_bnd[-1] + len(run))
    if not blocks:
        raise ValueError("every bin was dropped")
    return sp.hstack(blocks, format="csr"), new_bnd, dropped


def windows_ms_to_bins(windows_ms, bin_ms, pad_ms):
    """Per-recording ``(start_ms, end_ms)`` windows -> padded ``[lo, hi)`` bin
    windows, exactly as ``burstexcl_arm_compare.py`` rounds them."""
    out = []
    for w in windows_ms:
        w = np.asarray(w, float).reshape(-1, 2)
        if not len(w):
            out.append(np.empty((0, 2), np.int64))
            continue
        lo = np.round((w[:, 0] - pad_ms) / bin_ms).astype(np.int64)
        hi = np.round((w[:, 1] + pad_ms) / bin_ms).astype(np.int64)
        out.append(np.stack([np.maximum(lo, 0), hi], 1))
    return out
