"""Memory-efficient sparse re-implementation of the repo's lag-resolved ridge GLM.

Mathematically identical to glm_connectivity._lagged_features + glm_connectivity(),
but never materialises the [max_lag*N, T] dense feature block (13 GB at N=926).
"""
import glob
import os

import numpy as np
import scipy.sparse as sp


def load_session(session_dir, bin_ms=5.0):
    """Load spikes into a sparse [N, T] CSR matrix plus recording boundaries."""
    rec_paths = [p for p in sorted(glob.glob(os.path.join(session_dir, "recording*.npz")))
                 if "raster" not in os.path.basename(p)]
    rows, cols, boundaries = [], [], [0]
    n_neurons = None
    for rp in rec_paths:
        d = np.load(rp, allow_pickle=True)
        st, dur = d["spike_times"], float(d["duration"])
        if n_neurons is None:
            n_neurons = len(st)
        T = int(dur / bin_ms)
        off = boundaries[-1]
        for i in range(n_neurons):
            t = np.atleast_1d(np.asarray(st[i], float))
            if len(t):
                b = np.clip((t / bin_ms).astype(np.int64), 0, T - 1)
                b = np.unique(b)                      # binary bins, as in the repo
                rows.append(np.full(len(b), i, np.int32))
                cols.append((b + off).astype(np.int64))
        boundaries.append(off + T)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(n_neurons, boundaries[-1]))
    return M, boundaries


def _shift(M, boundaries, lag):
    """Sparse equivalent of the repo's shifted feature block for a given lag.

    A spike at bin t inside recording [s, e) moves to t + lag, and is dropped if
    it would cross into the next recording -- exactly the repo's zeroing of
    sh[:, b:b+lag] at every internal boundary.
    """
    Mc = M.tocoo()
    r, c = Mc.row, Mc.col
    ends = np.asarray(boundaries[1:], np.int64)
    seg_end = ends[np.searchsorted(ends, c, side="right")]
    new_c = c + lag
    keep = new_c < seg_end
    return sp.csr_matrix(
        (np.ones(int(keep.sum()), np.float32), (r[keep], new_c[keep])), shape=M.shape)


def fit_B(M, boundaries, max_lag=6, l2=2.0):
    """Joint lag-resolved ridge fit. Returns B [max_lag, N, N] = B[lag, pre, post]."""
    n = M.shape[0]
    S = [_shift(M, boundaries, k + 1) for k in range(max_lag)]
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
    G[np.diag_indices_from(G)] += l2
    B = np.linalg.solve(G, RHS)
    return B.reshape(max_lag, n, n)


def readout(B, mode="lag1"):
    """Reduce B [lag, pre, post] to a signed [N, N] score, as in the repo."""
    if mode == "lag1":
        W = B[0].copy()
    elif mode == "sum":
        W = B.sum(0)
    elif mode == "peak":
        idx = np.abs(B).argmax(0)
        W = np.take_along_axis(B, idx[None], 0)[0]
    else:
        raise ValueError(mode)
    np.fill_diagonal(W, 0.0)
    return W


def jitter(M, boundaries, jitter_bins, rng):
    """Spike-jitter surrogate: perturb each spike within its own recording."""
    Mc = M.tocoo()
    r, c = Mc.row, Mc.col.astype(np.int64)
    starts = np.asarray(boundaries[:-1], np.int64)
    ends = np.asarray(boundaries[1:], np.int64)
    k = np.searchsorted(ends, c, side="right")
    lo, hi = starts[k], ends[k] - 1
    nc = np.clip(c + rng.integers(-jitter_bins, jitter_bins + 1, len(c)), lo, hi)
    out = sp.csr_matrix((np.ones(len(r), np.float32), (r, nc)), shape=M.shape)
    out.sum_duplicates()
    out.data[:] = 1.0
    return out


def load_ground_truth(session_dir):
    net_files = sorted(glob.glob(os.path.join(session_dir, "network_*.npz")))
    net = np.load(net_files[0], allow_pickle=True)
    inh = net["neuron_is_inhibitory"].astype(bool)
    n = len(inh)
    A_exc = np.zeros((n, n), bool)
    A_inh = np.zeros((n, n), bool)
    for row in net["connections"]:
        pre, post = int(row[0]), int(row[1])
        if str(row[3]) == "inh" or inh[pre]:
            A_inh[pre, post] = True
        else:
            A_exc[pre, post] = True
    return {"A_exc": A_exc, "A_inh": A_inh, "is_inhibitory": inh}
