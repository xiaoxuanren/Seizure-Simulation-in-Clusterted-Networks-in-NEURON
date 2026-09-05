"""GLMCC: a generalised linear model fitted to each pair's cross-correlogram
(Kobayashi et al. 2019).  Registry name ``glmcc``; variant ``glmcc_d14``
(transmission delays searched over 1..14 ms instead of 1..4 ms).

Source
------
Kobayashi R, Kurita S, Kurth A, Kitano K, Mizuseki K, Diesmann M, Richmond BJ,
Shinomoto S (2019) Reconstructing neuronal circuitry from parallel spike
trains.  Nat Commun 10:4468.  doi:10.1038/s41467-019-12225-2
Reference implementation: https://github.com/NII-Kobayashi/GLMCC

Published model
---------------
For every unordered pair (i, j) the cross-correlogram over a window of
W = 50 ms on each side at DELTA = 1 ms resolution (100 bins covering
[-50, 50) ms) is modelled as an inhomogeneous Poisson process with rate

    c(t) = exp( a(t) + J_ij f(t) + J_ji f(-t) ),
    f(t) = exp(-(t - d) / tau)   for t > d,   0 otherwise,

where ``a(t)`` is a slowly varying background (shared drive, rate
co-modulation), ``J_ij`` multiplies the causal kernel on POSITIVE lags
(i fires before j: the i -> j synapse), ``J_ji`` its mirror image on
negative lags (the j -> i synapse), ``tau`` is the synaptic time constant
(4 ms for the simulated data of the paper, 1 ms for hippocampal data) and
``d`` the transmission delay, chosen per pair by maximising the fitted model
over d = 1..4 ms (the set the reference code tests).  The background carries
the smoothness prior  p(a) ~ exp[-(1/gamma) integral (da/dt)^2 dt]  with
gamma = 5e-4 ms^-1, which the reference code writes on the binned model as
log p = -(beta / (2 DELTA)) sum_k (a_{k+1} - a_k)^2 with beta = 2/gamma = 4000.
The MAP estimate is computed from a_k = log(mean count), J = 0.1, with J
constrained to [-3, 5].  A connection is declared when |J| exceeds

    J_thr = c_alpha / sqrt( T tau lambda_i lambda_j ),   c_alpha = 5.16  (alpha = 0.001),

T the recording length (s), tau in s, lambda the firing rates (Hz); the sign
of J gives its type.  The paper validates the method on a network of 1000
Hodgkin-Huxley neurons simulated for 5400 s and on rat CA1 recordings in
vivo.

What this module computes
-------------------------
The 1 ms pass of the shared accumulator, ``C = spikes.at(1.0).lag_products(50)``
(float32, lags 0..50, sliced per chunk and never upcast as a whole), supplies
the correlogram of the unordered pair (i < j): bin k (k = 0..99) holds lag
l = k - 50; for l >= 0 the count is ``C[l][i, j]`` (i before j), for l < 0 it
is ``C[-l][j, i]`` (j before i).  The kernel is evaluated at the bin centres
t_k = k - 50 + 0.5 ms.  For each delay in ``delays_ms`` the MAP of
(a_0..a_99, J_+, J_-) is found by Newton's method (below), the delay with the
highest log-posterior is kept, and

    S[i, j] = J_+ * sqrt(T tau lambda_i lambda_j) / c_alpha  =  J_+ / J_thr
    S[j, i] = J_- * sqrt(T tau lambda_j lambda_i) / c_alpha  =  J_- / J_thr

so ``S`` is the coupling in units of the published significance threshold:
|S| > 1 is the paper's test at alpha = 0.001, positive = excitatory,
negative = inhibitory.  ``T`` is the total valid recording time of the pass
(``spikes.at(1.0).T`` bins x 1 ms), ``lambda = n_spikes / T``.  Every ordered
off-diagonal pair is scored (both directions of every unordered pair);
diagonal 0; finite everywhere.

Orientation (package invariant, see ``accumulate.py``)
-------------------------------------------------------
``S[i, j]`` is evidence for the directed edge i -> j: row = presynaptic,
column = postsynaptic.  ``C[l][i, j]`` counts the bins in which neuron i
fired l bins BEFORE neuron j; a negative lag is the transpose,
``ccg_ij(-l) = C[l][j, i]``.  ``J_+`` (kernel on positive lags, i before j)
is written to ``S[i, j]``; ``J_-`` (kernel on negative lags, j before i) to
``S[j, i]``.  Nothing in this module transposes.

Optimiser
---------
The log-posterior  L = sum_k [c_k eta_k - exp(eta_k)] - P sum_k (a_{k+1} - a_k)^2
with eta_k = a_k + J_+ f_k^+ + J_- f_k^- and P = beta / (2 DELTA) is strictly
concave in (a, J), so Newton's method with a backtracking line search reaches
the same MAP as the reference code's Levenberg-Marquardt iteration.  The
negative Hessian is block structured, [[A, U], [U^T, V]] with A = diag(mu) +
2 P D^T D tridiagonal (100 x 100), U (100 x 2) and V (2 x 2); each Newton
step solves A^-1 [g_a, U] by a Thomas sweep vectorised over a chunk of pairs
and closes with the 2 x 2 Schur complement.  J is projected onto
``j_bounds`` after every step (the reference code bounds J the same way);
a pair converges when the accepted step changes L by less than ``tol``, or
after ``max_iter`` steps; the next delay is warm-started from the previous
solution.  Steps that fail to increase L after ``max_halvings`` halvings are
rejected and the pair is treated as converged.

Label-free operating point
--------------------------
The published test: untyped = typed = |S| > 1 over all off-diagonal pairs
(i.e. |J| > J_thr at alpha = 0.001); E/I typing by the shipped rank rule on
``-S.sum(1)`` (fraction 0.25).  No lag profile is defined (the delay is a
per-pair model choice, not a per-lag evidence).

Deviations from the published method
------------------------------------
* Optimiser: Newton + backtracking line search on the concave log-posterior,
  vectorised over pairs, instead of Levenberg-Marquardt.  Same objective,
  same initialisation (a = log mean count, J = 0.1), same J bounds; the MAP
  is unique so the estimates agree up to the convergence tolerance.
* Binned correlogram: the reference code histograms continuous spike-time
  differences; here the correlogram is the shared 1 ms lag-product stack (a
  lag-l bin collects differences in (l - 1, l + 1) ms) and the kernel is
  evaluated at bin centres t_k = k - W + 0.5 ms.  Counts never bridge a
  recording (segment) boundary, and T is the total valid time of the pass.
* Transmission delays: 1..4 ms by default, the reference code's search set,
  while this network's conduction delays run 1.5-13.5 ms.  With tau = 4 ms
  the d <= 4 kernels still reach out to ~20 ms with decaying weight, but for
  a 10 ms synapse the excess lies where the kernel is small and is partly
  absorbed by the background, which dilutes J.  The registry variant
  ``glmcc_d14`` (delays 1..14 ms) is the documented mismatch probe.
* Delay selection compares the log-posterior (likelihood + smoothness prior)
  across delays; the paper speaks of the likelihood.  The prior term varies
  little between delays, so the choice is the same in practice.
* Score: the continuous J / J_thr ratio (so the threshold-free metrics can
  rank pairs) instead of the binary decision; the published decision is
  recovered exactly by ``label_free`` (|S| > 1).
* c_alpha = 5.16 is the constant of the paper / reference code for
  alpha = 0.001; ``alpha`` is recorded in PARAMS for the CSV but is not used
  to recompute c_alpha.
* Pairs in which either neuron never fires in the pass are not fitted:
  their J_thr is infinite, so S = 0 exactly.
* ``pair_subsample`` (default None, NOT in the paper): a seeded random
  subsample of the unordered pairs; only those are fitted, the rest stay 0.
  Provided as a runtime escape hatch for the sweep; off by default.
* Data scale: the paper fits 5400 s of 1000 HH neurons at ~several Hz; here
  a dataset is 50-200 recordings of 60 s at ~0.3 Hz mean rate, so counts per
  1 ms bin are low, the smoothness prior dominates the background, and
  J_thr exceeds the J bounds for most low-rate pairs (the published test can
  only fire on the higher-rate pairs).
"""
from __future__ import annotations

import time

import numpy as np

from baselines import common as cm
from baselines.labelfree import rank_typing

NAME = "glmcc"
TIER = 2
DIRECTED = True
CITATION = ("Kobayashi R, Kurita S, Kurth A, Kitano K, Mizuseki K, Diesmann M, Richmond BJ, "
            "Shinomoto S (2019) Reconstructing neuronal circuitry from parallel spike trains. "
            "Nat Commun 10:4468; reference code github.com/NII-Kobayashi/GLMCC.")
PARAMS = dict(
    bin_ms=cm.FINE_BIN_MS,   # DELTA: correlogram resolution (the shared 1 ms pass)
    window_ms=50,            # W: half-width of the correlogram window, [-W, W)
    tau_ms=4.0,              # synaptic time constant of the kernel (simulated data value)
    delays_ms=(1, 2, 3, 4),  # transmission delays tested; best log-posterior kept per pair
    gamma=5e-4,              # smoothness prior on a(t), ms^-1; beta = 2/gamma = 4000
    c_alpha=5.16,            # J_thr = c_alpha / sqrt(T tau lambda_i lambda_j)
    alpha=0.001,             # nominal level of c_alpha (documentary; see Deviations)
    j_bounds=(-3.0, 5.0),    # J constrained to this interval
    j_init=0.1,              # initial J (a is initialised at log mean count)
    a_floor=1e-6,            # floor on the mean count for the initial a = log(mean)
    max_iter=50,             # Newton iterations per delay
    tol=1e-4,                # convergence: |delta log-posterior| below this
    max_halvings=12,         # backtracking line search: step halvings before giving up
    newton_damping=1e-12,    # added to the Hessian diagonal (guards a zero pivot only)
    chunk=32768,             # unordered pairs per vectorised Newton chunk
    pair_subsample=None,     # None | int (number) | float (fraction) of unordered pairs to fit
    subsample_seed=1,        # seed of that subsample
    verbose=False,           # print per-chunk timing and the projected run time
)

#: diagnostics of the last ``score`` call (timing, convergence, delay usage)
LAST_INFO = {}


def _p(params):
    q = dict(PARAMS)
    q.update(params)
    return q


# ------------------------------------------------------------------------- #
# model pieces
# ------------------------------------------------------------------------- #
def _kernels(n_bins, bin_ms, window_ms, d_ms, tau_ms):
    """Bin centres t_k and the kernels f(t_k) (positive lags) / f(-t_k)."""
    t = np.arange(n_bins) * bin_ms - window_ms + 0.5 * bin_ms
    with np.errstate(over="ignore"):
        fp = np.where(t > d_ms, np.exp(-(t - d_ms) / tau_ms), 0.0)
        fm = np.where(-t > d_ms, np.exp(-(-t - d_ms) / tau_ms), 0.0)
    return t, fp, fm


def _pair_counts(C, ii, jj, n_lag):
    """Correlograms of the unordered pairs (ii[b], jj[b]) as ``[2 n_lag, B]``.

    Bin k <-> lag l = k - n_lag: ``C[l][i, j]`` for l >= 0 (i before j),
    ``C[-l][j, i]`` for l < 0 (j before i).  ``C`` is the shared float32
    ``[n_lag + 1, N, N]`` stack; only the gathered chunk is upcast.
    """
    pos = C[:n_lag, ii, jj]            # lags 0 .. n_lag-1       [n_lag, B]
    neg = C[1:n_lag + 1, jj, ii]       # lags -1 .. -n_lag       [n_lag, B]
    out = np.empty((2 * n_lag, len(ii)), np.float64)
    out[:n_lag] = neg[::-1]            # bin 0 = lag -n_lag ... bin n_lag-1 = lag -1
    out[n_lag:] = pos                  # bin n_lag = lag 0 ... bin 2 n_lag-1 = lag n_lag-1
    return out


def _logpost(cnt, a, Jp, Jm, fp, fm, P):
    """Log-posterior (up to constants), rate mu and background increments."""
    eta = a + fp[:, None] * Jp[None, :] + fm[:, None] * Jm[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        mu = np.exp(eta)
        da = a[1:] - a[:-1]
        L = (cnt * eta - mu).sum(0) - P * (da * da).sum(0)
    return L, mu, da


def _thomas(D, e, rhs):
    """Solve the tridiagonal systems (diag D[k, b], constant off-diagonal e)
    for every column b and every right-hand side m: ``rhs[k, b, m]``."""
    K = D.shape[0]
    cp = np.empty_like(D)
    rp = np.empty_like(rhs)
    cp[0] = e / D[0]
    rp[0] = rhs[0] / D[0][:, None]
    for k in range(1, K):
        denom = D[k] - e * cp[k - 1]
        cp[k] = e / denom
        rp[k] = (rhs[k] - e * rp[k - 1]) / denom[:, None]
    x = np.empty_like(rhs)
    x[-1] = rp[-1]
    for k in range(K - 2, -1, -1):
        x[k] = rp[k] - cp[k][:, None] * x[k + 1]
    return x


def _newton_direction(cnt, mu, da, fp, fm, P, damping):
    """Newton ascent direction (d_a [K, B], dJp [B], dJm [B]) from the
    block-tridiagonal negative Hessian [[A, U], [U^T, V]]."""
    K, B = cnt.shape
    r = cnt - mu
    lap = np.zeros_like(cnt)
    lap[1:] += da
    lap[:-1] -= da
    g_a = r - 2.0 * P * lap                       # gradient wrt a
    gJp = fp @ r                                  # gradient wrt J_+
    gJm = fm @ r
    lapdiag = np.full(K, 2.0)
    lapdiag[0] = lapdiag[-1] = 1.0
    D = mu + (2.0 * P) * lapdiag[:, None] + damping   # diag of A = diag(mu) + 2P D^T D
    e = -2.0 * P                                       # off-diagonal of A
    U0 = mu * fp[:, None]                              # U[:, 0]
    U1 = mu * fm[:, None]                              # U[:, 1]
    V00 = (fp * fp) @ mu
    V01 = (fp * fm) @ mu
    V11 = (fm * fm) @ mu
    rhs = np.stack([g_a, U0, U1], 2)                   # [K, B, 3]
    Y = _thomas(D, e, rhs)                             # A^-1 [g_a, U0, U1]
    Y0, Y1, Y2 = Y[:, :, 0], Y[:, :, 1], Y[:, :, 2]
    # Schur complement  Sc = V - U^T A^-1 U ;  rJ = g_J - U^T A^-1 g_a
    S00 = V00 - (U0 * Y1).sum(0)
    S01 = V01 - (U0 * Y2).sum(0)
    S11 = V11 - (U1 * Y2).sum(0)
    rJ0 = gJp - (U0 * Y0).sum(0)
    rJ1 = gJm - (U1 * Y0).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        det = S00 * S11 - S01 * S01
        dJp = (S11 * rJ0 - S01 * rJ1) / det
        dJm = (S00 * rJ1 - S01 * rJ0) / det
    bad = ~(np.isfinite(dJp) & np.isfinite(dJm))
    if bad.any():                                      # degenerate pair: no move
        dJp[bad] = 0.0
        dJm[bad] = 0.0
    d_a = Y0 - Y1 * dJp[None, :] - Y2 * dJm[None, :]
    d_a[:, ~np.isfinite(d_a).all(0)] = 0.0
    return d_a, dJp, dJm


def _fit_delay(cnt, a, Jp, Jm, fp, fm, P, lo, hi, max_iter, tol, max_halvings, damping):
    """MAP of (a, J_+, J_-) for one delay, vectorised over the chunk.

    In place on ``a``, ``Jp``, ``Jm`` (warm start).  Returns (L, n_iter[B]).
    """
    K, B = cnt.shape
    L, mu, da = _logpost(cnt, a, Jp, Jm, fp, fm, P)
    done = np.zeros(B, bool)
    n_iter = np.zeros(B, np.int32)
    for it in range(max_iter):
        act = np.flatnonzero(~done)
        if not len(act):
            break
        n_iter[act] += 1
        c_s, a_s, mu_s, da_s = cnt[:, act], a[:, act], mu[:, act], da[:, act]
        Jp_s, Jm_s, L_s = Jp[act], Jm[act], L[act]
        d_a, dJp, dJm = _newton_direction(c_s, mu_s, da_s, fp, fm, P, damping)
        step = np.ones(len(act))
        pending = np.ones(len(act), bool)
        conv = np.zeros(len(act), bool)
        for _ in range(max_halvings + 1):
            pi = np.flatnonzero(pending)
            s = step[pi]
            a_t = a_s[:, pi] + s[None, :] * d_a[:, pi]
            Jp_t = np.clip(Jp_s[pi] + s * dJp[pi], lo, hi)
            Jm_t = np.clip(Jm_s[pi] + s * dJm[pi], lo, hi)
            L_t, mu_t, da_t = _logpost(c_s[:, pi], a_t, Jp_t, Jm_t, fp, fm, P)
            L_old = L_s[pi]
            ok = L_t >= L_old - 1e-9 * (1.0 + np.abs(L_old))
            ok &= np.isfinite(L_t)
            if ok.any():
                acc = pi[ok]
                a_s[:, acc] = a_t[:, ok]
                mu_s[:, acc] = mu_t[:, ok]
                da_s[:, acc] = da_t[:, ok]
                Jp_s[acc] = Jp_t[ok]
                Jm_s[acc] = Jm_t[ok]
                conv[acc] = np.abs(L_t[ok] - L_old[ok]) < tol
                L_s[acc] = L_t[ok]
                pending[acc] = False
            step[pi[~ok]] *= 0.5
            if not pending.any():
                break
        conv[pending] = True          # no ascent found: stuck (at a bound), stop
        a[:, act], mu[:, act], da[:, act] = a_s, mu_s, da_s
        Jp[act], Jm[act], L[act] = Jp_s, Jm_s, L_s
        done[act] = conv
    return L, n_iter


def _fit_pairs(C, ii, jj, p):
    """Fit every unordered pair (ii[b], jj[b]) over all delays.

    Returns dict(Jp, Jm, delay, L, n_iter, elapsed) for the chunk; ``Jp`` is
    the coupling on positive lags (ii before jj), ``Jm`` on negative lags.
    """
    t0 = time.time()
    n_lag = int(round(p["window_ms"] / p["bin_ms"]))
    K = 2 * n_lag
    P = (2.0 / p["gamma"]) / (2.0 * p["bin_ms"])          # beta / (2 DELTA)
    lo, hi = float(p["j_bounds"][0]), float(p["j_bounds"][1])
    cnt = _pair_counts(C, ii, jj, n_lag)                   # [K, B]
    B = cnt.shape[1]
    a = np.log(np.maximum(cnt.mean(0), p["a_floor"]))[None, :].repeat(K, 0)
    Jp = np.full(B, float(p["j_init"]))
    Jm = np.full(B, float(p["j_init"]))
    best_L = np.full(B, -np.inf)
    best_Jp = np.zeros(B)
    best_Jm = np.zeros(B)
    best_d = np.full(B, np.nan)
    iters = np.zeros(B, np.int32)
    for d in p["delays_ms"]:
        _, fp, fm = _kernels(K, p["bin_ms"], p["window_ms"], float(d), p["tau_ms"])
        L, n_it = _fit_delay(cnt, a, Jp, Jm, fp, fm, P, lo, hi, int(p["max_iter"]),
                             float(p["tol"]), int(p["max_halvings"]), float(p["newton_damping"]))
        iters += n_it
        better = L > best_L
        best_L[better] = L[better]
        best_Jp[better] = Jp[better]
        best_Jm[better] = Jm[better]
        best_d[better] = float(d)
    return dict(Jp=best_Jp, Jm=best_Jm, delay=best_d, L=best_L, n_iter=iters,
                elapsed=time.time() - t0)


def _unordered_pairs(n, active, p):
    """Unordered pairs (i < j) with both neurons active, optionally subsampled."""
    ii, jj = np.triu_indices(n, 1)
    keep = active[ii] & active[jj]
    ii, jj = ii[keep], jj[keep]
    sub = p["pair_subsample"]
    if sub is not None:
        m = len(ii)
        k = int(round(sub * m)) if isinstance(sub, float) and sub <= 1.0 else int(sub)
        k = max(0, min(k, m))
        rng = np.random.default_rng(int(p["subsample_seed"]))
        sel = np.sort(rng.choice(m, k, replace=False))
        ii, jj = ii[sel], jj[sel]
    return ii, jj


# ------------------------------------------------------------------------- #
# contract
# ------------------------------------------------------------------------- #
def score(spikes, n_neurons, **params):
    p = _p(params)
    t_start = time.time()
    sd = spikes.at(p["bin_ms"])
    n = sd.n_neurons
    n_lag = int(round(p["window_ms"] / p["bin_ms"]))
    C = sd.lag_products(n_lag)                              # shared float32 [n_lag+1, N, N]
    T_s = sd.T * p["bin_ms"] * 1e-3                         # valid recording time, s
    lam = sd.n_spikes / T_s                                 # Hz
    tau_s = p["tau_ms"] * 1e-3
    scale = np.sqrt(T_s * tau_s * np.outer(lam, lam)) / p["c_alpha"]   # 1 / J_thr
    active = sd.n_spikes > 0
    ii, jj = _unordered_pairs(n, active, p)
    S = np.zeros((n, n), np.float64)
    chunk = max(int(p["chunk"]), 1)
    n_pairs = len(ii)
    info = dict(n_pairs=int(n_pairs), n_ordered=int(n * (n - 1)), chunks=0, fit_s=0.0,
                n_iter_mean=0.0, at_lower=0, at_upper=0, delay_hist={})
    iters_sum = 0
    for start in range(0, n_pairs, chunk):
        sl = slice(start, min(start + chunk, n_pairs))
        r = _fit_pairs(C, ii[sl], jj[sl], p)
        S[ii[sl], jj[sl]] = r["Jp"] * scale[ii[sl], jj[sl]]
        S[jj[sl], ii[sl]] = r["Jm"] * scale[jj[sl], ii[sl]]
        info["chunks"] += 1
        info["fit_s"] += r["elapsed"]
        iters_sum += int(r["n_iter"].sum())
        lo, hi = p["j_bounds"]
        info["at_lower"] += int((r["Jp"] <= lo).sum() + (r["Jm"] <= lo).sum())
        info["at_upper"] += int((r["Jp"] >= hi).sum() + (r["Jm"] >= hi).sum())
        for d, c in zip(*np.unique(r["delay"], return_counts=True)):
            info["delay_hist"][float(d)] = info["delay_hist"].get(float(d), 0) + int(c)
        if p["verbose"] and info["chunks"] == 1:
            per_k = 1e3 * r["elapsed"] / max(sl.stop - sl.start, 1)
            print("  glmcc: first chunk %d pairs in %.1fs (%.2fs per 1000 pairs) -> projected "
                  "%.0fs for %d pairs" % (sl.stop - sl.start, r["elapsed"], per_k,
                                          per_k * n_pairs / 1e3, n_pairs), flush=True)
    info["n_iter_mean"] = iters_sum / max(n_pairs, 1)
    info["elapsed_s"] = time.time() - t_start
    np.fill_diagonal(S, 0.0)
    S[~np.isfinite(S)] = 0.0
    LAST_INFO.clear()
    LAST_INFO.update(info)
    if p["verbose"]:
        print("  glmcc: %d pairs, %d chunks, fit %.0fs, mean Newton iters %.1f, J at bounds "
              "lo %d / hi %d, delays %s" % (n_pairs, info["chunks"], info["fit_s"],
                                             info["n_iter_mean"], info["at_lower"],
                                             info["at_upper"], info["delay_hist"]), flush=True)
    return S


def label_free(spikes, n_neurons, S, **params):
    """The published test: |J| > J_thr(alpha)  <=>  |S| > 1."""
    p = _p(params)
    n = S.shape[0]
    off = cm.offdiag(n)
    pred = (np.abs(np.asarray(S, np.float64)) > 1.0) & off
    inh = rank_typing(-np.asarray(S, np.float64).sum(1))
    info = dict(threshold=1.0, alpha=float(p["alpha"]), c_alpha=float(p["c_alpha"]),
                n_pred=int(pred.sum()))
    typed_info = dict(info, n_inh_typed=int(inh.sum()))
    return dict(untyped=(pred, info), typed=(pred.copy(), typed_info), typing_inh=inh,
                rule="GLMCC |J| > J_thr(alpha=%g)" % p["alpha"])
