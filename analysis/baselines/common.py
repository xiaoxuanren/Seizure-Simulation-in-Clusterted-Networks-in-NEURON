"""Session-level data access shared by the runner and every method module.

A :class:`Session` is one dataset (``<session>/<state>``, optionally the first
``n_recordings`` recordings, optionally with burst windows excluded).  It
hands out :class:`SpikeData` passes at any bin width and caches the expensive
accumulations (lag products, ridge Gram, ridge solve, jitter surrogates) so
that every method reads the SAME sufficient statistics and the SAME
surrogate realisations.

Nothing in this module reads ground truth.  Burst-window detection uses
spikes only (the project's 0.35 participation gate).

Orientation of every (N, N) array: ``[pre, post]`` -- see ``accumulate.py``.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import accumulate as acc  # noqa: E402
from session_paths import resolve, results_dir, DATA  # noqa: E402

#: the project's burst definition for the burst-excluded arm
BURST_GATE = 0.35
BURST_BURN_IN_MS = 0.0
PAD_MS = 25.0

#: the shipped GLM operating point (glm_fit_current.py / chtc/analysis_one.py)
GLM_BIN_MS, GLM_MAX_LAG, GLM_L2, GLM_KSUM = 5.0, 6, 2.0, 4
JITTER_MS, N_SURR, SEED = 25.0, 8, 1
TARGET_FDR = 0.70
TYPING_LAGS, TYPING_FRACTION = 2, 0.25

#: the fine-timing pass
FINE_BIN_MS, FINE_MAX_LAG = 1.0, 50


class SpikeData:
    """One binned pass of a dataset: sparse binary ``[N, T]`` + boundaries."""

    def __init__(self, M, bnd, bin_ms):
        self.M = M.tocsr()
        self.bnd = list(int(b) for b in bnd)
        self.bin_ms = float(bin_ms)
        self.n_neurons = int(M.shape[0])
        self.T = int(self.bnd[-1])
        self.n_segments = len(self.bnd) - 1
        self.n_spikes = np.asarray(self.M.sum(1)).ravel().astype(np.float64)
        self._C, self._C_maxlag = None, -1
        self._gram, self._B = {}, {}

    # -- accumulations (cached) ------------------------------------------
    def lag_products(self, max_lag):
        """``C[l][pre, post]`` for l = 0..max_lag (float32 counts)."""
        if self._C is None or self._C_maxlag < max_lag:
            self._C = acc.lag_products(self.M, self.bnd, max_lag)
            self._C_maxlag = max_lag
        return self._C[:max_lag + 1]

    def valid_pairs(self, lags):
        return acc.valid_pairs_per_lag(self.bnd, lags)

    def gram(self, max_lag):
        if max_lag not in self._gram:
            self._gram[max_lag] = acc.gram(self.M, self.bnd, max_lag)
        return self._gram[max_lag]

    def ridge_B(self, max_lag, l2):
        key = (max_lag, float(l2))
        if key not in self._B:
            G, RHS = self.gram(max_lag)
            self._B[key] = acc.solve_ridge(G, RHS, l2, max_lag)
        return self._B[key]

    def release(self):
        """Drop cached dense arrays (the runner calls this between arms)."""
        self._C, self._C_maxlag = None, -1
        self._gram.clear()
        self._B.clear()

    # -- derived passes ----------------------------------------------------
    def jittered(self, jitter_bins, rng):
        return SpikeData(acc.jitter(self.M, self.bnd, jitter_bins, rng), self.bnd, self.bin_ms)

    def trimmed(self, windows_bins):
        M2, bnd2, dropped = acc.trim_windows(self.M, self.bnd, windows_bins)
        return SpikeData(M2, bnd2, self.bin_ms), dropped


class Session:
    """All passes of one dataset, sharing surrogate realisations.

    ``exclude_windows_ms`` (per recording, ``(k, 2)`` ms) switches on the
    burst-excluded arm: the windows, padded by ``pad_ms``, are dropped from
    every pass and each retained interval becomes its own segment.
    """

    def __init__(self, session_dir, n_recordings=None, exclude_windows_ms=None,
                 pad_ms=PAD_MS, label=None):
        self.session_dir = session_dir
        self.n_recordings = n_recordings
        self.exclude_windows_ms = exclude_windows_ms
        self.pad_ms = float(pad_ms)
        self.label = label or os.path.basename(session_dir)
        self._passes, self._surrogates, self._dropped = {}, {}, {}

    @property
    def arm(self):
        return "full" if self.exclude_windows_ms is None else "burst_excluded"

    def at(self, bin_ms):
        """The :class:`SpikeData` pass at ``bin_ms`` (loaded once, cached)."""
        key = float(bin_ms)
        if key not in self._passes:
            M, bnd = acc.load_session(self.session_dir, key, self.n_recordings)
            sd = SpikeData(M, bnd, key)
            if self.exclude_windows_ms is not None:
                wb = acc.windows_ms_to_bins(self.exclude_windows_ms, key, self.pad_ms)
                sd, dropped = sd.trimmed(wb)
                self._dropped[key] = (dropped, int(bnd[-1]))
            self._passes[key] = sd
        return self._passes[key]

    @property
    def n_neurons(self):
        return self.at(GLM_BIN_MS).n_neurons

    def dropped(self, bin_ms):
        """(dropped bins, total bins) of the exclusion at this bin width."""
        self.at(bin_ms)
        return self._dropped.get(float(bin_ms), (0, self.at(bin_ms).T))

    def surrogates(self, bin_ms, jitter_ms=JITTER_MS, n_surr=N_SURR, seed=SEED):
        """Spike-jitter surrogates of the ``bin_ms`` pass.

        Same generator stream as the shipped pipeline
        (``rng = default_rng(seed)`` then ``n_surr`` sequential draws of
        ``sparse_glm.jitter``), so for the 5 ms pass these are the exact
        surrogate matrices behind the shipped jitter null.
        """
        key = (float(bin_ms), float(jitter_ms), int(n_surr), int(seed))
        if key not in self._surrogates:
            base = self.at(bin_ms)
            jb = max(1, int(round(jitter_ms / bin_ms)))
            rng = np.random.default_rng(seed)
            self._surrogates[key] = [base.jittered(jb, rng) for _ in range(n_surr)]
        return self._surrogates[key]

    def excluded(self, windows_ms, pad_ms=PAD_MS):
        return Session(self.session_dir, self.n_recordings, windows_ms, pad_ms,
                       label=self.label)

    def release(self):
        for sd in self._passes.values():
            sd.release()
        for lst in self._surrogates.values():
            for sd in lst:
                sd.release()


# ------------------------------------------------------------------------- #
# burst windows (spikes only)
# ------------------------------------------------------------------------- #
def _load_burst_detector():
    """``neuron_simulation.analysis`` loaded by path: the package __init__
    imports NEURON, which this numpy-only code does not need."""
    spec = importlib.util.spec_from_file_location(
        "_nsim_analysis", os.path.join(REPO, "neuron_simulation", "analysis.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.detect_network_bursts


def recording_paths(session_dir):
    """Same ordering as ``sparse_glm.load_session``."""
    return [p for p in sorted(glob.glob(os.path.join(session_dir, "recording*.npz")))
            if "raster" not in os.path.basename(p)]


def burst_windows(session_dir, participation=BURST_GATE, burn_in_ms=BURST_BURN_IN_MS,
                  n_recordings=None, cache_path=None):
    """Per-recording ``(k, 2)`` arrays of ``(start_ms, end_ms)`` burst windows,
    RECOMPUTED with ``neuron_simulation.analysis.detect_network_bursts`` at
    the project's participation gate (stored windows are never loaded)."""
    paths = recording_paths(session_dir)
    if n_recordings is not None:
        paths = paths[:n_recordings]
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            c = json.load(fh)
        if (c.get("participation") == participation and c.get("burn_in_ms") == burn_in_ms
                and len(c["windows"]) >= len(paths)):
            return [np.asarray(w, float).reshape(-1, 2) for w in c["windows"][:len(paths)]]
    detect = _load_burst_detector()
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        st = d["spike_times"]
        spike_data = {i: np.atleast_1d(np.asarray(st[i], float)) for i in range(len(st))}
        bursts = detect(spike_data, len(st), float(d["duration"]),
                        participation_threshold=participation, burn_in_ms=burn_in_ms)
        out.append(np.array([[b["start_ms"], b["end_ms"]] for b in bursts], float).reshape(-1, 2))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(dict(participation=participation, burn_in_ms=burn_in_ms,
                           n_recordings=len(paths),
                           windows=[w.tolist() for w in out]), fh)
    return out


# ------------------------------------------------------------------------- #
# paths
# ------------------------------------------------------------------------- #
def dataset_dir(session, state):
    return resolve(session, state)


def baselines_results_dir(session, state, create=True):
    return results_dir(session, state, "baselines", create=create)


def summary_dir(create=True):
    d = os.path.join(DATA, "sweep_summary", "baselines")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def sweep_sessions():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DATA, "sweep_c[45]0_seed*")))


def offdiag(n):
    return ~np.eye(n, dtype=bool)
