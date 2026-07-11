"""Fine-resolution multivariate GLM connectivity inference.

Motivation
----------
The CCG baseline scores each ``(pre, post)`` pair *independently*, so it cannot
tell a real synapse from two neurons that merely co-activate in the same burst
(common input). This module instead estimates each neuron's activity as a
*joint* linear function of every other neuron's recent spikes, so shared drive
is discounted and the fitted weight ``W[pre, post]`` reflects the *direct*
coupling. Two things make it work on NEURON output:

1. **Fine re-binning.** NEURON sessions are resampled at 10 Hz (100 ms), far too
   coarse for the ~1.5-30 ms synaptic delays. We re-bin the raw ``spike_times``
   at ``bin_ms`` (default 5 ms) so the short-latency window is monosynaptic.
2. **A causal window feature.** For each presynaptic neuron we sum its spikes in
   the ``1..max_lag`` bins *before* each target bin (never crossing recording
   boundaries), then solve one regularized least-squares for all targets at once.

On a 236-neuron normal-state session (3 x 60 s) this lifts candidate-set AUC from
~0.73 (CCG) to ~0.87, and recovers ~50% of edges at precision 0.8 (vs ~2% for
CCG). Recall scales strongly with recording time -- use many recordings.

The weight is *signed*: ``W[pre, post] > 0`` excitatory coupling, ``< 0``
inhibitory. Score excitatory edges by ``+W``, inhibitory edges by ``-W``.

Only numpy / scipy / scikit-learn are required (no torch).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading + re-binning
# --------------------------------------------------------------------------- #
def load_session(session_dir):
    """Load ground-truth network + per-recording raw spike times."""
    net_path = sorted(glob.glob(os.path.join(session_dir, "network_*.npz")))[0]
    net = np.load(net_path, allow_pickle=True)
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
    rec_paths = sorted(glob.glob(os.path.join(session_dir, "recording*.npz")))
    rec_paths = [p for p in rec_paths if "raster" not in os.path.basename(p)]
    recs = []
    for rp in rec_paths:
        d = np.load(rp, allow_pickle=True)
        recs.append((d["spike_times"], float(d["duration"])))
    return {
        "n_neurons": n,
        "is_inhibitory": inh,
        "positions": net["neuron_positions"],
        "cluster_assignments": net["cluster_assignments"],
        "A_exc": A_exc,
        "A_inh": A_inh,
        "recordings": recs,
    }


def build_spike_matrix(recordings, n_neurons, bin_ms):
    """Re-bin raw spike times into one concatenated ``[N, T]`` matrix.

    Returns the matrix and the segment ``boundaries`` (start bin of each
    recording, plus the total), so features never cross recordings.
    """
    mats, boundaries = [], [0]
    for spike_times, duration in recordings:
        T = int(duration / bin_ms)
        m = np.zeros((n_neurons, T), np.float32)
        for i in range(n_neurons):
            t = np.atleast_1d(np.asarray(spike_times[i], float))
            if len(t):
                m[i, np.clip((t / bin_ms).astype(int), 0, T - 1)] = 1.0
        mats.append(m)
        boundaries.append(boundaries[-1] + T)
    return np.concatenate(mats, axis=1), boundaries


# --------------------------------------------------------------------------- #
# The GLM
# --------------------------------------------------------------------------- #
def _causal_window(spike_matrix, boundaries, max_lag):
    """Sum of each neuron's spikes in the 1..max_lag bins before each bin,
    zeroing the leaked bins at every recording start."""
    F = np.zeros_like(spike_matrix)
    for lag in range(1, max_lag + 1):
        shifted = np.zeros_like(spike_matrix)
        shifted[:, lag:] = spike_matrix[:, :-lag]
        for b in boundaries[1:-1]:
            shifted[:, b:b + lag] = 0.0
        F += shifted
    return F


def glm_connectivity(spike_matrix, boundaries, max_lag=4, l2=2.0):
    """Signed direct-coupling matrix ``W[pre, post]`` via one ridge solve.

    Solves ``W = (F Fᵀ + l2 I)^{-1} (F Mᵀ)`` where ``F`` is the causal-window
    feature and ``M`` the spike matrix; ``W[i, j]`` is the coupling from neuron
    ``i`` onto neuron ``j``. Self-coupling (the diagonal) is zeroed.
    """
    n = spike_matrix.shape[0]
    F = _causal_window(spike_matrix, boundaries, max_lag)
    G = F @ F.T + l2 * np.eye(n)
    W = np.linalg.solve(G, F @ spike_matrix.T)
    np.fill_diagonal(W, 0.0)
    return W


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def spatial_candidates(positions, radius=None, edge_mask=None, percentile=90.0):
    """Boolean ``[N, N]`` candidate mask of pairs within ``radius`` (off-diagonal).

    If ``radius`` is None it is set to the ``percentile`` of true-edge distances
    (needs ``edge_mask``), which keeps ~that fraction of real edges as candidates.
    """
    n = positions.shape[0]
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    off = ~np.eye(n, dtype=bool)
    if radius is None:
        radius = np.percentile(d[edge_mask & off], percentile) if edge_mask is not None else d.max()
    return (d < radius) & off, float(radius)


def evaluate(W, candidates, A_exc, A_inh, target_precision=0.8):
    """AUC (exc + inh) over candidates, plus a confusion matrix at the smallest
    excitatory threshold reaching ``target_precision``."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    out = {"n_candidates": int(candidates.sum())}
    ye, yi = A_exc[candidates], A_inh[candidates]
    se, si = W[candidates], -W[candidates]  # inhibitory: more negative -> more inhibitory
    if ye.any() and (~ye).any():
        out["auc_excitatory"] = float(roc_auc_score(ye, se))
    if yi.any() and (~yi).any():
        out["auc_inhibitory"] = float(roc_auc_score(yi, si))

    P, R, T = precision_recall_curve(ye, se)
    ok = np.where(P[:-1] >= target_precision)[0]
    thr = float(T[ok[0]]) if len(ok) else float(T[-1])
    pred = (W > thr) & candidates
    TP = int((pred & A_exc).sum())
    FP = int((pred & ~A_exc & candidates).sum())
    FN = int((~pred & A_exc & candidates).sum())
    TN = int((~pred & ~A_exc & candidates).sum())
    out.update({
        "threshold": thr, "target_precision": target_precision,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": TP / (TP + FP) if TP + FP else 0.0,
        "recall": TP / (TP + FN) if TP + FN else 0.0,
    })
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(session_dir, bin_ms=5.0, max_lag=4, l2=2.0, candidate_radius=None,
        target_precision=0.8, save=True):
    """Full pipeline: load -> re-bin -> GLM -> evaluate -> (save)."""
    s = load_session(session_dir)
    M, bnd = build_spike_matrix(s["recordings"], s["n_neurons"], bin_ms)
    W = glm_connectivity(M, bnd, max_lag=max_lag, l2=l2)
    cand, radius = spatial_candidates(s["positions"], candidate_radius,
                                      edge_mask=(s["A_exc"] | s["A_inh"]))
    metrics = evaluate(W, cand, s["A_exc"], s["A_inh"], target_precision)
    metrics["bin_ms"] = bin_ms
    metrics["candidate_radius"] = radius
    metrics["total_spikes"] = int(M.sum())
    if save:
        out = os.path.join(session_dir, f"glm_connectivity_{int(bin_ms)}ms.npz")
        np.savez_compressed(out, W=W, candidates=cand,
                            A_exc=s["A_exc"], A_inh=s["A_inh"])
        metrics["saved"] = out
    return W, metrics


def build_parser():
    p = argparse.ArgumentParser(description="Fine-resolution GLM connectivity inference")
    p.add_argument("session_dir")
    p.add_argument("--bin-ms", type=float, default=5.0)
    p.add_argument("--max-lag", type=int, default=4)
    p.add_argument("--l2", type=float, default=2.0)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--target-precision", type=float, default=0.8)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    W, m = run(a.session_dir, bin_ms=a.bin_ms, max_lag=a.max_lag, l2=a.l2,
               candidate_radius=a.radius, target_precision=a.target_precision)
    print(f"GLM connectivity @ {m['bin_ms']}ms | {m['total_spikes']} spikes | "
          f"{m['n_candidates']} candidates (radius {m['candidate_radius']:.1f})")
    print(f"  AUC excitatory = {m.get('auc_excitatory'):.3f} | "
          f"inhibitory = {m.get('auc_inhibitory'):.3f}")
    print(f"  @precision>={m['target_precision']}: "
          f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']} "
          f"(precision {m['precision']:.2f}, recall {m['recall']:.2f})")
