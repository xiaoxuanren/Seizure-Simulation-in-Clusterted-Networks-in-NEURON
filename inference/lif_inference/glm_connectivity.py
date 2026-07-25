"""Fine-resolution multivariate GLM connectivity inference (sum_k readout: fit wide, read narrow).

What changed vs the summed-window version
-----------------------------------------
1. **Lag-resolved fit, per-edge peak readout.** We fit a separate coefficient for
   each presynaptic lag (1..``max_lag``) jointly, then reduce to one score per
   edge. The original code summed lags 1..``max_lag`` into a single feature,
   which averages the sharp monosynaptic signal with lags that are mostly
   common-input / polysynaptic noise; ``readout='lag1'`` fixed that by reading
   only the lag-1 coefficient.

   ``readout='sum4'`` (the current default, family ``sum_k``) goes further:
   fit wide (lags 1..``max_lag``) but read narrow (sum only lags 1..k). The wide
   joint fit absorbs slow common-input structure so it does not leak into the
   short lags, while the narrow readout drops the late lags (5-6) that carry no
   signal (per-lag AUC ~0.5-0.6). This is the best excitatory *ranking* measured
   on the flagship (k=4: AP 0.873, vs peak 0.834, sum-all 0.860, lag1 0.447).

   ``readout='peak'`` picks the largest-|coefficient| lag *per edge*; it ranks
   slightly worse than ``sum4`` but uniquely recovers the per-edge conduction
   delay (peak lag vs true delay r=0.807), so keep it for that analysis.

   IMPORTANT -- readout interacts with the FDR rule (see 2). The spike-jitter
   null is conservative, and integrating readouts ('sum'/'sum_k') span 0-20 ms
   so they are far more exposed to the surrogate common-drive inflation than the
   sharp 'peak'. On the flagship, at the nominal ``target_fdr=0.10`` the realized
   FDR is 0.043 for 'peak' but only ~0.001 for 'sum4' -- i.e. 'sum4' at the
   default target runs *very* tight and returns fewer edges than 'peak' does.
   The better ranking only turns into more recovered edges once the target is set
   against the realized FDR: use ``calibrate_fdr`` / ``--calibrate`` to do this.
   At matched realized FDR ~0.10, 'sum4' beats 'peak' by roughly +500 TP.

       readout  target  realFDR  TP     FP     P      R      F1
       peak     0.10     0.043   7009    314   0.957  0.525  0.678
       peak     0.20     0.097   7811    838   0.903  0.585  0.710
       sum4     0.10     0.001   3691      4   0.999  0.276  0.433   <- too tight
       sum4     0.50     0.053   7724    430   0.947  0.578  0.718
       sum4     0.70     0.126   8787   1271   0.874  0.658  0.751

   ``readout='lag1'`` and ``readout='sum'`` restore the older behaviours.

   NOTE: ``max_lag`` defaults to 6 -- wide enough to contain the delay spread;
   ``sum_k`` reads only the first k of those lags (k=4 default).

2. **Label-free operating point (jitter null).** The threshold is chosen so the
   expected number of exceedances under a *spike-jitter* surrogate divided by the
   observed exceedances is <= ``target_fdr``. Jittering each spike by +/-``jitter_ms``
   preserves the coarse burst/rate envelope but destroys the 1..``max_lag``-bin
   synaptic window, so the excess above it is genuine fine-timescale coupling.
   This replaces the old oracle-precision threshold (which needed ground truth)
   and is calibrated: realized FDR tracks the target on both regimes. The old
   circular-shift surrogate is *not* used here -- it destroys common input and
   under-estimates the null (realized FDR ~0.75 at target 0.10).

3. **E/I typing from sign, no cell-type labels.** A neuron is called inhibitory
   from the sign of its net outgoing weight (AUC ~0.86-0.91 vs truth). Inhibitory
   edges are then scored (by ``-W``) only among inferred-inhibitory presynaptic
   neurons, which is what makes the inhibitory layer usable.

4. **No ground-truth leakage.** The pipeline runs from spikes alone and needs no
   ``network_*.npz``; it produces a signed weight matrix, a predicted directed
   adjacency (excitatory + type-constrained inhibitory layers), and the inferred
   neuron types. If a network file *is* present, an evaluation block reports
   AUC/AP and the recovered-topology confusion for convenience. The candidate set
   defaults to whole-map (no spatial prior); pass ``--radius`` for a geometry
   prior (whole-map and a matched geometry radius score within ~1%).

Only numpy is required for inference; scikit-learn is used only in the optional
evaluation block. No torch.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# Data loading (spikes only -- works with no network file)
# --------------------------------------------------------------------------- #
def load_spikes(session_dir):
    """Load per-recording raw spike times and (if present) neuron positions.

    Requires only ``recording*.npz`` files. ``neuron_positions`` is read from a
    ``network_*.npz`` if one exists, otherwise returned as ``None`` (whole-map
    candidates). No connectivity ground truth is read here.
    """
    rec_paths = sorted(glob.glob(os.path.join(session_dir, "recording*.npz")))
    rec_paths = [p for p in rec_paths if "raster" not in os.path.basename(p)]
    if not rec_paths:
        raise FileNotFoundError(f"no recording*.npz in {session_dir}")
    recs = []
    for rp in rec_paths:
        d = np.load(rp, allow_pickle=True)
        recs.append((d["spike_times"], float(d["duration"])))
    n_neurons = len(recs[0][0])

    positions = None
    net_files = sorted(glob.glob(os.path.join(session_dir, "network_*.npz")))
    if net_files:
        net = np.load(net_files[0], allow_pickle=True)
        if "neuron_positions" in net.files:
            positions = net["neuron_positions"]
    return {"n_neurons": n_neurons, "positions": positions, "recordings": recs}


def load_ground_truth(session_dir):
    """Load ground-truth exc/inh adjacency for *evaluation only*, or ``None``.

    Returns a dict with boolean ``A_exc``/``A_inh`` (``[N, N]``, indexed
    ``[pre, post]``) and ``is_inhibitory`` when a ``network_*.npz`` is present,
    otherwise ``None``. Never used by the inference path.
    """
    net_files = sorted(glob.glob(os.path.join(session_dir, "network_*.npz")))
    if not net_files:
        return None
    net = np.load(net_files[0], allow_pickle=True)
    if "connections" not in net.files:
        return None
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


def build_spike_matrix(recordings, n_neurons, bin_ms):
    """Re-bin raw spike times into one concatenated ``[N, T]`` matrix.

    Returns the matrix and the segment ``boundaries`` (start bin of each
    recording, plus the total) so features never cross recordings.
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
# The GLM (lag-resolved, per-edge peak readout)
# --------------------------------------------------------------------------- #
def _lagged_features(spike_matrix, boundaries, max_lag):
    """Stack per-lag shifted spike matrices into ``[max_lag * N, T]``.

    Row block ``k`` (0-indexed) holds every neuron's spikes shifted by lag
    ``k + 1`` bins, with the leaked bins at each recording start zeroed.
    """
    n = spike_matrix.shape[0]
    feats = np.zeros((max_lag * n, spike_matrix.shape[1]), np.float32)
    for k in range(max_lag):
        lag = k + 1
        sh = feats[k * n:(k + 1) * n]
        sh[:, lag:] = spike_matrix[:, :-lag]
        for b in boundaries[1:-1]:
            sh[:, b:b + lag] = 0.0
    return feats


def glm_connectivity(spike_matrix, boundaries, max_lag=6, l2=2.0, readout="sum4"):
    """Signed direct-coupling matrix ``W[pre, post]`` via one lag-resolved ridge.

    Fits ``B[lag, pre, post]`` jointly across lags 1..``max_lag`` with a single
    ridge solve, then reduces to a ``[N, N]`` score:

    * ``readout='sum4'`` (default, family ``sum_k``) -- fit wide (``max_lag``),
      read narrow: sum only lags 1..k. The wide joint fit absorbs slow
      common-input structure while the narrow readout drops the late lags (5-6)
      that are pure noise. Best excitatory ranking measured on the flagship
      (k=4: AP 0.873 vs peak 0.834, sum-all 0.860). NOTE: integrating readouts
      make the jitter null more conservative than ``peak`` does, so the nominal
      ``target_fdr`` runs tighter than its face value -- use ``calibrate_fdr``
      to choose the target against the *realized* FDR.
    * ``readout='peak'`` -- the signed coefficient at the largest-|coef| lag,
      chosen per edge. Sharp-timing; recovers per-edge conduction delay (peak
      lag vs true delay r=0.807), so keep it for the delay-recovery analysis.
    * ``readout='lag1'`` -- the lag-1 coefficient ``B[0]`` only (correct when
      every edge shares a 0-5 ms delay; discards edges that peak at lag 2).
    * ``readout='sum'`` -- the sum over *all* lags (original behaviour;
      == ``sum_k`` with k=``max_lag``).

    ``W[i, j] > 0`` excitatory coupling, ``< 0`` inhibitory. Self-coupling
    (diagonal) is zeroed. Score excitatory edges by ``+W``, inhibitory by ``-W``.
    """
    n = spike_matrix.shape[0]
    F = _lagged_features(spike_matrix, boundaries, max_lag)          # [max_lag*N, T]
    G = F @ F.T + l2 * np.eye(F.shape[0])
    B = np.linalg.solve(G, F @ spike_matrix.T.astype(np.float32))    # [max_lag*N, N]
    B = B.reshape(max_lag, n, n)                                     # [lag, pre, post]

    readout = str(readout).strip().lower()
    if readout == "lag1":
        W = B[0].copy()
    elif readout == "sum":
        W = B.sum(0)
    elif readout == "peak":
        idx = np.abs(B).argmax(0)
        W = np.take_along_axis(B, idx[None], 0)[0]
    elif readout.startswith("sum"):
        # 'sum_k' / 'sumN': fit wide (max_lag), read narrow -- sum only lags 1..k.
        suffix = readout[3:].lstrip("_")
        if not suffix.isdigit() or int(suffix) < 1:
            raise ValueError(f"'sum_k' readout needs a positive integer k, got {readout!r}")
        k = min(int(suffix), max_lag)          # clamp: 'sum4' with max_lag=3 == 'sum'
        W = B[:k].sum(0)
    else:
        raise ValueError(
            f"readout must be 'lag1', 'sum', 'peak', or 'sum_k' (e.g. 'sum4'), got {readout!r}")
    np.fill_diagonal(W, 0.0)
    return W


def infer_inhibitory(W):
    """Boolean ``[N]`` mask of inferred-inhibitory neurons (Dale, label-free).

    A neuron is called inhibitory when its net signed outgoing weight is
    negative. On the benchmarks this recovers true inhibitory neurons at
    AUC ~0.86-0.91.
    """
    return W.sum(1) < 0.0


# --------------------------------------------------------------------------- #
# Label-free thresholding (spike-jitter null -> FDR control)
# --------------------------------------------------------------------------- #
def _jitter_matrix(spike_matrix, boundaries, jitter_bins, rng):
    """One spike-jitter surrogate: shift each spike by +/-jitter_bins within its
    recording segment (rate/burst envelope preserved, synaptic window destroyed).
    """
    seg = list(zip(boundaries[:-1], boundaries[1:]))
    n = spike_matrix.shape[0]
    out = np.zeros_like(spike_matrix)
    for a, b in seg:
        L = b - a
        for i in range(n):
            idx = np.flatnonzero(spike_matrix[i, a:b])
            if len(idx):
                j = np.clip(idx + rng.integers(-jitter_bins, jitter_bins + 1, len(idx)), 0, L - 1)
                out[i, a + j] = 1.0
    return out


def _fdr_threshold(obs, null_scores, target_fdr):
    """Loosest (smallest) threshold with E[#null >= t] / #obs>=t <= target_fdr.

    ``obs`` : ``[n_cand]`` observed (already sign-oriented) scores.
    ``null_scores`` : ``[n_surrogates, n_cand]`` surrogate scores, same orientation.
    Vectorized over the unique observed values.
    """
    cand = np.unique(obs)
    obs_sorted = np.sort(obs)
    null_sorted = np.sort(null_scores.ravel())
    n_surr = null_scores.shape[0]
    n_obs = len(obs_sorted) - np.searchsorted(obs_sorted, cand, side="left")
    n_null = (len(null_sorted) - np.searchsorted(null_sorted, cand, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target_fdr)[0]
    return float(cand[ok[0]]) if len(ok) else float("inf")


def jitter_fdr_threshold(spike_matrix, boundaries, W, candidates, target_fdr=0.1,
                         jitter_bins=5, n_surrogates=8, max_lag=6, l2=2.0,
                         readout="sum4", sign=+1, seed=1):
    """Threshold ``sign * W`` at a target FDR using a spike-jitter null.

    Refits the GLM (same ``readout``) on ``n_surrogates`` jittered copies of the
    spikes to build the null over the same candidate pairs, then returns the
    loosest threshold whose surrogate FDR <= ``target_fdr`` and the predicted
    boolean mask. ``sign=+1`` for excitatory (``W >= thr``), ``-1`` for
    inhibitory (``-W >= thr``). No ground truth used.
    """
    rng = np.random.default_rng(seed)
    obs = (sign * W)[candidates]
    null = np.stack([
        (sign * glm_connectivity(
            _jitter_matrix(spike_matrix, boundaries, jitter_bins, rng),
            boundaries, max_lag=max_lag, l2=l2, readout=readout))[candidates]
        for _ in range(n_surrogates)
    ])
    thr = _fdr_threshold(obs, null, target_fdr)
    pred = (sign * W >= thr) & candidates if np.isfinite(thr) else np.zeros_like(candidates)
    return thr, pred


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def candidate_mask(n_neurons, positions=None, radius=None):
    """Boolean off-diagonal candidate mask.

    Whole-map (all off-diagonal pairs) unless a numeric spatial ``radius`` and
    ``positions`` are given, in which case only pairs within ``radius`` are kept.
    The radius is a *geometry* prior (electrode spacing), never derived from
    ground-truth edges.
    """
    off = ~np.eye(n_neurons, dtype=bool)
    if radius is None or positions is None:
        return off, None
    d = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1))
    return (d < radius) & off, float(radius)


# --------------------------------------------------------------------------- #
# Inference (no ground truth needed)
# --------------------------------------------------------------------------- #
def infer_connectivity(session_dir, bin_ms=5.0, max_lag=6, l2=2.0, readout="sum4",
                       target_fdr=0.1, jitter_ms=25.0, n_surrogates=8, radius=None,
                       seed=1):
    """Infer a directed, signed adjacency from spikes alone (label-free).

    Returns a dict with the signed weight ``W`` (per-edge peak), the inferred neuron
    types, the excitatory and type-constrained inhibitory predicted edge masks
    and their thresholds, the combined predicted adjacency, and metadata.
    """
    s = load_spikes(session_dir)
    n = s["n_neurons"]
    M, bnd = build_spike_matrix(s["recordings"], n, bin_ms)
    W = glm_connectivity(M, bnd, max_lag=max_lag, l2=l2, readout=readout)
    inferred_inh = infer_inhibitory(W)

    cand, radius = candidate_mask(n, s["positions"], radius)
    pre_inh = np.zeros((n, n), bool)
    pre_inh[np.where(inferred_inh)[0], :] = True

    jitter_bins = max(1, int(round(jitter_ms / bin_ms)))
    thr_exc, pred_exc = jitter_fdr_threshold(
        M, bnd, W, cand, target_fdr, jitter_bins, n_surrogates,
        max_lag, l2, readout, sign=+1, seed=seed)
    thr_inh, pred_inh = jitter_fdr_threshold(
        M, bnd, W, cand & pre_inh, target_fdr, jitter_bins, n_surrogates,
        max_lag, l2, readout, sign=-1, seed=seed)

    pred_adjacency = pred_exc | pred_inh          # directed, unsigned presence
    return {
        "W": W,                                    # signed [pre, post], peak-lag
        "candidates": cand,
        "inferred_inhibitory": inferred_inh,       # [N] bool
        "edges_exc": pred_exc,                     # high-confidence excitatory
        "edges_inh": pred_inh,                     # type-constrained inhibitory
        "pred_adjacency": pred_adjacency,
        "thr_exc": thr_exc, "thr_inh": thr_inh,
        "n_pred_exc": int(pred_exc.sum()), "n_pred_inh": int(pred_inh.sum()),
        "bin_ms": bin_ms, "max_lag": max_lag, "l2": l2, "readout": readout,
        "target_fdr": target_fdr, "jitter_ms": jitter_ms,
        "n_surrogates": n_surrogates, "candidate_radius": radius,
        "total_spikes": int(M.sum()),
    }


# --------------------------------------------------------------------------- #
# Evaluation (optional -- only when a network file is present)
# --------------------------------------------------------------------------- #
def evaluate(result, gt):
    """AUC/AP + recovered-topology confusion against ground truth (eval only)."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    W, cand = result["W"], result["candidates"]
    A_exc, A_inh = gt["A_exc"], gt["A_inh"]
    off = ~np.eye(W.shape[0], dtype=bool)
    ye, yi = A_exc[cand], A_inh[cand]
    se, si = W[cand], -W[cand]

    out = {"n_candidates": int(cand.sum())}
    if ye.any() and (~ye).any():
        out["auc_excitatory"] = float(roc_auc_score(ye, se))
        out["ap_excitatory"] = float(average_precision_score(ye, se))
    if yi.any() and (~yi).any():
        out["auc_inhibitory"] = float(roc_auc_score(yi, si))
        out["ap_inhibitory"] = float(average_precision_score(yi, si))
    # neuron-type inference quality
    if "is_inhibitory" in gt and len(set(gt["is_inhibitory"].tolist())) > 1:
        out["auc_neuron_type"] = float(roc_auc_score(gt["is_inhibitory"], -W.sum(1)))

    def confusion(pred, truth):
        TP = int((pred & truth).sum()); FP = int((pred & ~truth & cand).sum())
        FN = int((truth & off & ~pred).sum())
        P = TP / (TP + FP) if TP + FP else 0.0
        R = TP / (TP + FN) if TP + FN else 0.0
        return {"TP": TP, "FP": FP, "FN": FN, "precision": P, "recall": R,
                "f1": 2 * P * R / (P + R) if P + R else 0.0}

    allE, allI, allEdge = A_exc & off, A_inh & off, (A_exc | A_inh) & off
    out["confusion_excitatory"] = confusion(result["edges_exc"], allE)
    out["confusion_inhibitory"] = confusion(result["edges_inh"], allI)
    out["confusion_all_edges"] = confusion(result["pred_adjacency"], allEdge)
    out["n_true_edges"] = int(allEdge.sum())
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(session_dir, bin_ms=5.0, max_lag=6, l2=2.0, readout="sum4", target_fdr=0.1,
        jitter_ms=25.0, n_surrogates=8, radius=None, seed=1, save=True):
    """Infer -> (optionally evaluate) -> save. Runs with or without ground truth."""
    result = infer_connectivity(session_dir, bin_ms, max_lag, l2, readout,
                                target_fdr, jitter_ms, n_surrogates, radius, seed)
    gt = load_ground_truth(session_dir)
    metrics = evaluate(result, gt) if gt is not None else None

    if save:
        out = os.path.join(session_dir, f"glm_connectivity_{readout}_{int(bin_ms)}ms.npz")
        payload = {k: result[k] for k in
                   ("W", "candidates", "inferred_inhibitory", "edges_exc",
                    "edges_inh", "pred_adjacency")}
        if gt is not None:
            payload["A_exc"] = gt["A_exc"]
            payload["A_inh"] = gt["A_inh"]
        np.savez_compressed(out, **payload)
        result["saved"] = out
    return result, metrics


def calibrate_fdr(session_dir, targets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
                  bin_ms=5.0, max_lag=6, l2=2.0, readout="sum4",
                  jitter_ms=25.0, n_surrogates=8, seed=1, verbose=True):
    """Report ESTIMATED (jitter-null) vs REALIZED (needs ground truth) FDR across
    ``target_fdr`` values, for the excitatory (+W) discovery problem.

    The spike-jitter null is conservative -- surrogates keep the burst envelope
    and cluster co-activation, and with no true edges to absorb that variance the
    common drive spreads across all pairs and inflates the null. Integrating
    readouts ('sum'/'sum_k') span 0-20 ms and are far more exposed to this than
    the sharp-timing 'peak', so the nominal ``target_fdr`` overstates the true
    FDR *more* for 'sum_k'. When ground truth is present (simulation) this prints
    the gap so the operating target can be chosen against the realized FDR; on
    real data (no gt) only the estimated column is available.

    Fits W and the surrogate nulls ONCE, then sweeps ``targets`` over the stored
    scores (cheap). Returns a list of per-target dicts.
    """
    s = load_spikes(session_dir)
    n = s["n_neurons"]
    M, bnd = build_spike_matrix(s["recordings"], n, bin_ms)
    W = glm_connectivity(M, bnd, max_lag=max_lag, l2=l2, readout=readout)
    cand, _ = candidate_mask(n, s["positions"], None)
    jitter_bins = max(1, int(round(jitter_ms / bin_ms)))
    rng = np.random.default_rng(seed)
    null = np.array([
        glm_connectivity(_jitter_matrix(M, bnd, jitter_bins, rng),
                         bnd, max_lag=max_lag, l2=l2, readout=readout)[cand]
        for _ in range(n_surrogates)])           # [n_surrogates, n_candidates]
    obs = W[cand]

    gt = load_ground_truth(session_dir)
    y = None
    if gt is not None and "A_exc" in gt:
        any_edge = gt["A_exc"] | gt.get("A_inh", np.zeros_like(gt["A_exc"]))
        y = any_edge[cand]                        # any-edge, matches confusion_all_edges

    rows = []
    for t in targets:
        thr = _fdr_threshold(obs, null, t)
        sel = obs >= thr
        n_pred = int(sel.sum())
        est_fp = float((null >= thr).sum()) / null.shape[0]
        row = {"target_fdr": t, "threshold": float(thr), "n_pred": n_pred,
               "est_fdr": est_fp / max(n_pred, 1)}
        if y is not None:
            TP = int((sel & y).sum()); FP = int((sel & ~y).sum())
            FN = int((~sel & y).sum())
            row.update(TP=TP, FP=FP, FN=FN,
                       realized_fdr=FP / max(TP + FP, 1),
                       precision=TP / max(TP + FP, 1),
                       recall=TP / max(TP + FN, 1),
                       f1=(2 * TP) / max(2 * TP + FP + FN, 1))
        rows.append(row)

    if verbose:
        has_gt = y is not None
        print(f"FDR calibration | readout={readout} bin={bin_ms}ms max_lag={max_lag} "
              f"l2={l2} jitter={jitter_ms}ms n_surrogates={n_surrogates}")
        if has_gt:
            print(f"  {'target':>7} {'thr':>8} {'n_pred':>7} {'est_FDR':>8} "
                  f"{'realFDR':>8} {'TP':>6} {'FP':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
            for r in rows:
                print(f"  {r['target_fdr']:>7.2f} {r['threshold']:>8.4f} {r['n_pred']:>7} "
                      f"{r['est_fdr']:>8.4f} {r['realized_fdr']:>8.4f} {r['TP']:>6} "
                      f"{r['FP']:>6} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")
            print("  NOTE: est_FDR is what the jitter null claims; realFDR is the truth. "
                  "Pick target by the realFDR column.")
        else:
            print(f"  {'target':>7} {'thr':>8} {'n_pred':>7} {'est_FDR':>8}   (no ground truth: realized FDR unavailable)")
            for r in rows:
                print(f"  {r['target_fdr']:>7.2f} {r['threshold']:>8.4f} {r['n_pred']:>7} {r['est_fdr']:>8.4f}")
    return rows


def build_parser():
    p = argparse.ArgumentParser(description="Fine-resolution GLM connectivity (sum_k readout, jitter FDR)")
    p.add_argument("session_dir")
    p.add_argument("--bin-ms", type=float, default=5.0)
    p.add_argument("--max-lag", type=int, default=6)
    p.add_argument("--l2", type=float, default=2.0)
    p.add_argument("--readout", type=_readout_arg, default="sum4",
                   help="lag1 | sum | peak | sum_k (e.g. sum4, the default)")
    p.add_argument("--target-fdr", type=float, default=0.1)
    p.add_argument("--jitter-ms", type=float, default=25.0)
    p.add_argument("--n-surrogates", type=int, default=8)
    p.add_argument("--radius", type=float, default=None,
                   help="geometry candidate radius; default whole-map")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calibrate", action="store_true",
                   help="sweep target_fdr and report estimated vs realized FDR, then exit")
    return p


def _readout_arg(s):
    """argparse validator: accept lag1, sum, peak, or sum_k / sumN."""
    s2 = str(s).strip().lower()
    if s2 in ("lag1", "sum", "peak"):
        return s2
    if s2.startswith("sum") and s2[3:].lstrip("_").isdigit() and int(s2[3:].lstrip("_")) >= 1:
        return s2
    raise argparse.ArgumentTypeError(
        f"invalid readout {s!r}; use lag1, sum, peak, or sum_k (e.g. sum4)")


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.calibrate:
        calibrate_fdr(a.session_dir, bin_ms=a.bin_ms, max_lag=a.max_lag, l2=a.l2,
                      readout=a.readout, jitter_ms=a.jitter_ms,
                      n_surrogates=a.n_surrogates, seed=a.seed)
        sys.exit(0)
    result, metrics = run(a.session_dir, bin_ms=a.bin_ms, max_lag=a.max_lag, l2=a.l2,
                          readout=a.readout, target_fdr=a.target_fdr,
                          jitter_ms=a.jitter_ms, n_surrogates=a.n_surrogates,
                          radius=a.radius, seed=a.seed)
    scope = "whole-map" if result["candidate_radius"] is None else f"radius {result['candidate_radius']:.1f}"
    print(f"GLM ({result['readout']}) @ {result['bin_ms']}ms | {result['total_spikes']} spikes | {scope}")
    print(f"  predicted edges: {result['n_pred_exc']} excitatory (thr {result['thr_exc']:.4f}), "
          f"{result['n_pred_inh']} inhibitory (thr {result['thr_inh']:.4f}); "
          f"target FDR {result['target_fdr']}")
    if metrics is not None:
        print(f"  [eval] exc AUC {metrics.get('auc_excitatory'):.3f} AP {metrics.get('ap_excitatory'):.3f} | "
              f"inh AUC {metrics.get('auc_inhibitory'):.3f} AP {metrics.get('ap_inhibitory'):.3f} | "
              f"type AUC {metrics.get('auc_neuron_type', float('nan')):.3f}")
        a_ = metrics["confusion_all_edges"]
        print(f"  [eval] recovered wiring: TP {a_['TP']}/{metrics['n_true_edges']} "
              f"FP {a_['FP']} FN {a_['FN']} (P {a_['precision']:.2f} R {a_['recall']:.2f} F1 {a_['f1']:.2f})")
