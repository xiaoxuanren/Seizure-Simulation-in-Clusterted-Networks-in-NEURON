"""Part C: zero-edge negative control.

Runs the shipped label-free GLM operating point on recordings from a network
with NO connections (exc_weight_scale = inh_weight_scale = 0). Ground truth is
an empty edge set, so every predicted edge is a false positive by construction
and the meaningful number is ``n_pred`` itself.

Also runs the identical pipeline on the SAME NUMBER of flagship recordings, so
the zero-edge count is compared against a real-network count at matched data
volume rather than against the 200-recording headline.

Writes ``zeroedge_glm_metrics.json`` into the zero-edge session directory.
"""

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sparse_glm as sg  # noqa: E402
from burstexcl_glm_arm import (BIN_MS, MAX_LAG, L2, KSUM, TARGET_FDR,  # noqa: E402
                               JITTER_MS, N_SURR, SEED, JBINS,
                               fdr_threshold, sum4_W)

ZERO = os.path.join(REPO, "notebooks", "NEURON data parallel",
                    "zeroedge_control_15rec")
FLAGSHIP = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal",
                        "20260721_163430")


def run(session, n_rec, label, gt=None):
    t0 = time.time()
    M, bnd = sg.load_session(session, bin_ms=BIN_MS)
    n_avail = len(bnd) - 1
    n_rec = min(n_rec, n_avail)
    T = bnd[n_rec]
    Mn = M[:, :T].tocsr()
    bndn = bnd[:n_rec + 1]
    N = Mn.shape[0]
    cand = ~np.eye(N, dtype=bool)
    print("[%s] %d/%d recordings, [%d x %d], %d occupied bins | load %.1fs"
          % (label, n_rec, n_avail, N, Mn.shape[1], Mn.nnz, time.time() - t0),
          flush=True)

    W = sum4_W(Mn, bndn)
    se = W[cand]
    rng = np.random.default_rng(SEED)
    null = np.stack([sum4_W(sg.jitter(Mn, bndn, JBINS, rng), bndn)[cand]
                     for _ in range(N_SURR)])
    thr, est = fdr_threshold(se, null, TARGET_FDR)
    pred = se >= thr if np.isfinite(thr) else np.zeros_like(se, bool)
    n_pred = int(pred.sum())

    out = dict(label=label, session=os.path.basename(session), n_rec=n_rec,
               n_neurons=int(N), n_candidates=int(cand.sum()),
               occupied_bins=int(Mn.nnz), thr=thr, estimated_fdr=est,
               n_pred=n_pred, elapsed_s=time.time() - t0,
               bin_ms=BIN_MS, max_lag=MAX_LAG, l2=L2, ksum=KSUM,
               target_fdr=TARGET_FDR, jitter_ms=JITTER_MS,
               n_surrogates=N_SURR, seed=SEED)

    if gt is None:                      # zero-edge: truth is the empty set
        out.update(n_true_edges=0, TP=0, FP=n_pred,
                   realized_fdr=(1.0 if n_pred > 0 else float("nan")))
        print("  thr=%s  n_pred=%d  (ground truth = 0 edges -> every prediction "
              "is a false positive)"
              % (("%.5f" % thr) if np.isfinite(thr) else "inf", n_pred), flush=True)
        print("  realized FDR = %s"
              % ("1.000" if n_pred > 0 else "undefined (no predictions)"),
              flush=True)
    else:
        ye = gt["A_exc"][cand].astype(bool)
        TP = int((pred & ye).sum())
        FP = int((pred & ~ye).sum())
        out.update(n_true_edges=int(ye.sum()), TP=TP, FP=FP,
                   precision=TP / max(TP + FP, 1),
                   recall=TP / max(int(ye.sum()), 1),
                   realized_fdr=FP / max(TP + FP, 1))
        print("  thr=%.5f  n_pred=%d  TP=%d  FP=%d  P=%.3f  realFDR=%.3f"
              % (thr, n_pred, TP, FP, out["precision"], out["realized_fdr"]),
              flush=True)
    return out


def main():
    n_zero = len([f for f in os.listdir(ZERO)
                  if f.startswith("recording") and f.endswith(".npz")])
    print("zero-edge recordings available: %d" % n_zero, flush=True)

    res_zero = run(ZERO, n_zero, "zero-edge (no connections)", gt=None)
    gt = sg.load_ground_truth(FLAGSHIP)
    res_real = run(FLAGSHIP, n_zero, "flagship (duration-matched)", gt=gt)

    print("\n--- Part C verdict ---", flush=True)
    print("  zero-edge network : n_pred = %d   (true edges = 0)"
          % res_zero["n_pred"], flush=True)
    print("  real network, same duration : n_pred = %d  (TP = %d)"
          % (res_real["n_pred"], res_real["TP"]), flush=True)
    ratio = res_zero["n_pred"] / max(res_real["n_pred"], 1)
    print("  zero-edge predictions as a fraction of real-network predictions: "
          "%.4f" % ratio, flush=True)

    out = dict(zero_edge=res_zero, flagship_matched=res_real,
               zero_over_real=ratio)
    p = os.path.join(ZERO, "zeroedge_glm_metrics.json")
    json.dump(out, open(p, "w"), indent=2)
    print("saved -> %s" % p, flush=True)


if __name__ == "__main__":
    main()
