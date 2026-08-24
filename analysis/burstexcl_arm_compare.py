"""Full vs burst-excluded GLM arms on a sweep session: does excluding the
session's own saved burst windows restore edge-prediction precision?

Protocol constants (sum4 |W|, jitter-null FDR 0.70) are imported from
burstexcl_glm_arm.py so both arms match the shipped label-free pipeline.
The EXCL arm drops every recording's saved ``burst_windows`` (padded
+-PAD_MS) with each kept interval its own segment, so lagged features never
bridge a cut (same mechanics as the flagship ic-artifact arms).

Sweep finding this script established (13 networks, 2026-08): precision falls
monotonically with burst rate on the FULL arm; the EXCL arm restores every
bursty network to a common ~0.88-0.94 band and is a no-op on quiet ones.

    python analysis/burstexcl_arm_compare.py sweep_c50_seed03 [more sessions...]
"""
import glob
import importlib
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

PAD_MS = 25.0


def run_session(session_name):
    os.environ["DATASET_SESSION"] = session_name   # must precede the import
    import burstexcl_glm_arm as bx
    importlib.reload(bx)                            # re-resolve SESSION per session
    import sparse_glm as sg

    sd = bx.SESSION
    M, bnd = sg.load_session(sd, bin_ms=bx.BIN_MS)
    n = M.shape[0]

    # burst windows per recording, same ordering as load_session's glob sort
    rec_paths = [p for p in sorted(glob.glob(os.path.join(sd, "recording*.npz")))
                 if "raster" not in os.path.basename(p)]
    windows = []
    for rp in rec_paths:
        d = np.load(rp, allow_pickle=True)
        w = np.asarray(d["burst_windows"], float)
        windows.append(w.reshape(-1, 2) if w.size else np.empty((0, 2)))

    def trim_burst_windows(M, bnd):
        """bx.trim generalized to per-recording window lists."""
        blocks, new_bnd = [], [0]
        dropped = 0
        for r in range(len(bnd) - 1):
            s, e = bnd[r], bnd[r + 1]
            rec_bins = e - s
            keep = np.ones(rec_bins, bool)
            for (a_ms, b_ms) in windows[r]:
                lo = max(0, int(round((a_ms - PAD_MS) / bx.BIN_MS)))
                hi = min(rec_bins, int(round((b_ms + PAD_MS) / bx.BIN_MS)))
                keep[lo:hi] = False
            dropped += int((~keep).sum())
            idx = np.flatnonzero(keep)
            if not len(idx):
                continue
            splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for run in splits:
                blocks.append(M[:, s + run[0]:s + run[-1] + 1])
                new_bnd.append(new_bnd[-1] + len(run))
        return sp.hstack(blocks, format="csr"), new_bnd, dropped

    gt = sg.load_ground_truth(sd)
    true_adj = (gt["A_exc"] | gt["A_inh"]) & ~np.eye(n, dtype=bool)

    def arm(Mn, bndn, tag):
        W = bx.sum4_W(Mn, bndn)
        obs = np.abs(W)[~np.eye(n, dtype=bool)]
        rng = np.random.default_rng(bx.SEED)
        null = []
        for _ in range(bx.N_SURR):
            Mj = sg.jitter(Mn, bndn, bx.JBINS, rng)
            Wj = bx.sum4_W(Mj, bndn)
            null.append(np.abs(Wj)[~np.eye(n, dtype=bool)])
        thr, est_fdr = bx.fdr_threshold(obs, np.array(null), bx.TARGET_FDR)
        pred = np.abs(W) > thr
        np.fill_diagonal(pred, False)
        tp = int((pred & true_adj).sum()); fp = int((pred & ~true_adj).sum())
        fn = int((~pred & true_adj).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-12)
        print("  %-5s thr=%.5f estFDR=%.2f | pred %6d | TP %6d FP %6d FN %6d | "
              "P %.3f R %.3f F1 %.3f" % (tag, thr, est_fdr, int(pred.sum()),
                                         tp, fp, fn, P, R, F1), flush=True)
        return dict(tag=tag, threshold=float(thr), est_fdr=float(est_fdr),
                    n_pred=int(pred.sum()), tp=tp, fp=fp, fn=fn,
                    precision=float(P), recall=float(R), f1=float(F1))

    total_bins = bnd[-1]
    n_bursts = sum(len(w) for w in windows)
    print("%s: N=%d, %d recordings, %d burst windows" % (
        session_name, n, len(rec_paths), n_bursts), flush=True)

    full = arm(M, bnd, "FULL")
    Mx, bndx, dropped = trim_burst_windows(M, bnd)
    print("  EXCL drops %d/%d bins (%.2f%%), %d segments" % (
        dropped, total_bins, 100 * dropped / total_bins, len(bndx) - 1), flush=True)
    excl = arm(Mx, bndx, "EXCL")
    print("  precision delta: %+.3f\n" % (excl["precision"] - full["precision"]),
          flush=True)

    # persist for sweep_summary.py
    import json
    from session_paths import results_dir
    out = os.path.join(
        results_dir(session_name, os.environ.get("DATASET_STATE", "normal"), "glm"),
        "burstexcl_arms.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(session=session_name, n_neurons=n,
                       n_burst_windows=n_bursts, pad_ms=PAD_MS,
                       dropped_bins=dropped, total_bins=total_bins,
                       full=full, excl=excl), fh, indent=1)
    print("  saved -> %s" % out, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for s in sys.argv[1:]:
        run_session(s)
