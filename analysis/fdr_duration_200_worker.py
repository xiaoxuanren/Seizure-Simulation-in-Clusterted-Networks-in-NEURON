"""FDR-target x duration sweep, extended to the full 200-recording session.

For each duration the sum4 fit and its jitter nulls are computed ONCE, then every
nominal target FDR (0.1..1.0) is read off the stored score distributions -- the
threshold rule is post-hoc on the same scores, so targets are nearly free.

Settings are the shipped operating point (bin 5 ms, max_lag 6, l2 2.0, sum4,
25 ms jitter, 8 surrogates, seed 1), identical to the exclusion arms.

Durations are sharded across workers; ``fdr_duration_200_combine.py`` merges the
partial files and draws the figures. Outputs are tagged ``fdrdur10to200_`` and do
not collide with the existing 5-100 min sweep.

    python fdr_duration_200_worker.py --sizes 200 60
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg  # noqa: E402
from burstexcl_glm_arm import (BIN_MS, JBINS, N_SURR, SEED,  # noqa: E402
                               fdr_threshold, sum4_W, SESSION)
from session_paths import results_dir  # noqa: E402
_S = os.environ.get('DATASET_SESSION', 'IC-locked_flagship_200rec')
_T = os.environ.get('DATASET_STATE', 'normal')
RESULTS = results_dir(_S, _T, 'glm')
FIGS = results_dir(_S, _T, 'glm')

TARGETS = np.round(np.arange(0.1, 1.001, 0.1), 2)
OUTDIR = os.path.join(RESULTS, "fdrdur10to200_parts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    t0 = time.time()
    M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
    gt = sg.load_ground_truth(SESSION)
    N = M.shape[0]
    cand = ~np.eye(N, dtype=bool)
    ye = gt["A_exc"][cand].astype(bool)
    n_true = int(ye.sum())
    n_rec_avail = len(bnd) - 1
    print("loaded [%d x %d] %d rec, true_exc=%d | %.1fs"
          % (N, M.shape[1], n_rec_avail, n_true, time.time() - t0), flush=True)

    for n_rec in a.sizes:
        if n_rec > n_rec_avail:
            print("  skip n=%d (only %d available)" % (n_rec, n_rec_avail), flush=True)
            continue
        out = os.path.join(OUTDIR, "fdrdur_n%03d.json" % n_rec)
        if os.path.exists(out):
            print("  skip n=%d (done)" % n_rec, flush=True)
            continue
        tf = time.time()
        Mn = M[:, :bnd[n_rec]].tocsr()
        bndn = bnd[:n_rec + 1]

        W = sum4_W(Mn, bndn)
        se = W[cand]
        rng = np.random.default_rng(SEED)
        null = np.stack([sum4_W(sg.jitter(Mn, bndn, JBINS, rng), bndn)[cand]
                         for _ in range(N_SURR)])

        rows = []
        for tgt in TARGETS:
            thr, est = fdr_threshold(se, null, float(tgt))
            pred = se >= thr if np.isfinite(thr) else np.zeros_like(se, bool)
            TP = int((pred & ye).sum())
            FP = int((pred & ~ye).sum())
            FN = int((~pred & ye).sum())
            P = TP / max(TP + FP, 1)
            R = TP / max(TP + FN, 1)
            rows.append(dict(target=float(tgt), thr=thr, estimated_fdr=est,
                             n_pred=TP + FP, TP=TP, FP=FP, precision=P, recall=R,
                             f1=2 * P * R / max(P + R, 1e-9),
                             realized_fdr=FP / max(TP + FP, 1)))
        json.dump(dict(n_rec=n_rec, minutes=n_rec, n_true_exc=n_true,
                       occupied_bins=int(Mn.nnz), rows=rows,
                       bin_ms=BIN_MS, n_surrogates=N_SURR, seed=SEED),
                  open(out, "w"), indent=2)
        print("  n=%3d done %.1fs  (realFDR @target0.1=%.3f  @1.0=%.3f)"
              % (n_rec, time.time() - tf, rows[0]["realized_fdr"],
                 rows[-1]["realized_fdr"]), flush=True)
    print("worker TOTAL %.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
