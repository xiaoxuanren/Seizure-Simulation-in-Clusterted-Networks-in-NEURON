"""Label-free GLM hyperparameter sweep for the fine-resolution ridge GLM.

Two LABEL-FREE selection criteria are computed for each (bin_ms, max_lag, l2):

  * reproducibility -- mean pairwise Spearman rank-corr of |W| (peak readout)
    across independent recording-FOLD fits. Measures whether the edge ranking is
    a reliable, reproducible signal. Connectivity-aligned, label-free. PRIMARY.
  * heldout_r2 -- k-fold held-out spike-prediction R^2 (held-out version of the
    ridge objective). Also label-free but rewards coarse bins that are easy to
    predict rather than good connectivity, so it is a SECONDARY cross-check.

Ground-truth connectivity is used ONLY for the final report and a transparency
"oracle gap"; it NEVER enters selection.

    python glm_sweep.py --session "<spikeonly session>" --folds 3

Robustness: results are checkpointed to --out after every (bin,max_lag) block, so
a re-run resumes (skips completed configs). Each block is error-isolated (a config
that OOMs/fails is logged and skipped, not fatal). numpy/scipy/sklearn only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg  # noqa: E402


def subset(M, bnd, recs):
    cols, nb = [], [0]
    for r in recs:
        s, e = bnd[r], bnd[r + 1]
        cols.append(M[:, s:e])
        nb.append(nb[-1] + (e - s))
    return sp.hstack(cols, format="csr"), nb


def build_gram(M, bnd, max_lag):
    n = M.shape[0]
    S = [sg._shift(M, bnd, k + 1) for k in range(max_lag)]
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
    return G, RHS, n


def solve_B(G, RHS, n, max_lag, l2):
    Gl = G.copy()
    Gl[np.diag_indices_from(Gl)] += l2
    return np.linalg.solve(Gl, RHS).reshape(max_lag, n, n)


def val_artifacts(Mv, bv, max_lag):
    S = [sg._shift(Mv, bv, k + 1) for k in range(max_lag)]
    Y = Mv.toarray().astype(np.float64)
    mu = Y.mean(1, keepdims=True)
    ss_tot = ((Y - mu) ** 2).sum(1)
    w = np.asarray(Mv.sum(1)).ravel()
    return S, Y, ss_tot, w


def score_r2(B, S, Y, ss_tot, w, max_lag):
    pred = np.zeros(Y.shape, np.float64)
    for lag in range(max_lag):
        pred += np.asarray(S[lag].T @ B[lag]).T
    ss_res = ((Y - pred) ** 2).sum(1)
    valid = ss_tot > 0
    r2 = np.zeros(Y.shape[0])
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    m = valid & (w > 0)
    return float(np.average(r2[m], weights=w[m])) if m.any() else float("nan")


def reproducibility(W_list):
    from scipy.stats import spearmanr
    cs = []
    for i in range(len(W_list)):
        for j in range(i + 1, len(W_list)):
            c = spearmanr(W_list[i], W_list[j]).correlation
            if np.isfinite(c):
                cs.append(c)
    return float(np.mean(cs)) if cs else float("nan")


def _best(rows, k):
    """argmax over rows on key k, ignoring non-finite values (NaN-guarded)."""
    fin = [r for r in rows if np.isfinite(r.get(k, float("nan")))]
    return max(fin or rows, key=lambda r: r[k])


def conn_auc(B, gt, off):
    """Connectivity AUC/AP vs ground truth -- REPORTING / transparency ONLY."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = (gt["A_exc"] | gt["A_inh"])[off]
    out = {}
    for r in ("peak", "lag1", "sum"):
        s = np.abs(sg.readout(B, r)[off])
        out[r] = dict(auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--bins", type=float, nargs="+", default=[1, 2, 3, 4, 5, 7, 10])
    ap.add_argument("--lags", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--l2s", type=float, nargs="+", default=[0.5, 1, 2, 5, 10])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", default="glm_sweep_results.json")
    a = ap.parse_args()

    gt = sg.load_ground_truth(a.session)
    off = ~np.eye(len(gt["is_inhibitory"]), dtype=bool)

    rows, done = [], set()
    if os.path.exists(a.out):
        try:
            rows = json.load(open(a.out)).get("grid", [])
            done = {(r["bin_ms"], r["max_lag"], r["l2"]) for r in rows}
            print(f"[sweep] resume: {len(rows)} configs already on disk", flush=True)
        except Exception as e:
            print(f"[sweep] could not resume ({e}); fresh start", flush=True)

    def save(final=False):
        p = {"grid": sorted(rows, key=lambda r: -r["reproducibility"]), "n_configs": len(rows),
             "protocol": "select by fold reproducibility of |W| (primary) / held-out R^2 (secondary); labels report-only"}
        if final and rows:
            sel = _best(rows, "reproducibility")
            sr2 = _best(rows, "heldout_r2")
            orc = _best(rows, "conn_auc_peak")
            p.update(selected_reproducibility=sel, selected_heldout_r2=sr2,
                     oracle_labelbased=orc, leakage_gap_auc=orc["conn_auc_peak"] - sel["conn_auc_peak"])
        json.dump(p, open(a.out, "w"), indent=2)

    print(f"[sweep] {a.session}\n[sweep] grid bins={a.bins} lags={a.lags} l2s={a.l2s} folds={a.folds}"
          f"  ({len(a.bins)*len(a.lags)*len(a.l2s)} configs)", flush=True)

    for bin_ms in a.bins:
        if all((bin_ms, lag, l2) in done for lag in a.lags for l2 in a.l2s):
            print(f"[skip] bin={bin_ms} (all done)", flush=True)
            continue
        t0 = time.time()
        M, bnd = sg.load_session(a.session, bin_ms)
        n, n_rec = M.shape[0], len(bnd) - 1
        fold_of = np.array([i % a.folds for i in range(n_rec)])
        print(f"[sweep] bin={bin_ms}ms  M[{n}x{M.shape[1]}]  {n_rec}rec  load {time.time()-t0:.1f}s", flush=True)
        for max_lag in a.lags:
            if all((bin_ms, max_lag, l2) in done for l2 in a.l2s):
                continue
            tl = time.time()
            try:
                fold_r2 = {l2: [] for l2 in a.l2s}
                fold_W = {l2: [] for l2 in a.l2s}
                for f in range(a.folds):
                    val = [r for r in range(n_rec) if fold_of[r] == f]
                    trn = [r for r in range(n_rec) if fold_of[r] != f]
                    Mt, bt = subset(M, bnd, trn)
                    Mv, bv = subset(M, bnd, val)
                    G, RHS, _ = build_gram(Mt, bt, max_lag)
                    S, Y, ss_tot, w = val_artifacts(Mv, bv, max_lag)
                    for l2 in a.l2s:
                        B = solve_B(G, RHS, n, max_lag, l2)
                        fold_r2[l2].append(score_r2(B, S, Y, ss_tot, w, max_lag))
                        fold_W[l2].append(np.abs(sg.readout(B, "peak")[off]))
                    del G, RHS, S, Y, Mt, Mv
                Ga, Ra, _ = build_gram(M, bnd, max_lag)
                for l2 in a.l2s:
                    if (bin_ms, max_lag, l2) in done:
                        continue
                    cm = conn_auc(solve_B(Ga, Ra, n, max_lag, l2), gt, off)
                    r2m = float(np.mean(fold_r2[l2]))
                    rep = reproducibility(fold_W[l2])
                    rows.append(dict(bin_ms=bin_ms, max_lag=max_lag, l2=l2, reproducibility=rep,
                                     heldout_r2=r2m, conn_auc_peak=cm["peak"]["auc"], conn_ap_peak=cm["peak"]["ap"],
                                     conn_auc_lag1=cm["lag1"]["auc"], conn_auc_sum=cm["sum"]["auc"]))
                    done.add((bin_ms, max_lag, l2))
                    print(f"  bin={bin_ms:<4} lag={max_lag} l2={l2:<4}  repro={rep:.3f}  R2={r2m:+.4f}  "
                          f"connAUC[peak/lag1/sum]={cm['peak']['auc']:.3f}/{cm['lag1']['auc']:.3f}/{cm['sum']['auc']:.3f}",
                          flush=True)
                del Ga, Ra
                save()
                print(f"    (bin={bin_ms} lag={max_lag} {time.time()-tl:.0f}s | {len(rows)} configs saved)", flush=True)
            except Exception as e:
                print(f"  [FAIL] bin={bin_ms} lag={max_lag}: {type(e).__name__}: {e}", flush=True)
                continue
        del M

    save(final=True)
    if not rows:
        print("[sweep] no configs completed", flush=True)
        return
    sel = _best(rows, "reproducibility")
    sr2 = _best(rows, "heldout_r2")
    orc = _best(rows, "conn_auc_peak")

    def line(tag, r):
        return (f"  {tag}: bin={r['bin_ms']} max_lag={r['max_lag']} l2={r['l2']} | "
                f"repro={r['reproducibility']:.3f} R2={r['heldout_r2']:+.4f} "
                f"-> connAUC(peak)={r['conn_auc_peak']:.4f} AP={r['conn_ap_peak']:.4f}")

    print("\n" + "=" * 72, flush=True)
    print("LABEL-FREE SELECTION", flush=True)
    print(line("reproducibility-pick (PRIMARY)", sel), flush=True)
    print(line("heldout-R2-pick (secondary)   ", sr2), flush=True)
    print("ORACLE (max connectivity AUC -- NOT used for selection; honesty check):", flush=True)
    print(line("oracle                        ", orc), flush=True)
    print(f"LEAKAGE GAP AVOIDED (repro pick vs oracle): "
          f"{orc['conn_auc_peak']-sel['conn_auc_peak']:+.4f} AUC", flush=True)
    print("=" * 72 + f"\n[sweep] saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
