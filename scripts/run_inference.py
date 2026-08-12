"""Sparse-GLM connectivity-inference runner for a NEURON session.

Usage:
    python scripts/run_inference.py --session <session_dir> env
    python scripts/run_inference.py --session <session_dir> glm \
        [--readout peak|lag1|sum|sum4] [--max-lag 6] [--bin-ms 5] [--edges]

- Uses ALL recordings in the session (no cap).
- GLM: one joint lag-resolved ridge fit, then reports per-lag AUC/AP to find the
  best lag (memory-frugal block-wise Gram assembly so all recordings fit).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# --------------------------------------------------------------------------- #
# env / feasibility
# --------------------------------------------------------------------------- #
def cmd_env(args):
    try:
        import psutil
        vm = psutil.virtual_memory()
        print("RAM total=%.1f GB  free=%.1f GB  CPU=%d" % (
            vm.total / 1e9, vm.available / 1e9, os.cpu_count()))
    except Exception as e:
        print("psutil unavailable (%s); CPU=%d" % (e, os.cpu_count()))
    for m in ("numpy", "sklearn"):
        try:
            mod = __import__(m)
            print("%-8s %s" % (m, getattr(mod, "__version__", "?")))
        except Exception as e:
            print("%-8s MISSING (%s)" % (m, e))

    recs = sorted(glob.glob(os.path.join(args.session, "recording[0-9][0-9][0-9].npz")))
    print("session: %s" % args.session)
    print("recordings found: %d" % len(recs))
    if recs:
        d = np.load(recs[0], allow_pickle=True)
        dur = float(d["duration"])
        n = len(d["spike_times"])
        print("per-recording: N=%d neurons, duration=%.0f ms, has_voltage=%s" % (
            n, dur, "voltage_traces" in d.files))
        total_bins_1ms = int(len(recs) * dur)
        print("all-recordings spike matrix @1ms  ~ [%d x %d]  (~%.1f GB float32)" % (
            n, total_bins_1ms, n * total_bins_1ms * 4 / 1e9))


# --------------------------------------------------------------------------- #
# GLM lag sweep (memory-frugal, all recordings)
# --------------------------------------------------------------------------- #
def _shifted(M, boundaries, lag):
    """M shifted right by `lag` bins, with leaked bins zeroed at segment starts."""
    s = np.zeros_like(M)
    s[:, lag:] = M[:, :-lag]
    for b in boundaries[1:-1]:
        s[:, b:b + lag] = 0.0
    return s


def fit_B_blockwise(M, boundaries, max_lag, l2):
    """Joint lag-resolved ridge coefficients B[lag, pre, post] without ever
    materialising the full [max_lag*N, T] design matrix.

    Solves (F F^T + l2 I) B = F M^T where F stacks lag-shifted copies of M,
    assembling the Gram matrix block-by-block (peak memory ~ two shifted copies).
    """
    N, T = M.shape
    ML = max_lag
    G = np.zeros((ML * N, ML * N), np.float64)
    RHS = np.zeros((ML * N, N), np.float64)
    for a in range(ML):
        Sa = _shifted(M, boundaries, a + 1)
        RHS[a * N:(a + 1) * N] = Sa @ M.T
        for b in range(a, ML):
            Sb = Sa if b == a else _shifted(M, boundaries, b + 1)
            blk = (Sa @ Sb.T).astype(np.float64)
            G[a * N:(a + 1) * N, b * N:(b + 1) * N] = blk
            if b != a:
                G[b * N:(b + 1) * N, a * N:(a + 1) * N] = blk.T
            del Sb
        del Sa
    G[np.diag_indices_from(G)] += l2
    B = np.linalg.solve(G, RHS)
    return B.reshape(ML, N, N)


def cmd_glm(args):
    import glm_connectivity as glm
    from sklearn.metrics import roc_auc_score, average_precision_score

    t0 = time.time()
    s = glm.load_spikes(args.session)
    n = s["n_neurons"]
    M, bnd = glm.build_spike_matrix(s["recordings"], n, args.bin_ms)
    nrec = len(s["recordings"])
    print("[GLM] %d recordings | spike matrix [%d x %d] (%.0fs of data) | load %.1fs" % (
        nrec, M.shape[0], M.shape[1], M.shape[1] * args.bin_ms / 1000, time.time() - t0))

    gt = glm.load_ground_truth(args.session)
    if gt is None:
        print("[GLM] no network file -> cannot evaluate AUC; aborting sweep.")
        return
    cand, _ = glm.candidate_mask(n, s["positions"], None)
    ye, yi = gt["A_exc"][cand], gt["A_inh"][cand]

    if args.calibrate:
        print("\n[GLM] FDR calibration (estimated vs realized FDR across targets)...")
        glm.calibrate_fdr(args.session, bin_ms=args.bin_ms, max_lag=args.max_lag,
                          l2=args.l2, readout=args.readout)
        return

    tf = time.time()
    B = fit_B_blockwise(M, bnd, args.max_lag, args.l2)
    print("[GLM] joint lag fit (max_lag=%d, l2=%.1f) in %.1fs" % (
        args.max_lag, args.l2, time.time() - tf))

    print("\n[GLM] per-lag ranking | candidates=%d  true exc=%d  inh=%d" % (
        int(cand.sum()), int(ye.sum()), int(yi.sum())))
    print("  lag   window     exc_AUC  exc_AP   inh_AUC  inh_AP")
    rows = []
    for k in range(args.max_lag):
        W = B[k].copy()
        np.fill_diagonal(W, 0.0)
        se, si = W[cand], -W[cand]
        ea = roc_auc_score(ye, se); ep = average_precision_score(ye, se)
        ia = roc_auc_score(yi, si); ip = average_precision_score(yi, si)
        rows.append((k + 1, ea, ep, ia, ip))
        print("  %-3d   %2d-%2dms    %.3f    %.3f    %.3f    %.3f" % (
            k + 1, k * args.bin_ms, (k + 1) * args.bin_ms, ea, ep, ia, ip))

    best = max(rows, key=lambda r: r[1])
    print("\n[GLM] BEST lag by excitatory AUC = lag %d (%d-%dms): "
          "exc_AUC=%.3f exc_AP=%.3f | inh_AUC=%.3f inh_AP=%.3f" % (
              best[0], (best[0] - 1) * args.bin_ms, best[0] * args.bin_ms,
              best[1], best[2], best[3], best[4]))

    import json
    out = os.path.join(args.session, "glm_lag_sweep.json")
    with open(out, "w") as fh:
        json.dump({"n_recordings": nrec, "bin_ms": args.bin_ms, "max_lag": args.max_lag,
                   "per_lag": [{"lag": r[0], "exc_auc": r[1], "exc_ap": r[2],
                                "inh_auc": r[3], "inh_ap": r[4]} for r in rows],
                   "best_lag": best[0]}, fh, indent=2)
    print("[GLM] sweep saved -> %s" % out)

    if args.edges:
        print("\n[GLM] label-free predicted edges via glm.run (readout=%s, all recordings)..."
              % args.readout)
        res, m = glm.run(args.session, bin_ms=args.bin_ms, max_lag=args.max_lag,
                         l2=args.l2, readout=args.readout,
                         target_fdr=args.target_fdr, save=True)
        ca = m["confusion_all_edges"]
        print("[GLM] %s @FDR%.2f: %d exc + %d inh edges -> TP=%d FP=%d FN=%d "
              "(P=%.2f R=%.2f F1=%.2f); neuron-type AUC=%.3f" % (
                  args.readout,
                  res["target_fdr"], res["n_pred_exc"], res["n_pred_inh"],
                  ca["TP"], ca["FP"], ca["FN"], ca["precision"], ca["recall"], ca["f1"],
                  m.get("auc_neuron_type", float("nan"))))


def _readout_arg(x):
    """Accept lag1, sum, peak, or sum_k / sumN (e.g. sum4)."""
    x2 = str(x).strip().lower()
    if x2 in ("lag1", "sum", "peak"):
        return x2
    if x2.startswith("sum") and x2[3:].lstrip("_").isdigit() and int(x2[3:].lstrip("_")) >= 1:
        return x2
    raise argparse.ArgumentTypeError(
        "invalid readout %r; use lag1, sum, peak, or sum_k (e.g. sum4)" % (x,))


def main():
    p = argparse.ArgumentParser(description="Sparse-GLM connectivity inference on a NEURON session")
    p.add_argument("--session", required=True,
                   help="path to a session directory holding recording*.npz + network_*.npz")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("env"); pe.set_defaults(func=cmd_env)

    pg = sub.add_parser("glm"); pg.set_defaults(func=cmd_glm)
    pg.add_argument("--max-lag", type=int, default=6)
    pg.add_argument("--bin-ms", type=float, default=5.0)
    pg.add_argument("--l2", type=float, default=2.0)
    pg.add_argument("--readout", type=_readout_arg, default="sum4",
                    help="lag score reduction: lag1|sum|peak|sum_k; 'sum4' "
                         "(default) sums lags 1-4 -- best exc ranking; see --calibrate")
    pg.add_argument("--target-fdr", type=float, default=0.1,
                    help="jitter-null FDR target for --edges (nominal; see --calibrate)")
    pg.add_argument("--calibrate", action="store_true",
                    help="sweep target_fdr, report estimated vs realized FDR, then exit")
    pg.add_argument("--edges", action="store_true",
                    help="also run the label-free jitter-FDR edge prediction (heavier)")

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
