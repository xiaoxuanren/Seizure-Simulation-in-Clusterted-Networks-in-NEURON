"""Full learned-LIF connectivity inference on a NEURON session, with explicit
TP/FP/FN at three operating points.

Runs the validated `run_learned_lif_pipeline` on ALL recordings with
`num_workers` (fast data loading) + per-epoch checkpoint/resume, then writes
`learned_lif_full_results.json` reporting the edge confusion matrix at:

  * oracle_f1              -- best-case F1 (uses labels; a CEILING, not deployable)
  * surrogate_fdr          -- label-free / honest deployable operating point
  * low_fp_precision_target -- loosest threshold with precision >= target (few FP)

All three are computed from the pipeline's own score matrix + ground truth, so
they are internally consistent. Designed to be launched detached / as a
scheduled task so it survives a Claude-session exit; resumes from --checkpoint-dir.

Run under py39 (see memory neuron-runtime-setup):
    python _run_lif_full.py --session <dir> --num-workers 8 --epochs 30 --K 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
for _p in (REPO, os.path.join(REPO, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SESSION = os.path.join(
    REPO, "notebooks", "NEURON data parallel",
    "IC-locked_flagship_200rec", "normal"
)


def _confusion(scores, labels, threshold):
    """Confusion matrix + P/R/F1 for `scores >= threshold`."""
    pred = scores >= threshold
    pos = labels == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=round(precision, 4), recall=round(recall, 4),
                f1=round(f1, 4), threshold=float(threshold))


def _flatten_candidate_scores(conn_matrix, true_binary, neighbor_indices):
    """Flatten (score, label) over candidate edges only, dropping non-finite."""
    scores, labels = [], []
    n = conn_matrix.shape[0]
    for post in range(n):
        cand = np.asarray(neighbor_indices[post]).ravel()
        if cand.size == 0:
            continue
        # Match evaluate_connectivity: rank by |connectivity| over ALL candidates.
        scores.append(np.abs(np.asarray(conn_matrix[post, cand], dtype=np.float64)))
        labels.append(np.asarray(true_binary[post, cand], dtype=np.int64))
    return np.concatenate(scores), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=DEFAULT_SESSION)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--max-delay", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None,
                    help="torch device: 'cpu', 'cuda', or omit for auto (cuda if available)")
    ap.add_argument("--output-tag", default=None,
                    help="tag appended to pipeline output filenames (keeps runs separate)")
    ap.add_argument("--select", default="val_loss", choices=["val_loss", "conn_auc"],
                    help="checkpoint selection metric: val_loss (default) or conn_auc (best connectivity)")
    ap.add_argument("--exclude-bursts", default="true", choices=["true", "false"])
    ap.add_argument("--target-precision", type=float, default=0.9)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from lif_inference import run_learned_lif_pipeline
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve)

    ckpt_dir = args.checkpoint_dir or os.path.join(args.session, "lif_checkpoints")
    out_path = args.out or os.path.join(args.session, "learned_lif_full_results.json")

    print(f"[LIF-full] session={args.session}", flush=True)
    print(f"[LIF-full] device={args.device or 'auto'} workers={args.num_workers} "
          f"epochs={args.epochs} K={args.K} max_delay={args.max_delay} "
          f"batch={args.batch_size} exclude_bursts={args.exclude_bursts}", flush=True)
    print(f"[LIF-full] checkpoints -> {ckpt_dir}", flush=True)

    t0 = time.time()
    # Run in surrogate_fdr mode so `results` carries the honest (label-free)
    # operating point directly; oracle + low-FP are derived from conn_matrix.
    results, conn_matrix = run_learned_lif_pipeline(
        args.session,
        K=args.K, n_epochs=args.epochs, max_delay=args.max_delay,
        batch_size=args.batch_size,
        use_all_recordings=True,
        exclude_detected_bursts=(args.exclude_bursts == "true"),
        candidate_mode="hybrid",
        connectivity_threshold_mode="surrogate_fdr",
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_dir=ckpt_dir,
        output_tag=args.output_tag,
        select_by=args.select,
    )
    elapsed = time.time() - t0

    true_binary = results["true_binary"]
    neighbor_indices = results["neighbor_indices"]
    scores, labels = _flatten_candidate_scores(conn_matrix, true_binary, neighbor_indices)

    two_class = len(np.unique(labels)) > 1
    auc = float(roc_auc_score(labels, scores)) if two_class else float("nan")
    ap_ = float(average_precision_score(labels, scores)) if two_class else float("nan")

    # Sanity: our reconstructed AUC should match the pipeline's own.
    pipe_auc = results.get("auc")
    if pipe_auc is not None and np.isfinite(auc) and abs(auc - float(pipe_auc)) > 0.01:
        print(f"[LIF-full][warn] reconstructed AUC {auc:.4f} != pipeline AUC "
              f"{float(pipe_auc):.4f}; candidate reconstruction may differ.", flush=True)

    # --- operating points ---
    prec, rec, thr = precision_recall_curve(labels, scores)
    # oracle_f1: threshold maximizing F1 over the PR curve
    f1_curve = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    oi = int(np.argmax(f1_curve[:-1])) if thr.size else 0
    oracle_pt = _confusion(scores, labels, thr[oi] if thr.size else 0.0)

    # surrogate_fdr: the pipeline's calibrated threshold (honest, label-free)
    surrogate_pt = _confusion(scores, labels, float(results.get("threshold", np.inf)))
    surrogate_pt["estimated_fdr"] = results.get("estimated_fdr")

    # low_fp: loosest threshold (max recall) with precision >= target
    ok = prec[:-1] >= args.target_precision
    if np.any(ok):
        idx = np.where(ok)[0]
        low_thr = float(thr[idx[np.argmax(rec[:-1][idx])]])
        low_fp_pt = _confusion(scores, labels, low_thr)
    else:
        low_fp_pt = None

    out = dict(
        session=args.session, seconds=round(elapsed, 1), hours=round(elapsed / 3600, 3),
        params=dict(K=args.K, epochs=args.epochs, max_delay=args.max_delay,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    exclude_bursts=args.exclude_bursts),
        n_candidate_edges=int(labels.size), n_true_edges=int(labels.sum()),
        auc=round(auc, 4), ap=round(ap_, 4),
        operating_points=dict(
            oracle_f1=oracle_pt,
            surrogate_fdr=surrogate_pt,
            low_fp_precision_target=dict(target_precision=args.target_precision,
                                         **(low_fp_pt or {})),
        ),
    )
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print(f"\n[LIF-full] DONE {elapsed/3600:.2f}h | AUC={auc:.4f} AP={ap_:.4f} | "
          f"{int(labels.sum())} true / {int(labels.size)} candidate edges", flush=True)
    print(f"[LIF-full] oracle_f1      TP={oracle_pt['tp']} FP={oracle_pt['fp']} "
          f"FN={oracle_pt['fn']} (P={oracle_pt['precision']} R={oracle_pt['recall']} "
          f"F1={oracle_pt['f1']})", flush=True)
    print(f"[LIF-full] surrogate_fdr  TP={surrogate_pt['tp']} FP={surrogate_pt['fp']} "
          f"FN={surrogate_pt['fn']} (P={surrogate_pt['precision']} R={surrogate_pt['recall']} "
          f"F1={surrogate_pt['f1']} estFDR={surrogate_pt['estimated_fdr']})", flush=True)
    if low_fp_pt:
        print(f"[LIF-full] low_fp(P>={args.target_precision}) TP={low_fp_pt['tp']} "
              f"FP={low_fp_pt['fp']} FN={low_fp_pt['fn']} (P={low_fp_pt['precision']} "
              f"R={low_fp_pt['recall']} F1={low_fp_pt['f1']})", flush=True)
    else:
        print(f"[LIF-full] low_fp: no threshold reaches precision "
              f">= {args.target_precision}", flush=True)
    print(f"[LIF-full] results -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
