"""Run every registered baseline on ONE dataset (session, state), both arms.

    python analysis/baselines/run_dataset.py --session sweep_c50_seed01 --state normal \
        [--n-recordings 200] [--methods all|tier1|tier2|a,b,c] \
        [--arms full,burst_excluded] [--force]

Per (method, arm) it writes, into ``<session>/results/<state>/baselines/``:

    <method>__<arm>__n<NNN>.npz   S (float32, [pre, post]), the predicted masks
                                  (oracle best-F1 per read-out, label-free
                                  untyped / typed when the method defines one),
                                  the E/I typing used
    <method>__<arm>__n<NNN>.json  long-format metric rows (readout,
                                  operating_point, denominator, metric, value),
                                  distance- and weight-resolved recovery,
                                  params, timings, arm bookkeeping

The accumulator is shared across methods within a (dataset, arm): every
method reads the same lag products and, where it uses one, the same jitter
surrogates.  Ground truth enters only through ``evaluate``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402
from baselines import evaluate as ev  # noqa: E402
from baselines import registry  # noqa: E402
from baselines.labelfree import rank_typing  # noqa: E402

ARMS = ("full", "burst_excluded")


def _packbits(mask):
    return np.packbits(np.asarray(mask, bool).ravel())


def evaluate_method(S, gt, lf, method_typing):
    """All metric rows + structure-resolved recovery for one score matrix."""
    S = np.asarray(S, np.float64)
    inh_eval = method_typing if method_typing is not None else rank_typing(-S.sum(1))
    res, masks, inh_eval = ev.threshold_free(S, gt, inh_eval)
    rows = ev.rows_threshold_free(res)
    structure = {}
    for ro, mask in masks.items():
        rows += ev.rows_confusion(mask, gt, ro, "oracle_best_f1")
    for ro in ("all_absS", "typed"):
        structure["oracle_best_f1/" + ro] = dict(
            distance=ev.distance_recovery(masks[ro], gt),
            weight=ev.weight_quantile_recovery(masks[ro], gt))
    lf_masks = {}
    if lf:
        for key, ro in (("untyped", "all_absS"), ("typed", "typed")):
            item = lf.get(key)
            if item is None:
                continue
            pred, info = item
            op = "label_free_" + key
            extra = {k: v for k, v in info.items()
                     if isinstance(v, (int, float)) and k not in ("n_pred",)}
            rows += ev.rows_confusion(pred, gt, ro, op, extra)
            structure[op + "/" + ro] = dict(distance=ev.distance_recovery(pred, gt),
                                            weight=ev.weight_quantile_recovery(pred, gt))
            lf_masks[key] = pred
    # neuron typing quality (rank rule) against the true labels
    typ = lf.get("typing_inh") if lf else None
    typ = inh_eval if typ is None else np.asarray(typ, bool)
    tp = int((typ & gt.is_inhibitory).sum())
    rows.append(dict(readout="typing", operating_point="rank_0.25", denominator="neurons",
                     metric="precision", value=tp / max(int(typ.sum()), 1)))
    rows.append(dict(readout="typing", operating_point="rank_0.25", denominator="neurons",
                     metric="recall", value=tp / max(int(gt.is_inhibitory.sum()), 1)))
    return rows, masks, lf_masks, structure, inh_eval


def run_one(session, state, n_recordings, methods, arms, force=False, out_dir=None,
            pad_ms=cm.PAD_MS, session_dir=None):
    """``session_dir`` / ``out_dir`` override the repo layout (CHTC worker)."""
    sd = session_dir or cm.dataset_dir(session, state)
    out_dir = out_dir or cm.baselines_results_dir(session, state)
    os.makedirs(out_dir, exist_ok=True)
    gt = ev.GroundTruth(sd)
    n_tag = "n%03d" % (n_recordings or len(cm.recording_paths(sd)))
    print("[%s/%s %s] N=%d | true exc %d inh %d | methods %s | arms %s"
          % (session, state, n_tag, gt.n, gt.n_exc, gt.n_inh, ",".join(methods), ",".join(arms)),
          flush=True)

    windows = None
    if "burst_excluded" in arms:
        t0 = time.time()
        windows = cm.burst_windows(sd, n_recordings=n_recordings,
                                   cache_path=os.path.join(out_dir, "burst_windows_p035.json"))
        print("  burst windows @%.2f gate: %d over %d recordings (%.0fs)"
              % (cm.BURST_GATE, sum(len(w) for w in windows), len(windows), time.time() - t0),
              flush=True)

    summary = []
    for arm in arms:
        sess = cm.Session(sd, n_recordings, None if arm == "full" else windows, pad_ms,
                          label="%s/%s" % (session, state))
        arm_info = dict(arm=arm)
        if arm != "full":
            for b in (cm.GLM_BIN_MS, cm.FINE_BIN_MS):
                d, T = sess.dropped(b)
                arm_info["dropped_bins_%gms" % b] = d
                arm_info["total_bins_%gms" % b] = T
                arm_info["dropped_frac_%gms" % b] = d / max(T, 1)
            arm_info["n_burst_windows"] = int(sum(len(w) for w in windows))
            arm_info["pad_ms"] = pad_ms
            arm_info["n_segments_5ms"] = sess.at(cm.GLM_BIN_MS).n_segments
            print("  [%s] dropped %.3f%% of 5 ms bins, %d segments"
                  % (arm, 100 * arm_info["dropped_frac_5ms"], arm_info["n_segments_5ms"]), flush=True)
        for name in methods:
            base = os.path.join(out_dir, "%s__%s__%s" % (name, arm, n_tag))
            if not force and os.path.exists(base + ".json"):
                print("  [%s] %s: exists, skipped" % (arm, name), flush=True)
                continue
            try:
                mod = registry.get(name)
                t0 = time.time()
                S = mod.score(sess, gt.n)
                S = np.asarray(S, np.float64)
                assert S.shape == (gt.n, gt.n), "bad shape"
                assert np.isfinite(S).all(), "non-finite scores"
                np.fill_diagonal(S, 0.0)
                t_score = time.time() - t0
                lf, t_lf = None, 0.0
                if hasattr(mod, "label_free"):
                    t1 = time.time()
                    lf = mod.label_free(sess, gt.n, S)
                    t_lf = time.time() - t1
                mt = None
                if hasattr(mod, "typing_inhibitory"):
                    mt = mod.typing_inhibitory(sess, gt.n)
                t2 = time.time()
                rows, masks, lf_masks, structure, inh_eval = evaluate_method(S, gt, lf, mt)
                t_eval = time.time() - t2
                meta = dict(session=session, state=state, n_recordings=n_recordings,
                            n_neurons=gt.n, method=name, module=mod.MODULE, tier=mod.TIER,
                            directed=mod.DIRECTED, citation=mod.CITATION,
                            params={k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in mod.PARAMS.items()},
                            arm=arm, arm_info=arm_info,
                            n_true_exc=gt.n_exc, n_true_inh=gt.n_inh, n_candidates=int(gt.off.sum()),
                            label_free_rule=(lf or {}).get("rule"),
                            elapsed_s=dict(score=t_score, label_free=t_lf, evaluate=t_eval))
                with open(base + ".json", "w", encoding="utf-8") as fh:
                    json.dump(dict(meta=meta, rows=rows, structure=structure), fh, indent=1)
                payload = dict(S=S.astype(np.float32), shape=np.array(S.shape),
                               typing_inh=inh_eval)
                for ro, m in masks.items():
                    payload["oracle_" + ro] = _packbits(m)
                for k, m in lf_masks.items():
                    payload["labelfree_" + k] = _packbits(m)
                np.savez_compressed(base + ".npz", **payload)
                hl = {(r["readout"], r["operating_point"], r["metric"]): r["value"] for r in rows}
                line = ("  [%s] %-16s exc AUC %.3f AP %.3f | untyped AP %.3f oracleF1 %.3f"
                        % (arm, name, hl[("exc", "threshold_free", "auc")],
                           hl[("exc", "threshold_free", "ap")],
                           hl[("all_absS", "threshold_free", "ap")],
                           hl[("all_absS", "oracle_best_f1", "f1")]))
                if ("all_absS", "label_free_untyped", "precision") in hl:
                    line += " | label-free P %.3f R %.3f" % (
                        hl[("all_absS", "label_free_untyped", "precision")],
                        hl[("all_absS", "label_free_untyped", "recall")])
                line += " | %.0fs" % (t_score + t_lf + t_eval)
                print(line, flush=True)
                summary.append(dict(method=name, arm=arm, ok=True))
            except Exception as e:  # keep the batch going; report loudly
                print("  [%s] %s FAILED: %s: %s" % (arm, name, type(e).__name__, e), flush=True)
                traceback.print_exc()
                with open(base + ".FAILED.txt", "w", encoding="utf-8") as fh:
                    fh.write(traceback.format_exc())
                summary.append(dict(method=name, arm=arm, ok=False, error=str(e)))
        sess.release()
        del sess
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--state", default="normal")
    ap.add_argument("--n-recordings", type=int, default=None)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--arms", default="full,burst_excluded")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--session-dir", default=None,
                    help="explicit directory holding recording*.npz + network_*.npz "
                         "(default: the repo layout)")
    a = ap.parse_args()
    methods = registry.resolve(a.methods)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    for x in arms:
        if x not in ARMS:
            raise SystemExit("unknown arm %r (use %s)" % (x, "/".join(ARMS)))
    t0 = time.time()
    run_one(a.session, a.state, a.n_recordings, methods, arms, a.force, a.out_dir,
            session_dir=a.session_dir)
    print("TOTAL %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
