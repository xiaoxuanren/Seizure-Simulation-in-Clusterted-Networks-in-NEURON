"""Aggregate every per-dataset baseline JSON into the cross-network tables.

    python analysis/baselines/aggregate.py [--n-recordings 200]

Writes into notebooks/NEURON data parallel/sweep_summary/baselines/:

    metrics.csv      long format, one row per measurement:
                     session, group, state, n_recordings, method, module, tier,
                     directed, arm, readout, operating_point, denominator,
                     metric, value
    metrics_wide.csv one row per (session, state, method, arm) with the headline
                     columns (exc_auc, exc_ap, inh_ap, untyped_ap, typed_ap,
                     oracle_f1_untyped, labelfree_precision/recall/f1_untyped,
                     labelfree_precision/recall/f1_typed, realized_fdr ...)
    structure.json   distance- and weight-resolved recovery per
                     (session, state, method, arm, operating point)
    compute.json     wall time per (session, state, method, arm)
    params.json      the parameter set every method ran with

Only reads the JSON outputs of run_dataset.py; never touches spikes or labels.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
for _p in (ANALYSIS,):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402

HEADLINE = [
    ("exc_auc", "exc", "threshold_free", "auc"),
    ("exc_ap", "exc", "threshold_free", "ap"),
    ("inh_auc", "inh", "threshold_free", "auc"),
    ("inh_ap", "inh", "threshold_free", "ap"),
    ("untyped_auc", "all_absS", "threshold_free", "auc"),
    ("untyped_ap", "all_absS", "threshold_free", "ap"),
    ("typed_auc", "typed", "threshold_free", "auc"),
    ("typed_ap", "typed", "threshold_free", "ap"),
    ("oracle_f1_untyped", "all_absS", "oracle_best_f1", "f1"),
    ("oracle_precision_untyped", "all_absS", "oracle_best_f1", "precision"),
    ("oracle_recall_untyped", "all_absS", "oracle_best_f1", "recall"),
    ("oracle_f1_typed", "typed", "oracle_best_f1", "f1"),
    ("oracle_recall_exc_only_untyped", "all_absS", "oracle_best_f1", "recall@exc_only"),
    ("recall_true_fdr10_exc", "exc", "oracle_true_fdr10", "recall"),
    ("labelfree_precision_untyped", "all_absS", "label_free_untyped", "precision"),
    ("labelfree_recall_untyped", "all_absS", "label_free_untyped", "recall"),
    ("labelfree_f1_untyped", "all_absS", "label_free_untyped", "f1"),
    ("labelfree_realized_fdr_untyped", "all_absS", "label_free_untyped", "realized_fdr"),
    ("labelfree_n_pred_untyped", "all_absS", "label_free_untyped", "n_pred"),
    ("labelfree_precision_typed", "typed", "label_free_typed", "precision"),
    ("labelfree_recall_typed", "typed", "label_free_typed", "recall"),
    ("labelfree_f1_typed", "typed", "label_free_typed", "f1"),
    ("labelfree_realized_fdr_typed", "typed", "label_free_typed", "realized_fdr"),
    ("labelfree_recall_exc_only_untyped", "all_absS", "label_free_untyped", "recall@exc_only"),
    ("labelfree_f1_exc_only_untyped", "all_absS", "label_free_untyped", "f1@exc_only"),
    ("typing_precision", "typing", "rank_0.25", "precision"),
    ("typing_recall", "typing", "rank_0.25", "recall"),
]


def group_of(session):
    return "c50" if "_c50_" in session else "c40"


def load_all(n_tag):
    files = sorted(glob.glob(os.path.join(cm.DATA, "sweep_c[45]0_seed*", "results", "*",
                                          "baselines", "*__*__%s.json" % n_tag)))
    for p in files:
        with open(p, encoding="utf-8") as fh:
            yield p, json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-recordings", type=int, default=200)
    ap.add_argument("--suffix", default="",
                    help="appended to every output name (e.g. _pilot_n050 keeps a pilot "
                         "aggregate next to the main one)")
    a = ap.parse_args()
    n_tag = "n%03d" % a.n_recordings
    out = cm.summary_dir()
    sfx = a.suffix

    long_rows, wide, structure, compute, params = [], {}, {}, {}, {}
    n_files = 0
    for path, d in load_all(n_tag):
        n_files += 1
        m = d["meta"]
        key = (m["session"], m["state"], m["method"], m["arm"])
        base = dict(session=m["session"], group=group_of(m["session"]), state=m["state"],
                    n_recordings=m["n_recordings"], method=m["method"], module=m.get("module", m["method"]),
                    tier=m["tier"], directed=m["directed"], arm=m["arm"])
        lookup = {}
        for r in d["rows"]:
            row = dict(base)
            row.update(r)
            long_rows.append(row)
            lookup[(r["readout"], r["operating_point"], r["denominator"], r["metric"])] = r["value"]
        w = dict(base)
        for col, ro, op, metric in HEADLINE:
            if "@" in metric:
                metric, denom = metric.split("@")
            else:
                denom = {"exc": "exc_only", "inh": "inh_only", "typing": "neurons"}.get(ro, "all_edges")
            w[col] = lookup.get((ro, op, denom, metric), "")
        ai = m.get("arm_info", {})
        w["dropped_frac_5ms"] = ai.get("dropped_frac_5ms", 0.0)
        w["n_burst_windows"] = ai.get("n_burst_windows", 0)
        w["elapsed_s"] = sum(m["elapsed_s"].values())
        wide[key] = w
        structure["/".join(key)] = d["structure"]
        compute["/".join(key)] = m["elapsed_s"]
        params.setdefault(m["method"], m["params"])

    cols = ["session", "group", "state", "n_recordings", "method", "module", "tier", "directed",
            "arm", "readout", "operating_point", "denominator", "metric", "value"]
    with open(os.path.join(out, "metrics%s.csv" % sfx), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(long_rows)
    wcols = ["session", "group", "state", "n_recordings", "method", "module", "tier", "directed", "arm"] + \
            [h[0] for h in HEADLINE] + ["dropped_frac_5ms", "n_burst_windows", "elapsed_s"]
    with open(os.path.join(out, "metrics_wide%s.csv" % sfx), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=wcols)
        wr.writeheader()
        for key in sorted(wide):
            wr.writerow(wide[key])
    for name, obj in (("structure%s.json" % sfx, structure), ("compute%s.json" % sfx, compute),
                      ("params%s.json" % sfx, params)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
    print("%d result files -> %s (%d long rows, %d wide rows)"
          % (n_files, out, len(long_rows), len(wide)))
    methods = sorted({k[2] for k in wide})
    for st in ("normal", "seizure"):
        for arm in ("full", "burst_excluded"):
            print("\n%s / %s   (mean over sessions)" % (st, arm))
            print("  %-16s %5s %7s %7s %8s %8s %8s %8s" % ("method", "n", "excAUC", "excAP", "untypAP",
                                                          "orF1", "lfP", "lfR"))
            for meth in methods:
                rows = [w for (s, t, mm, ar), w in wide.items() if t == st and mm == meth and ar == arm]
                if not rows:
                    continue

                def mean(col):
                    v = [float(r[col]) for r in rows if r[col] != ""]
                    return sum(v) / len(v) if v else float("nan")
                print("  %-16s %5d %7.3f %7.3f %8.3f %8.3f %8.3f %8.3f"
                      % (meth, len(rows), mean("exc_auc"), mean("exc_ap"), mean("untyped_ap"),
                         mean("oracle_f1_untyped"), mean("labelfree_precision_untyped"),
                         mean("labelfree_recall_untyped")))


if __name__ == "__main__":
    main()
