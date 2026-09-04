"""Figure suite for the baseline comparison.  Reads only the outputs of
run_dataset.py / aggregate.py (metrics_wide.csv, structure.json, the per-
dataset npz) and the ground truth for the maps.

    python analysis/baselines/figures.py --n-recordings 200 [--which all|cross|maps]
        [--sessions all] [--representative sweep_c50_seed09]

Outputs (sweep_summary/baselines/):
    maps/<session>_<state>_<method>.png         per dataset x method: predicted-vs-
                                                true cluster-sorted adjacency (TP
                                                green, FP red, FN orange) at the
                                                method's operating point, plus
                                                distance-resolved recall and
                                                weight-quintile recovery (full arm
                                                solid, burst-excluded dashed)
    figures/roc_pr_<state>.png                  ROC + PR overlays, all methods,
                                                full arm; representative network
                                                bold, the other 19 faint
    figures/ap_heatmap_<state>_<arm>.png        method x network AP heatmaps
                                                (untyped and excitatory), networks
                                                ordered by burstiness, with the
                                                Spearman rho of AP vs burstiness
    figures/burst_gap.png                       full -> burst-excluded change per
                                                method (AP and operating-point
                                                precision), both states
    figures/delay_gate.png                      delay-recovery r per method
    figures/te_binwidth.png                     transfer-entropy bin-width sweep
    figures/summary_bars_<state>.png            mean +/- range of the headline
                                                metrics per method, both arms
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
for _p in (REPO, ANALYSIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from baselines import common as cm  # noqa: E402
from baselines import evaluate as ev  # noqa: E402
from baselines import registry  # noqa: E402

GREEN, RED, ORANGE = "#1a9850", "#d73027", "#f0a000"
C50, C40 = "#2C6FB5", "#C0392B"
STATE_COL = {"normal": "#2C6FB5", "seizure": "#C0392B"}
METHOD_ORDER = ["glm_ridge", "ccg_jitter", "glmcc", "stg", "tspe", "fncch", "ccg_classical",
                "te_1ms", "te_2ms", "te_5ms", "te_10ms", "pearson_lag", "pearson0",
                "stg_roi15", "glmcc_d14"]
LABEL = {"glm_ridge": "ridge GLM (shipped)", "ccg_jitter": "jittered CCG", "glmcc": "GLMCC",
         "stg": "STG (deconv.)", "tspe": "TSPE", "fncch": "FNCCH", "ccg_classical": "classical CCG",
         "te_1ms": "TE 1 ms", "te_2ms": "TE 2 ms", "te_5ms": "TE 5 ms", "te_10ms": "TE 10 ms",
         "pearson_lag": "lagged Pearson", "pearson0": "zero-lag Pearson",
         "stg_roi15": "STG ROI 15 ms", "glmcc_d14": "GLMCC d<=14 ms"}
_cmap = plt.get_cmap("tab20")
COLOR = {m: _cmap(i / max(len(METHOD_ORDER) - 1, 1)) for i, m in enumerate(METHOD_ORDER)}


def lab(m):
    return LABEL.get(m, m)


def order(methods):
    return sorted(methods, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)


# ------------------------------------------------------------------------- #
# inputs
# ------------------------------------------------------------------------- #
def read_wide(out):
    p = os.path.join(out, "metrics_wide.csv")
    with open(p, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("session", "group", "state", "method", "module", "arm", "directed"):
                try:
                    r[k] = float(v) if v != "" else np.nan
                except ValueError:
                    pass
    return rows


def burstiness(session, state):
    p = os.path.join(cm.DATA, session, state, "session_metadata.json")
    if not os.path.exists(p):
        return np.nan
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    return float(np.mean([r.get("n_bursts", 0) for r in m["recordings"]]))


def _unpack(npz, key, n):
    if key not in npz.files:
        return None
    return np.unpackbits(npz[key])[:n * n].reshape(n, n).astype(bool)


def load_result(session, state, method, arm, n_tag):
    d = cm.baselines_results_dir(session, state, create=False)
    base = os.path.join(d, "%s__%s__%s" % (method, arm, n_tag))
    if not os.path.exists(base + ".npz"):
        return None
    z = np.load(base + ".npz", allow_pickle=True)
    with open(base + ".json", encoding="utf-8") as fh:
        j = json.load(fh)
    n = int(z["shape"][0])
    return dict(S=z["S"].astype(np.float64), n=n, json=j,
                oracle_untyped=_unpack(z, "oracle_all_absS", n),
                oracle_typed=_unpack(z, "oracle_typed", n),
                lf_untyped=_unpack(z, "labelfree_untyped", n),
                lf_typed=_unpack(z, "labelfree_typed", n))


def operating_mask(res):
    """The mask drawn in the maps: label-free untyped if the method defines
    one, else the oracle best-F1 untyped mask.  Returns (mask, label)."""
    if res["lf_untyped"] is not None:
        return res["lf_untyped"], "label-free (%s)" % (res["json"]["meta"].get("label_free_rule") or "own rule")
    return res["oracle_untyped"], "oracle best-F1 (|S|)"


# ------------------------------------------------------------------------- #
# per dataset x method maps
# ------------------------------------------------------------------------- #
def fig_map(session, state, method, n_tag, out_dir):
    full = load_result(session, state, method, "full", n_tag)
    if full is None:
        return None
    excl = load_result(session, state, method, "burst_excluded", n_tag)
    gt = ev.GroundTruth(cm.dataset_dir(session, state))
    pred, op_label = operating_mask(full)
    pred = pred & gt.off
    tp, fp, fn = pred & gt.A_all, pred & ~gt.A_all, gt.A_all & ~pred
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-12)
    order_ = np.argsort(gt.clusters, kind="stable")
    Mx = np.zeros((gt.n, gt.n))
    Mx[fn] = 1
    Mx[fp] = 2
    Mx[tp] = 3

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32, left=0.05, right=0.985,
                          top=0.86, bottom=0.14)
    ax = fig.add_subplot(gs[0])
    ax.imshow(Mx[np.ix_(order_, order_)], cmap=ListedColormap(["white", ORANGE, RED, GREEN]),
              vmin=0, vmax=3, interpolation="nearest", aspect="equal")
    ax.set_xlabel("post (cluster-sorted)")
    ax.set_ylabel("pre (cluster-sorted)")
    ax.set_title("predicted vs true adjacency\nTP %d  FP %d  FN %d   P %.2f  R %.2f  F1 %.2f"
                 % (TP, FP, FN, P, R, F1), fontsize=10)
    ax.legend(handles=[Line2D([0], [0], color=GREEN, lw=4, label="TP"),
                       Line2D([0], [0], color=RED, lw=4, label="FP"),
                       Line2D([0], [0], color=ORANGE, lw=4, label="FN")],
              loc="lower left", fontsize=8, framealpha=0.9)

    def struct(res, key):
        for k in (key + "/all_absS",):
            if k in res["json"]["structure"]:
                return res["json"]["structure"][k]
        return None

    op_key = "label_free_untyped" if full["lf_untyped"] is not None else "oracle_best_f1"
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    for res, ls, tag in ((full, "-", "full"), (excl, "--", "burst-excluded")):
        if res is None:
            continue
        s = struct(res, op_key)
        if s is None:
            continue
        d = s["distance"]
        edges = np.asarray(d["bin_edges"])
        xc = 0.5 * (edges[:-1] + edges[1:])
        ax2.plot(xc, d["recall"], ls, color=GREEN, lw=1.6, label="recall (%s)" % tag)
        ax2.plot(xc, d["precision"], ls, color=C50, lw=1.6, label="precision (%s)" % tag)
        w = s["weight"]
        ax3.plot(range(1, 6), w["exc"], "o" + ls, color=GREEN, lw=1.6, label="excitatory (%s)" % tag)
        ax3.plot(range(1, 6), w["inh"], "s" + ls, color=RED, lw=1.6, label="inhibitory (%s)" % tag)
    ax2.set_xlabel("inter-neuron distance (space units)")
    ax2.set_ylabel("rate")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("distance-resolved recovery", fontsize=10)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(alpha=0.3)
    ax3.set_xlabel("true |weight| quintile within class (weak -> strong)")
    ax3.set_ylabel("detection probability")
    ax3.set_ylim(0, 1.02)
    ax3.set_xticks(range(1, 6))
    ax3.set_title("weight-quintile recovery", fontsize=10)
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(alpha=0.3)
    fig.suptitle("%s / %s / %s  --  %s  --  operating point: %s"
                 % (session, state, n_tag, lab(method), op_label), fontsize=11, weight="bold")
    p = os.path.join(out_dir, "%s_%s_%s.png" % (session, state, method))
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------- #
# cross-method figures
# ------------------------------------------------------------------------- #
def roc_pr_curves(score, truth):
    order_ = np.argsort(score)[::-1]
    y = truth[order_].astype(np.float64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, Nn = y.sum(), len(y) - y.sum()
    tpr, fpr = tp / P, fp / Nn
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / P
    step = max(1, len(y) // 4000)
    return fpr[::step], tpr[::step], rec[::step], prec[::step]


def fig_roc_pr(state, n_tag, out, sessions, methods, representative, arm="full"):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    handles = []
    for m in order(methods):
        for s in sessions:
            res = load_result(s, state, m, arm, n_tag)
            if res is None:
                continue
            gt = ev.GroundTruth(cm.dataset_dir(s, state))
            sc, tr = np.abs(res["S"])[gt.off], gt.A_all[gt.off]
            fpr, tpr, rec, prec = roc_pr_curves(sc, tr)
            bold = s == representative
            kw = dict(color=COLOR.get(m, "0.5"), lw=1.8 if bold else 0.6, alpha=1.0 if bold else 0.18)
            axes[0].plot(fpr, tpr, **kw)
            axes[1].plot(rec, prec, **kw)
            if bold:
                handles.append(Line2D([0], [0], color=COLOR.get(m, "0.5"), lw=2, label=lab(m)))
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", lw=0.8)
    axes[0].set_xlabel("false-positive rate")
    axes[0].set_ylabel("true-positive rate")
    axes[0].set_title("ROC, untyped |S| vs all edges")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("precision-recall (edge density %.1f-%.1f %%)" % (1.5, 2.1))
    axes[1].set_ylim(0, 1.02)
    for a in axes:
        a.grid(alpha=0.3)
    axes[1].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("%s state, %s arm -- bold: %s, faint: the other networks" % (state, arm, representative),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, "roc_pr_%s.png" % state)
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def fig_ap_heatmap(state, arm, n_tag, out, wide):
    rows = [r for r in wide if r["state"] == state and r["arm"] == arm]
    if not rows:
        return None, {}
    sessions = sorted({r["session"] for r in rows})
    b = np.array([burstiness(s, state) for s in sessions])
    sess_order = [sessions[i] for i in np.argsort(b)]
    b_sorted = np.sort(b)
    methods = order({r["method"] for r in rows})
    rhos = {}
    fig, axes = plt.subplots(2, 1, figsize=(13, 0.55 * len(methods) * 2 + 3.2))
    for ax, col, title in ((axes[0], "untyped_ap", "average precision, untyped |S| vs all edges"),
                           (axes[1], "exc_ap", "average precision, +S vs excitatory edges")):
        Mx = np.full((len(methods), len(sess_order)), np.nan)
        for r in rows:
            Mx[methods.index(r["method"]), sess_order.index(r["session"])] = r[col]
        im = ax.imshow(Mx, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, "%.2f" % Mx[i, j], ha="center", va="center", fontsize=6.5,
                            color="white" if Mx[i, j] < 0.55 else "black")
        rho = [_spearman(b_sorted, Mx[i]) for i in range(len(methods))]
        rhos[col] = dict(zip(methods, rho))
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(["%s  (rho %+.2f)" % (lab(m), r) if np.isfinite(r) else lab(m)
                            for m, r in zip(methods, rho)], fontsize=8)
        ax.set_xticks(range(len(sess_order)))
        ax.set_xticklabels(["%s\n%.1f" % (s.replace("sweep_", ""), bb) for s, bb in zip(sess_order, b_sorted)],
                           fontsize=6.5, rotation=90)
        ax.set_title("%s  (columns ordered by bursts per recording; rho = Spearman AP vs burstiness)" % title,
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[1].set_xlabel("network (bursts per 60 s recording, %s state)" % state)
    fig.suptitle("%s state, %s arm, %s" % (state, arm, n_tag), fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out, "ap_heatmap_%s_%s.png" % (state, arm))
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p, rhos


def _op_precision(r):
    """Precision at the method's own operating point (label-free if defined,
    else oracle best-F1), untyped."""
    v = r.get("labelfree_precision_untyped", np.nan)
    if isinstance(v, float) and np.isfinite(v):
        return v, "label-free"
    return r.get("oracle_precision_untyped", np.nan), "oracle"


def fig_burst_gap(n_tag, out, wide):
    methods = order({r["method"] for r in wide})
    by = {}
    for r in wide:
        by[(r["session"], r["state"], r["method"], r["arm"])] = r
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(methods) + 2.6))
    summary = {}
    for ax, metric, title in ((axes[0], "ap", "untyped AP"), (axes[1], "prec", "operating-point precision (untyped)")):
        for i, m in enumerate(methods):
            for st, dy in (("normal", -0.17), ("seizure", 0.17)):
                deltas, kinds = [], set()
                for s in cm.sweep_sessions():
                    f, e = by.get((s, st, m, "full")), by.get((s, st, m, "burst_excluded"))
                    if f is None or e is None:
                        continue
                    if metric == "ap":
                        a, b = f["untyped_ap"], e["untyped_ap"]
                    else:
                        (a, ka), (b, kb) = _op_precision(f), _op_precision(e)
                        kinds.add(ka)
                    if np.isfinite(a) and np.isfinite(b):
                        deltas.append(b - a)
                if not deltas:
                    continue
                deltas = np.asarray(deltas)
                ax.barh(i + dy, deltas.mean(), height=0.3, color=STATE_COL[st], alpha=0.85)
                ax.errorbar(deltas.mean(), i + dy, xerr=[[deltas.mean() - deltas.min()], [deltas.max() - deltas.mean()]],
                            fmt="none", ecolor="0.3", lw=0.8, capsize=2)
                summary.setdefault(m, {})["%s_%s" % (metric, st)] = dict(
                    mean=float(deltas.mean()), min=float(deltas.min()), max=float(deltas.max()),
                    n=int(len(deltas)), kind=",".join(sorted(kinds)) or "threshold-free")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([lab(m) for m in methods], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("burst-excluded minus full (mean, min-max over networks)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="x")
    axes[0].legend(handles=[Line2D([0], [0], color=STATE_COL["normal"], lw=6, label="normal"),
                            Line2D([0], [0], color=STATE_COL["seizure"], lw=6, label="seizure")],
                   fontsize=8, loc="lower right")
    fig.suptitle("What burst exclusion buys each method (%s; precision at the method's own label-free "
                 "point where it has one, else the oracle best-F1 point)" % n_tag, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, "burst_gap.png")
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p, summary


def fig_delay_gate(out):
    files = sorted(glob.glob(os.path.join(cm.summary_dir(), "delay_gate_*.json")))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as fh:
        g = json.load(fh)
    ms = order([m for m, r in g["methods"].items() if r.get("applicable")])
    r = [g["methods"][m]["r"] for m in ms]
    rt = [g["methods"][m]["r_transposed"] for m in ms]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(ms) + 2))
    y = np.arange(len(ms))
    ax.barh(y - 0.18, r, height=0.34, color="#2C6FB5", label="as returned")
    ax.barh(y + 0.18, rt, height=0.34, color="0.7", label="transposed")
    ax.axvline(g["r_min"], color=RED, ls="--", lw=1, label="gate r = %.1f" % g["r_min"])
    ax.set_yticks(y)
    ax.set_yticklabels([lab(m) for m in ms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r: peak-evidence lag vs true conduction delay (true excitatory edges)")
    ax.set_title("Delay-recovery gate: %s/%s n=%d, %d edges" % (g["session"], g["state"], g["n_recordings"],
                                                              g["n_true_exc"]), fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = os.path.join(out, "delay_gate.png")
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p


def fig_te_binwidth(n_tag, out, wide):
    te = {"te_1ms": 1, "te_2ms": 2, "te_5ms": 5, "te_10ms": 10}
    rows = [r for r in wide if r["method"] in te]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, title in ((axes[0], "untyped_ap", "untyped AP"), (axes[1], "exc_ap", "excitatory AP")):
        for st in ("normal", "seizure"):
            for arm, ls in (("full", "-"), ("burst_excluded", "--")):
                xs, ms_, lo, hi = [], [], [], []
                for m, w in sorted(te.items(), key=lambda kv: kv[1]):
                    v = np.array([r[col] for r in rows if r["method"] == m and r["state"] == st and r["arm"] == arm])
                    v = v[np.isfinite(v)]
                    if not len(v):
                        continue
                    xs.append(w)
                    ms_.append(v.mean())
                    lo.append(v.min())
                    hi.append(v.max())
                if xs:
                    ax.plot(xs, ms_, "o" + ls, color=STATE_COL[st], lw=1.6, label="%s, %s" % (st, arm))
                    ax.fill_between(xs, lo, hi, color=STATE_COL[st], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10])
        ax.set_xticklabels([1, 2, 5, 10])
        ax.set_xlabel("bin width (ms)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("average precision (mean, min-max over networks)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Transfer entropy is bin-width dependent: sweep instead of pick (%s)" % n_tag, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out, "te_binwidth.png")
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p


def fig_summary_bars(state, n_tag, out, wide):
    rows = [r for r in wide if r["state"] == state]
    methods = order({r["method"] for r in rows})
    cols = [("exc_auc", "excitatory AUC"), ("exc_ap", "excitatory AP"), ("untyped_ap", "untyped AP"),
            ("oracle_f1_untyped", "oracle best F1 (untyped)"), ("labelfree_precision_untyped", "label-free precision"),
            ("labelfree_recall_untyped", "label-free recall")]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 0.36 * len(methods) + 2.2), sharey=True)
    y = np.arange(len(methods))
    for ax, (col, title) in zip(axes, cols):
        for arm, dy, alpha in (("full", -0.18, 0.9), ("burst_excluded", 0.18, 0.45)):
            for i, m in enumerate(methods):
                v = np.array([r[col] for r in rows if r["method"] == m and r["arm"] == arm], float)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                ax.barh(i + dy, v.mean(), height=0.34, color=COLOR.get(m, "0.5"), alpha=alpha)
                ax.plot([v.min(), v.max()], [i + dy, i + dy], color="0.2", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.grid(alpha=0.3, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab(m) for m in methods], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("%s state, %s: solid = full recordings, pale = burst windows excluded; "
                 "bar = mean over networks, line = min-max" % (state, n_tag), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(out, "summary_bars_%s.png" % state)
    fig.savefig(p, dpi=140, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-recordings", type=int, default=200)
    ap.add_argument("--which", default="all", choices=["all", "cross", "maps"])
    ap.add_argument("--sessions", default="all")
    ap.add_argument("--states", default="normal,seizure")
    ap.add_argument("--methods", default="present")
    ap.add_argument("--representative", default="sweep_c50_seed09")
    a = ap.parse_args()
    n_tag = "n%03d" % a.n_recordings
    root = cm.summary_dir()
    fig_dir = os.path.join(root, "figures")
    map_dir = os.path.join(root, "maps")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    sessions = cm.sweep_sessions() if a.sessions == "all" else [s.strip() for s in a.sessions.split(",")]
    states = [s.strip() for s in a.states.split(",")]
    wide = read_wide(root)
    wide = [r for r in wide if int(r["n_recordings"]) == a.n_recordings]
    present = sorted({r["method"] for r in wide})
    methods = present if a.methods == "present" else registry.resolve(a.methods)
    provenance = {}

    if a.which in ("all", "cross"):
        for st in states:
            if not any(r["state"] == st for r in wide):
                continue
            provenance["roc_pr_%s.png" % st] = fig_roc_pr(st, n_tag, fig_dir, sessions, methods, a.representative)
            for arm in ("full", "burst_excluded"):
                p, rhos = fig_ap_heatmap(st, arm, n_tag, fig_dir, wide)
                if p:
                    provenance["ap_heatmap_%s_%s.png" % (st, arm)] = p
                    provenance["ap_vs_burstiness_rho_%s_%s" % (st, arm)] = rhos
            provenance["summary_bars_%s.png" % st] = fig_summary_bars(st, n_tag, fig_dir, wide)
        p, gap = fig_burst_gap(n_tag, fig_dir, wide)
        provenance["burst_gap.png"] = p
        provenance["burst_gap_summary"] = gap
        provenance["delay_gate.png"] = fig_delay_gate(fig_dir)
        provenance["te_binwidth.png"] = fig_te_binwidth(n_tag, fig_dir, wide)
        with open(os.path.join(root, "figures_provenance.json"), "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=1)
        print("cross-method figures -> %s" % fig_dir)

    if a.which in ("all", "maps"):
        n = 0
        for s in sessions:
            for st in states:
                for m in methods:
                    p = fig_map(s, st, m, n_tag, map_dir)
                    if p:
                        n += 1
        print("%d maps -> %s" % (n, map_dir))


if __name__ == "__main__":
    main()
