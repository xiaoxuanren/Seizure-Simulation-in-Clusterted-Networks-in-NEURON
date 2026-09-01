"""Distance-resolved edge recovery for the label-free sum4 @FDR0.70 result.

Bins candidate pairs by inter-neuron Euclidean distance and reports:
  (a) recall & precision vs distance
  (b) recall vs distance, split within- vs between-cluster (disentangle distance
      from cluster co-activation)
  (c) mean |W| for true edges vs non-edges vs distance (does the jitter-null /
      background inflate at short range? -> common-input signature)
  (d) edge-count density vs distance (true / TP / FP)

Besides the PNG, every array behind the figure is persisted to
results/<state>/glm/distance_recovery.json so the analysis can be aggregated
or re-plotted without touching the fit npz.

    python analysis/glm_distance_recovery.py --session sweep_c50_seed09 --state normal
    (DATASET_SESSION / DATASET_STATE env vars still work as defaults)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_paths import resolve, results_dir  # noqa: E402

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = ("#1f5fd0", "#d0902f", "#2e8b57",
                                          "#c0392b", "#7d3c98", "0.5")


def _lst(a):
    return np.asarray(a, float).tolist()


def run(session, state, make_png=True):
    sd = resolve(session, state)
    results = results_dir(session, state, "glm")
    figs = results_dir(session, state, "figures")

    res = np.load(os.path.join(results, "glm_connectivity_sum4_5ms.npz"),
                  allow_pickle=True)
    W = np.asarray(res["W"], float)
    pred = res["pred_adjacency"].astype(bool)
    A_exc = res["A_exc"].astype(bool)
    A_inh = res["A_inh"].astype(bool)
    cand = res["candidates"].astype(bool)
    N = W.shape[0]
    off = ~np.eye(N, dtype=bool)
    true = (A_exc | A_inh) & off
    TP = pred & true
    FP = pred & ~true & cand

    net = np.load(sorted(glob.glob(os.path.join(sd, "network_*.npz")))[0],
                  allow_pickle=True)
    pos = np.asarray(net["neuron_positions"], float)
    clu = np.asarray(net["cluster_assignments"], int)
    D = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))

    def dvals(mask):
        pre, post = np.where(mask)
        return D[pre, post], clu[pre], clu[post]

    d_true, ct_pre, ct_post = dvals(true)
    d_tp, ctp_pre, ctp_post = dvals(TP)
    d_pred, _, _ = dvals(pred & cand)
    d_fp, _, _ = dvals(FP)

    dmax = np.ceil(max(d_true.max(), np.percentile(d_pred, 99.5)))
    bins = np.linspace(0, dmax, 26)
    xc = 0.5 * (bins[:-1] + bins[1:])

    def H(x):
        h, _ = np.histogram(x, bins=bins)
        return h.astype(float)

    n_true, n_tp, n_pred, n_fp = H(d_true), H(d_tp), H(d_pred), H(d_fp)
    recall = np.where(n_true >= 15, n_tp / np.maximum(n_true, 1), np.nan)
    precision = np.where(n_pred >= 15, n_tp / np.maximum(n_pred, 1), np.nan)

    win_t = ct_pre == ct_post
    win_tp = ctp_pre == ctp_post
    n_true_w, n_true_b = H(d_true[win_t]), H(d_true[~win_t])
    n_tp_w, n_tp_b = H(d_tp[win_tp]), H(d_tp[~win_tp])
    rec_w = np.where(n_true_w >= 10, n_tp_w / np.maximum(n_true_w, 1), np.nan)
    rec_b = np.where(n_true_b >= 10, n_tp_b / np.maximum(n_true_b, 1), np.nan)

    ne_pre, ne_post = np.where(cand & ~true)
    d_ne = D[ne_pre, ne_post]
    w_ne = np.abs(W[ne_pre, ne_post])
    w_true = np.abs(W[true])

    def binmean(x, w):
        s, _ = np.histogram(x, bins=bins, weights=w)
        c, _ = np.histogram(x, bins=bins)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    mw_true = binmean(d_true, w_true)
    mw_ne = binmean(d_ne, w_ne)

    med = float(np.median(d_true))
    near_R = float((d_tp < med).sum() / max((d_true < med).sum(), 1))
    far_R = float((d_tp >= med).sum() / max((d_true >= med).sum(), 1))
    n_rec = int(res["n_recordings"]) if "n_recordings" in res.files else -1

    out = dict(
        session=session, state=state, n_recordings=n_rec, n_neurons=int(N),
        bin_edges=_lst(bins), bin_centers=_lst(xc),
        recall=_lst(recall), precision=_lst(precision),
        mean_absW_true=_lst(mw_true), mean_absW_nonedge=_lst(mw_ne),
        n_true=_lst(n_true), n_tp=_lst(n_tp), n_fp=_lst(n_fp),
        n_pred=_lst(n_pred),
        recall_within=_lst(rec_w), recall_between=_lst(rec_b),
        n_true_within=_lst(n_true_w), n_true_between=_lst(n_true_b),
        median_true_distance=med, near_half_recall=near_R,
        far_half_recall=far_R)
    jp = os.path.join(results, "distance_recovery.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("%s/%s: near %.3f far %.3f -> %s"
          % (session, state, near_R, far_R, jp), flush=True)

    if make_png:
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        ax[0, 0].plot(xc, recall, "o-", color=GREEN, label="recall (TP/true)")
        ax[0, 0].plot(xc, precision, "s--", color=BLUE,
                      label="precision (TP/pred)")
        ax[0, 0].set_title("Recovery vs inter-neuron distance")
        ax[0, 0].set_ylabel("rate")
        ax[0, 0].set_ylim(0, 1.02)
        ax[0, 0].legend(fontsize=9)

        ax[0, 1].plot(xc, rec_w, "o-", color=RED, label="within-cluster edges")
        ax[0, 1].plot(xc, rec_b, "s-", color=BLUE,
                      label="between-cluster edges")
        ax[0, 1].set_title("Recall vs distance: within vs between cluster")
        ax[0, 1].set_ylabel("recall")
        ax[0, 1].set_ylim(0, 1.02)
        ax[0, 1].legend(fontsize=9)

        ax[1, 0].plot(xc, mw_true, "o-", color=GREEN,
                      label="true edges  mean |W|")
        ax[1, 0].plot(xc, mw_ne, "s--", color=GRAY,
                      label="non-edges (background) mean |W|")
        ax[1, 0].set_title(
            "GLM score vs distance  (background inflation = common input)")
        ax[1, 0].set_ylabel("mean |W|")
        ax[1, 0].legend(fontsize=9)

        ax[1, 1].plot(xc, n_true, color=GRAY, lw=1.5, label="true edges")
        ax[1, 1].fill_between(xc, n_tp, color=GREEN, alpha=0.5, label="TP")
        ax[1, 1].plot(xc, n_fp, color=RED, lw=1.2, label="FP")
        ax[1, 1].set_title("Edge-count density vs distance")
        ax[1, 1].set_ylabel("edges per bin")
        ax[1, 1].legend(fontsize=9)

        for a in ax.ravel():
            a.set_xlabel("inter-neuron distance (space units)")
            a.grid(alpha=0.25)
        fig.suptitle(
            "GLM sum4 @FDR0.70 (label-free) — distance-resolved edge "
            "recovery\n%s / %s, %d recordings   |   near-half recall %.2f vs "
            "far-half recall %.2f" % (session, state, n_rec, near_R, far_R),
            fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        op = os.path.join(figs, "glm_distance_recovery.png")
        fig.savefig(op, dpi=130, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("figure -> %s" % op, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session",
                    default=os.environ.get("DATASET_SESSION",
                                           "IC-locked_flagship_200rec"))
    ap.add_argument("--state",
                    default=os.environ.get("DATASET_STATE", "normal"))
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()
    run(a.session, a.state, make_png=not a.no_png)
