"""Performance vs amount of data, per sweep session.

Refits the sum4 GLM on the first n recordings for n in the grid, reporting
exc AUC/AP (threshold-free) and precision/recall at the label-free FDR-0.70
operating point (jitter null per subset). The expensive module of the suite:
~10-20 min per network.

Writes results/<state>/glm/scaling.json per session and the sweep figure
scaling_curves.png.

    python analysis/sweep_scaling.py [session ...]           # default: all sweep_*
"""
import importlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

GRID = [10, 20, 30, 50, 75, 100, 150, 200]


def subset_first(M, bnd, n):
    end = bnd[n]
    return M[:, :end], bnd[:n + 1]


def run_session(session_name):
    os.environ["DATASET_SESSION"] = session_name
    import burstexcl_glm_arm as bx
    importlib.reload(bx)
    import sparse_glm as sg
    from session_paths import results_dir

    M, bnd = sg.load_session(bx.SESSION, bin_ms=bx.BIN_MS)
    n = M.shape[0]
    gt = sg.load_ground_truth(bx.SESSION)
    off = ~np.eye(n, dtype=bool)
    true_adj = ((gt["A_exc"] | gt["A_inh"]) & off)
    y = true_adj[off]

    points = []
    for k in GRID:
        if k > len(bnd) - 1:
            break
        Mk, bk = subset_first(M, bnd, k)
        W = bx.sum4_W(Mk, bk)
        obs = np.abs(W)[off]
        auc = float(roc_auc_score(y, obs))
        ap = float(average_precision_score(y, obs))
        rng = np.random.default_rng(bx.SEED)
        null = np.array([np.abs(bx.sum4_W(sg.jitter(Mk, bk, bx.JBINS, rng), bk))[off]
                         for _ in range(bx.N_SURR)])
        thr, est = bx.fdr_threshold(obs, null, bx.TARGET_FDR)
        pred = np.abs(W) > thr
        np.fill_diagonal(pred, False)
        tp = int((pred & true_adj).sum()); fp = int((pred & ~true_adj).sum())
        fn = int((~pred & true_adj).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        points.append(dict(n_recordings=k, auc=auc, ap=ap, precision=P, recall=R,
                           n_pred=int(pred.sum())))
        print("  n=%3d: AUC %.3f AP %.3f | P %.3f R %.3f" % (k, auc, ap, P, R),
              flush=True)

    out = os.path.join(results_dir(session_name, os.environ.get("DATASET_STATE", "normal"),
                                   "glm"), "scaling.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(session=session_name, grid=points), fh, indent=1)
    return points


def figure(results):
    from session_paths import DATA, resolve
    OUT = os.path.join(DATA, "sweep_summary")
    os.makedirs(OUT, exist_ok=True)
    bursts = {}
    for session in results:
        with open(os.path.join(resolve(session, "normal"), "session_metadata.json"),
                  encoding="utf-8") as fh:
            m = json.load(fh)
        bursts[session] = float(np.mean([r.get("n_bursts", 0) for r in m["recordings"]]))
    bmax = max(bursts.values()) or 1.0
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for session, points in results.items():
        c = cmap(bursts[session] / bmax)
        xs = [p["n_recordings"] for p in points]
        ax1.plot(xs, [p["auc"] for p in points], "o-", color=c, ms=4, lw=1.3, alpha=0.85)
        ax2.plot(xs, [p["precision"] for p in points], "o-", color=c, ms=4, lw=1.3, alpha=0.85)
    for ax, lab in ((ax1, "excitatory+inhibitory AUC"), (ax2, "precision @ FDR 0.70")):
        ax.set_xlabel("recordings used (60 s each)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(GRID)
        ax.set_xticklabels(GRID)
    ax1.set_title("Ranking quality saturates early...")
    ax2.set_title("...but the operating point needs the data (color = burstiness)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, bmax))
    fig.colorbar(sm, ax=ax2, label="bursts per recording")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scaling_curves.png"), dpi=140, facecolor="white")
    plt.close(fig)
    print("figure -> %s" % os.path.join(OUT, "scaling_curves.png"))


def main():
    from session_paths import list_sessions
    sessions = sys.argv[1:] or sorted(s for s in list_sessions() if s.startswith("sweep_"))
    results = {}
    for s in sessions:
        print(s, flush=True)
        results[s] = run_session(s)
    if len(results) > 1:
        figure(results)


if __name__ == "__main__":
    main()
