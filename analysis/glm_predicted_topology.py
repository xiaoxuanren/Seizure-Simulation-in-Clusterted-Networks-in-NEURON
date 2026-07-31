"""Predicted vs true wiring: spatial map + cluster-sorted adjacency, TP/FP/FN.

Left  -- neurons in space, every edge drawn: true positives green, false
         positives red, missed edges orange dotted. Inhibitory neurons ringed.
Right -- the same three sets as a cluster-sorted adjacency matrix, which makes
         the block structure and where the misses fall visible.

Reads ``glm_connectivity_sum4_5ms.npz`` written by ``glm_fit_current.py`` -- the
label-free sum4 @FDR 0.70 result including the A1 E/I typing fix, so
``pred_adjacency`` carries a real inhibitory layer rather than the handful of
edges the pre-fix sign rule allowed.

Extracted from notebooks/neuron_network_simulation.ipynb so it can be re-run at
any dataset size without opening the notebook.

    python glm_predicted_topology.py
    DATASET_STATE=seizure python glm_predicted_topology.py
"""

import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_paths import resolve, results_dir  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SD = resolve(_S, _T)
RESULTS = results_dir(_S, _T, "glm")
FIGS = results_dir(_S, _T, "figures")

GREEN, RED, ORANGE = "#1a9850", "#d73027", "#f0a000"


def main():
    res = np.load(os.path.join(RESULTS, "glm_connectivity_sum4_5ms.npz"),
                  allow_pickle=True)
    pred = res["pred_adjacency"].astype(bool)
    cand = res["candidates"].astype(bool)
    A_exc = res["A_exc"].astype(bool)
    A_inh = res["A_inh"].astype(bool)
    n_rec = int(res["n_recordings"]) if "n_recordings" in res.files else -1

    net = np.load(sorted(glob.glob(os.path.join(SD, "network_*.npz")))[0],
                  allow_pickle=True)
    pos = np.asarray(net["neuron_positions"], float)
    ca = np.asarray(net["cluster_assignments"]).astype(int)
    inh_true = np.asarray(net["neuron_is_inhibitory"]).astype(bool)

    N = len(pos)
    off = ~np.eye(N, dtype=bool)
    A_edge = (A_exc | A_inh) & off
    tp = pred & A_edge
    fp = pred & ~A_edge & cand
    fn = A_edge & ~pred
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    Pr = TP / (TP + FP) if TP + FP else 0.0
    Rc = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * Pr * Rc / (Pr + Rc) if Pr + Rc else 0.0

    def segs(m):
        return [[pos[i], pos[j]] for i, j in np.argwhere(m)]

    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))

    a = ax[0]
    a.add_collection(LineCollection(segs(fn), colors=ORANGE, lw=0.3, alpha=0.45,
                                    linestyles="dotted"))
    a.add_collection(LineCollection(segs(tp), colors=GREEN, lw=0.55, alpha=0.85))
    a.add_collection(LineCollection(segs(fp), colors=RED, lw=0.55, alpha=0.85))
    a.scatter(pos[~inh_true, 0], pos[~inh_true, 1], s=13, c="#888", zorder=3)
    a.scatter(pos[inh_true, 0], pos[inh_true, 1], s=30, facecolors="none",
              edgecolors="k", lw=1.0, zorder=4)
    a.legend(handles=[
        Line2D([0], [0], color=GREEN, lw=2, label="TP (%d correct)" % TP),
        Line2D([0], [0], color=RED, lw=2, label="FP (%d false)" % FP),
        Line2D([0], [0], color=ORANGE, lw=2, ls=":", label="FN (%d missed)" % FN),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="k", label="inhibitory neuron"),
    ], fontsize=10, loc="upper right")
    a.set_title("predicted topology, label-free  (P=%.2f  R=%.2f  F1=%.2f)"
                % (Pr, Rc, F1), fontsize=12)
    a.set_xticks([])
    a.set_yticks([])
    a.autoscale_view()

    order = np.argsort(ca, kind="stable")
    M = np.zeros((N, N))
    M[fn] = 1
    M[fp] = 2
    M[tp] = 3
    b = ax[1]
    b.imshow(M[np.ix_(order, order)],
             cmap=ListedColormap(["white", ORANGE, RED, GREEN]),
             vmin=0, vmax=3, interpolation="nearest", aspect="equal")
    b.set_title("adjacency, cluster-sorted: TP green, FP red, FN orange",
                fontsize=12)
    b.set_xlabel("post")
    b.set_ylabel("pre")

    fig.suptitle("%s / %s, %d recordings - predicted vs true wiring: "
                 "%d of %d edges recovered"
                 % (_S, _T, n_rec, TP, int(A_edge.sum())),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGS, "glm_predicted_topology.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")

    print("%s / %s, %d recordings" % (_S, _T, n_rec))
    print("  true edges %d | TP %d  FP %d  FN %d" % (A_edge.sum(), TP, FP, FN))
    print("  precision %.4f  recall %.4f  F1 %.4f" % (Pr, Rc, F1))
    print("  figure -> %s" % out)


if __name__ == "__main__":
    main()
