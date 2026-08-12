"""Label-free result figure for the CURRENT sum4 @ target-fdr 0.70 run (100 recs).

Predicted-vs-true wiring from glm_connectivity_sum4_5ms.npz:
  (a) spatial map: TP green / FP red / FN orange, inhibitory neurons circled
  (b) cluster-sorted adjacency: TP green / FP red / FN orange
Rebuilds the style of the old figures/glm_predicted_topology_labelfree_normal.png
(which was a stale 236-neuron lag-1 run) for the 926-neuron flagship.
"""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_paths import resolve, results_dir  # noqa: E402

_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SD = resolve(_S, _T)
RESULTS = results_dir(_S, _T, "glm")
FIGS = results_dir(_S, _T, "figures")

res = np.load(os.path.join(RESULTS, "glm_connectivity_sum4_5ms.npz"), allow_pickle=True)
W = res["W"]; pred = res["pred_adjacency"].astype(bool)
A_exc = res["A_exc"].astype(bool); A_inh = res["A_inh"].astype(bool)
cand = res["candidates"].astype(bool)
N = W.shape[0]
off = ~np.eye(N, dtype=bool)
true = (A_exc | A_inh) & off

net = np.load(sorted(glob.glob(os.path.join(SD, "network_*.npz")))[0], allow_pickle=True)
pos = np.asarray(net["neuron_positions"], float)
clu = np.asarray(net["cluster_assignments"], int)
is_inh = np.asarray(net["neuron_is_inhibitory"]).astype(bool)

TP = pred & true
FP = pred & ~true & cand
FN = true & ~pred
tp, fp, fn = int(TP.sum()), int(FP.sum()), int(FN.sum())
P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F1 = 2 * P * R / max(P + R, 1e-9)
GREEN, RED, ORANGE = (0.18, 0.55, 0.34), (0.75, 0.23, 0.16), (0.94, 0.63, 0.19)

rng = np.random.default_rng(0)
def segs(mask, cap=None):
    pre, post = np.where(mask)
    if cap and len(pre) > cap:
        k = rng.choice(len(pre), cap, replace=False); pre, post = pre[k], post[k]
    return np.stack([pos[pre], pos[post]], axis=1)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 8.6))

# (a) spatial map -- FN faint (subsampled), FP red, TP green (subsampled for legibility)
axA.add_collection(LineCollection(segs(FN, cap=3500), colors=[ORANGE], alpha=0.05, lw=0.4))
axA.add_collection(LineCollection(segs(FP), colors=[RED], alpha=0.30, lw=0.4))
axA.add_collection(LineCollection(segs(TP, cap=6000), colors=[GREEN], alpha=0.12, lw=0.4))
axA.scatter(pos[:, 0], pos[:, 1], s=9, c="0.45", zorder=3)
axA.scatter(pos[is_inh, 0], pos[is_inh, 1], s=34, facecolors="none",
            edgecolors="k", lw=0.8, zorder=4)
from matplotlib.lines import Line2D
axA.legend(handles=[
    Line2D([0], [0], color=GREEN, lw=2, label="TP  (%d correct)" % tp),
    Line2D([0], [0], color=RED, lw=2, label="FP  (%d false)" % fp),
    Line2D([0], [0], color=ORANGE, lw=2, ls=":", label="FN  (%d missed)" % fn),
    Line2D([0], [0], marker="o", mfc="none", mec="k", ls="", label="inhibitory neuron"),
], loc="upper right", fontsize=9)
axA.set_title("(a) label-free predicted topology  (P=%.2f  R=%.2f  F1=%.2f)\n"
              "edges subsampled for legibility" % (P, R, F1), fontsize=11)
axA.set_xticks([]); axA.set_yticks([]); axA.set_aspect("equal")
axA.autoscale()

# (b) cluster-sorted adjacency image
order = np.argsort(clu, kind="stable")
ix = np.ix_(order, order)
img = np.ones((N, N, 3))
img[FN[ix]] = ORANGE
img[FP[ix]] = RED
img[TP[ix]] = GREEN
axB.imshow(img, interpolation="nearest", aspect="equal")
axB.set_title("(b) adjacency (cluster-sorted): TP green, FP red, FN orange", fontsize=11)
axB.set_xlabel("post"); axB.set_ylabel("pre")

fig.suptitle("GLM sum4 @FDR0.70 (label-free) \u2014 predicted vs true wiring, normal 100-rec flagship  |  "
             "%d/%d edges recovered  (P=%.2f R=%.2f F1=%.2f)"
             % (tp, int(true.sum()), P, R, F1), fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(FIGS, "glm_predicted_topology_labelfree_sum4_100rec.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("TP=%d FP=%d FN=%d | P=%.3f R=%.3f F1=%.3f | true_edges=%d" % (tp, fp, fn, P, R, F1, int(true.sum())))
print("figure -> %s" % out)
