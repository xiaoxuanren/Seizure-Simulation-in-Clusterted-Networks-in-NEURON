"""Per-cluster recovery heatmap for the label-free sum4 @FDR0.70 result.

For every ordered cluster pair (pre-cluster a -> post-cluster b) compute the
recall = TP / (true edges in that block). The diagonal is WITHIN-cluster
recovery, off-diagonal is BETWEEN-cluster. Also summarizes within vs between
recall/precision and shows recall vs block edge count.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal", "20260721_163430")

res = np.load(os.path.join(SD, "glm_connectivity_sum4_5ms.npz"), allow_pickle=True)
pred = res["pred_adjacency"].astype(bool)
A_exc = res["A_exc"].astype(bool); A_inh = res["A_inh"].astype(bool)
cand = res["candidates"].astype(bool)
N = pred.shape[0]
off = ~np.eye(N, dtype=bool)
true = (A_exc | A_inh) & off
TP = pred & true
FP = pred & ~true & cand

net = np.load(os.path.join(SD, "network_20260721_163430.npz"), allow_pickle=True)
clu = np.asarray(net["cluster_assignments"], int)
K = int(clu.max()) + 1

def block_counts(mask):
    pre, post = np.where(mask)
    M = np.zeros((K, K), np.int64)
    np.add.at(M, (clu[pre], clu[post]), 1)
    return M

Btrue = block_counts(true)
Btp = block_counts(TP)
Bpred = block_counts(pred & cand)
Bfp = block_counts(FP)

# order clusters by size for a cleaner block picture (largest first)
sizes = np.bincount(clu, minlength=K)
order = np.argsort(-sizes)
def reo(M): return M[np.ix_(order, order)]
Btrue_o, Btp_o = reo(Btrue), reo(Btp)

recall = np.where(Btrue_o > 0, Btp_o / np.maximum(Btrue_o, 1), np.nan)   # per-block recall
diag = np.eye(K, dtype=bool)

# within vs between aggregates
win_true, win_tp = Btrue[np.diag_indices(K)].sum(), Btp[np.diag_indices(K)].sum()
offd = ~np.eye(K, dtype=bool)
bet_true, bet_tp = Btrue[offd].sum(), Btp[offd].sum()
win_pred, win_fp = Bpred[np.diag_indices(K)].sum(), Bfp[np.diag_indices(K)].sum()
bet_pred, bet_fp = Bpred[offd].sum(), Bfp[offd].sum()
win_R = win_tp / max(win_true, 1); bet_R = bet_tp / max(bet_true, 1)
win_P = win_tp / max(win_pred, 1); bet_P = bet_tp / max(bet_pred, 1)

print("WITHIN : true=%d TP=%d recall=%.3f precision=%.3f" % (win_true, win_tp, win_R, win_P))
print("BETWEEN: true=%d TP=%d recall=%.3f precision=%.3f" % (bet_true, bet_tp, bet_R, bet_P))

# ---- figure ----
fig = plt.figure(figsize=(17, 7.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.35)

# (A) heatmap
axH = fig.add_subplot(gs[:, 0])
cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("0.85")
im = axH.imshow(np.ma.masked_invalid(recall), cmap=cmap, vmin=0, vmax=1, aspect="equal")
axH.set_title("Per-block recall  (TP / true edges)\ndiagonal = within-cluster, off-diagonal = between", fontsize=11)
axH.set_xlabel("post-cluster (sorted by size)"); axH.set_ylabel("pre-cluster (sorted by size)")
cb = fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04); cb.set_label("recall")

# (B) within vs between bars
axB = fig.add_subplot(gs[0, 1])
x = np.arange(2); w = 0.38
axB.bar(x - w/2, [win_R, bet_R], w, color="#2e8b57", label="recall")
axB.bar(x + w/2, [win_P, bet_P], w, color="#1f5fd0", label="precision")
for xi, (r, p) in zip(x, [(win_R, win_P), (bet_R, bet_P)]):
    axB.text(xi - w/2, r + .02, "%.2f" % r, ha="center", fontsize=9)
    axB.text(xi + w/2, p + .02, "%.2f" % p, ha="center", fontsize=9)
axB.set_xticks(x); axB.set_xticklabels(["within\ncluster", "between\nclusters"])
axB.set_ylim(0, 1.05); axB.set_ylabel("rate"); axB.legend(fontsize=8)
axB.set_title("Within vs between recovery", fontsize=11)

# (C) true-edge counts within vs between
axC = fig.add_subplot(gs[1, 1])
axC.bar([0, 1], [win_true, bet_true], 0.5, color="0.6")
axC.bar([0, 1], [win_tp, bet_tp], 0.5, color="#2e8b57")
for xi, (t, tp) in zip([0, 1], [(win_true, win_tp), (bet_true, bet_tp)]):
    axC.text(xi, t + 100, "%d/%d" % (tp, t), ha="center", fontsize=9)
axC.set_xticks([0, 1]); axC.set_xticklabels(["within", "between"])
axC.set_ylabel("edges (TP green / true gray)"); axC.set_title("Edge counts", fontsize=11)

# (D) recall vs block size, within vs between
axD = fig.add_subplot(gs[:, 2])
pre_c, post_c = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
flat_true = Btrue.ravel(); flat_tp = Btp.ravel()
m = flat_true > 0
rr = flat_tp[m] / flat_true[m]
sz = flat_true[m]
isdiag = (pre_c.ravel()[m] == post_c.ravel()[m])
axD.scatter(sz[~isdiag], rr[~isdiag], s=14, c="#d0902f", alpha=0.6, label="between-cluster block")
axD.scatter(sz[isdiag], rr[isdiag], s=30, c="#2e8b57", edgecolors="k", lw=0.4, label="within-cluster block")
axD.set_xscale("log"); axD.set_xlabel("true edges in block (log)"); axD.set_ylabel("block recall")
axD.set_ylim(-0.02, 1.02); axD.set_title("Recall vs block density", fontsize=11); axD.legend(fontsize=8)
axD.grid(alpha=0.25)

fig.suptitle("GLM sum4 @FDR0.70 (label-free) \u2014 per-cluster edge recovery  |  "
             "within-cluster recall %.2f vs between-cluster recall %.2f" % (win_R, bet_R),
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(REPO, "figures", "glm_cluster_recovery_heatmap_sum4_100rec.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("figure -> %s" % out)
