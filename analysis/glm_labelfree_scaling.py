"""LABEL-FREE scaling: sum4 jitter-FDR (target 0.70) edge recovery vs duration.

Unlike glm_scaling.py (which used oracle/ground-truth thresholds), this runs the
ACTUAL label-free pipeline at each data size: fit sum4, build the spike-jitter
null (n_surrogates), pick the threshold at target_fdr WITHOUT ground truth, then
evaluate the resulting excitatory edges against truth. Overlays the oracle curves
from glm_scaling_metrics.json for the achievable-vs-operating gap.
"""
import os, sys, json, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sparse_glm as sg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SESSION = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal", "20260721_163430")
SIZES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
TARGET_FDR, JITTER_MS, N_SURR, SEED = 0.70, 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))
REC_MIN = 1.0


def fdr_threshold(obs, null_scores, target):
    cand = np.unique(obs)
    obs_s = np.sort(obs); null_s = np.sort(null_scores.ravel())
    n_surr = null_scores.shape[0]
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / n_surr
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    return float(cand[ok[0]]) if len(ok) else float("inf")


def sum4_W(Mn, bndn):
    B = sg.fit_B(Mn, bndn, max_lag=MAX_LAG, l2=L2)
    W = B[:KSUM].sum(0); np.fill_diagonal(W, 0.0)
    return W


t0 = time.time()
print("loading (sparse) ...", flush=True)
M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
gt = sg.load_ground_truth(SESSION)
N = M.shape[0]
cand = ~np.eye(N, dtype=bool)
ye = gt["A_exc"][cand].astype(bool)
n_true_exc = int(ye.sum())
print("N=%d cand=%d true_exc=%d | %.1fs" % (N, cand.sum(), n_true_exc, time.time() - t0), flush=True)

rows = []
for n_rec in SIZES:
    tf = time.time()
    T = bnd[n_rec]; Mn = M[:, :T].tocsr(); bndn = bnd[:n_rec + 1]
    W = sum4_W(Mn, bndn)
    se = W[cand]
    rng = np.random.default_rng(SEED)
    null = np.stack([sum4_W(sg.jitter(Mn, bndn, JBINS, rng), bndn)[cand] for _ in range(N_SURR)])
    thr = fdr_threshold(se, null, TARGET_FDR)
    pred = se >= thr if np.isfinite(thr) else np.zeros_like(se, bool)
    TP = int((pred & ye).sum()); FP = int((pred & ~ye).sum()); FN = int((~pred & ye).sum())
    P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-9); rfdr = FP / max(TP + FP, 1)
    rows.append(dict(n_rec=n_rec, minutes=n_rec * REC_MIN, thr=thr, n_pred=TP + FP,
                     TP=TP, FP=FP, FN=FN, precision=P, recall=R, f1=F1, realized_fdr=rfdr))
    print("  n=%3d (%3.0f min) thr=%.4f n_pred=%5d TP=%5d FP=%4d | P=%.3f R=%.3f F1=%.3f realFDR=%.3f | %.1fs"
          % (n_rec, n_rec * REC_MIN, thr, TP + FP, TP, FP, P, R, F1, rfdr, time.time() - tf), flush=True)

out_json = os.path.join(SESSION, "glm_labelfree_scaling_metrics.json")
json.dump({"readout": "sum4", "target_fdr": TARGET_FDR, "jitter_ms": JITTER_MS,
           "n_surrogates": N_SURR, "n_true_exc": n_true_exc, "rows": rows},
          open(out_json, "w"), indent=2)
print("metrics -> %s" % out_json, flush=True)

# oracle overlay (best-F1 / @10%FDR) from the earlier ground-truth sweep
orc = None
op = os.path.join(SESSION, "glm_scaling_metrics.json")
if os.path.exists(op):
    orc = json.load(open(op))["rows"]

x = np.array([r["minutes"] for r in rows]); g = lambda k: np.array([r[k] for r in rows])
GREEN, BLUE, ORANGE, RED = "#2e8b57", "#1f5fd0", "#d0902f", "#c0392b"
def oc(k):
    return (np.array([r["minutes"] for r in orc]), np.array([r[k] for r in orc])) if orc else (None, None)

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

ax[0, 0].plot(x, g("TP"), "o-", color=GREEN, label="label-free sum4 @0.70")
if orc: ax[0, 0].plot(*oc("tp_bf"), "--", color=BLUE, alpha=.7, label="oracle best-F1")
if orc: ax[0, 0].plot(*oc("tp_90"), ":", color=ORANGE, alpha=.7, label="oracle @10% FDR")
ax[0, 0].axhline(n_true_exc, color="gray", ls=":", lw=1, label="all true exc (%d)" % n_true_exc)
ax[0, 0].set_title("True positives vs duration (label-free)"); ax[0, 0].set_ylabel("TP")

ax[0, 1].plot(x, g("precision"), "o-", color=GREEN, label="label-free")
if orc: ax[0, 1].plot(*oc("bestP"), "--", color=BLUE, alpha=.7, label="oracle best-F1")
ax[0, 1].axhline(0.90, color="gray", ls=":", lw=1, label="P=0.90")
ax[0, 1].set_title("Precision = TP / total predicted"); ax[0, 1].set_ylabel("precision"); ax[0, 1].set_ylim(0.5, 1.02)

ax[0, 2].plot(x, g("recall"), "o-", color=GREEN, label="label-free")
if orc: ax[0, 2].plot(*oc("bestR"), "--", color=BLUE, alpha=.7, label="oracle best-F1")
if orc: ax[0, 2].plot(*oc("r90"), ":", color=ORANGE, alpha=.7, label="oracle @10% FDR")
ax[0, 2].set_title("Recall = TP / total true"); ax[0, 2].set_ylabel("recall"); ax[0, 2].set_ylim(0, 1.0)

ax[1, 0].plot(x, g("f1"), "o-", color=GREEN, label="label-free")
if orc: ax[1, 0].plot(*oc("bestF1"), "--", color=BLUE, alpha=.7, label="oracle best-F1")
ax[1, 0].set_title("F1 vs duration"); ax[1, 0].set_ylabel("F1"); ax[1, 0].set_ylim(0, 1.0)

ax[1, 1].plot(x, g("realized_fdr"), "o-", color=RED, label="realized FDR @ target 0.70")
ax[1, 1].axhline(0.10, color="gray", ls=":", lw=1, label="10% ref")
ax[1, 1].set_title("Realized FDR vs duration (fixed target 0.70)"); ax[1, 1].set_ylabel("realized FDR"); ax[1, 1].set_ylim(0, max(0.2, g("realized_fdr").max() * 1.1))

ax[1, 2].plot(x, g("n_pred"), "o-", color=GREEN, label="predicted edges")
ax[1, 2].plot(x, g("TP"), "s-", color="#555", label="of which TP")
ax[1, 2].set_title("Predicted edges vs duration"); ax[1, 2].set_ylabel("count")

for a in ax.ravel():
    a.set_xlabel("recording duration (min)  [= # of 60 s recordings]"); a.grid(alpha=0.25); a.legend(fontsize=8)

fig.suptitle("GLM sum4 @FDR0.70 (LABEL-FREE) \u2014 excitatory edge recovery vs recording duration\n"
             "threshold chosen from spike-jitter null (no ground truth); dashed = oracle upper bounds",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_png = os.path.join(SESSION, "glm_labelfree_scaling_vs_duration.png")
fig.savefig(out_png, dpi=130, facecolor="white", bbox_inches="tight")
print("figure -> %s" % out_png, flush=True)
print("TOTAL %.1fs" % (time.time() - t0), flush=True)
