"""GLM edge-recovery vs recording duration (sum4 readout, ground-truth operating points).

Fits the sum4 GLM once per data size using the memory-frugal sparse solver, then
reads TP / precision / recall / F1 / AUC / AP off the ground-truth PR curve at two
operating points:
  * best-F1     : threshold that maximizes F1 at that data size
  * @10% FDR    : loosest threshold with precision >= 0.90 (FP/(TP+FP) <= 0.10)
No jitter surrogates needed (we have ground truth), so this is cheap.
"""
import os, sys, json, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
from session_paths import results_dir  # noqa: E402
_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
RESULTS = results_dir(_S, _T, "glm")
FIGS = results_dir(_S, _T, "figures")
SIZES = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4          # sum4 = sum of lags 1..4
REC_MIN = 1.0                                       # each recording = 60 s = 1 min
PR_SNAPSHOTS = (10, 50, 100)

t0 = time.time()
print("loading all recordings (sparse) ...", flush=True)
M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)     # M [N, T_total] CSR, bnd len = n_rec+1
gt = sg.load_ground_truth(SESSION)
N = M.shape[0]
cand = ~np.eye(N, dtype=bool)
ye = gt["A_exc"][cand].astype(bool)
yi = gt["A_inh"][cand].astype(bool)
n_true_exc, n_true_inh = int(ye.sum()), int(yi.sum())
print("N=%d | candidates=%d | true exc=%d inh=%d | loaded %.1fs"
      % (N, cand.sum(), n_true_exc, n_true_inh, time.time() - t0), flush=True)


def op_points(prec, rec):
    """From a PR curve return (best-F1 dict, @10%FDR dict)."""
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    bi = int(np.argmax(f1))
    best = dict(P=float(prec[bi]), R=float(rec[bi]), F1=float(f1[bi]))
    m90 = prec >= 0.90
    if m90.any():
        idxs = np.where(m90)[0]
        j = idxs[int(np.argmax(rec[idxs]))]
        fdr10 = dict(P=float(prec[j]), R=float(rec[j]))
    else:
        fdr10 = dict(P=float(prec[0]), R=0.0)
    return best, fdr10


rows, pr_store = [], {}
for n_rec in SIZES:
    tf = time.time()
    T = bnd[n_rec]
    Mn = M[:, :T].tocsr()
    B = sg.fit_B(Mn, bnd[:n_rec + 1], max_lag=MAX_LAG, l2=L2)   # [max_lag, N, N]
    W = B[:KSUM].sum(0)
    np.fill_diagonal(W, 0.0)
    se = W[cand]

    auc_e = roc_auc_score(ye, se)
    ap_e = average_precision_score(ye, se)
    auc_i = roc_auc_score(yi, -se)
    prec, rec, _ = precision_recall_curve(ye, se)
    best, fdr10 = op_points(prec, rec)

    def counts(P, R):
        tp = R * n_true_exc
        fp = tp * (1 - P) / max(P, 1e-12)
        return tp, fp, tp + fp

    tp_bf, fp_bf, np_bf = counts(best["P"], best["R"])
    tp_90, fp_90, np_90 = counts(fdr10["P"], fdr10["R"])
    row = dict(n_rec=n_rec, minutes=n_rec * REC_MIN, auc_e=auc_e, ap_e=ap_e, auc_i=auc_i,
               bestP=best["P"], bestR=best["R"], bestF1=best["F1"],
               tp_bf=tp_bf, fp_bf=fp_bf, npred_bf=np_bf,
               r90=fdr10["R"], p90=fdr10["P"], tp_90=tp_90, fp_90=fp_90)
    rows.append(row)
    if n_rec in PR_SNAPSHOTS:
        pr_store[n_rec] = (rec.tolist(), prec.tolist())
    print("  n=%3d (%3.0f min) exc AUC=%.3f AP=%.3f | bestF1=%.3f (P=%.2f R=%.2f TP=%d) "
          "| @10%%FDR R=%.2f TP=%d | %.1fs"
          % (n_rec, n_rec * REC_MIN, auc_e, ap_e, best["F1"], best["P"], best["R"], tp_bf,
             fdr10["R"], tp_90, time.time() - tf), flush=True)

# ---- persist metrics ----
out_json = os.path.join(RESULTS, "glm_scaling_metrics.json")
with open(out_json, "w") as fh:
    json.dump({"session": SESSION, "readout": "sum4", "bin_ms": BIN_MS, "max_lag": MAX_LAG,
               "l2": L2, "n_true_exc": n_true_exc, "n_true_inh": n_true_inh, "rows": rows},
              fh, indent=2)
print("metrics -> %s" % out_json, flush=True)

# ---- plot ----
x = np.array([r["minutes"] for r in rows])
g = lambda k: np.array([r[k] for r in rows])
BLUE, ORANGE, GREEN, RED, PURPLE = "#1f5fd0", "#d0902f", "#2e8b57", "#c0392b", "#7d3c98"

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# A: TP vs duration
ax[0, 0].plot(x, g("tp_bf"), "o-", color=BLUE, label="best-F1 threshold")
ax[0, 0].plot(x, g("tp_90"), "s--", color=ORANGE, label="@10% FDR (P\u22650.90)")
ax[0, 0].axhline(n_true_exc, color="gray", ls=":", lw=1, label="all true exc (%d)" % n_true_exc)
ax[0, 0].set_title("True positives vs recording duration")
ax[0, 0].set_ylabel("TP (true excitatory edges recovered)")

# B: precision
ax[0, 1].plot(x, g("bestP"), "o-", color=BLUE, label="best-F1 threshold")
ax[0, 1].plot(x, g("p90"), "s--", color=ORANGE, label="@10% FDR")
ax[0, 1].axhline(0.90, color="gray", ls=":", lw=1)
ax[0, 1].set_title("Precision = TP / total predicted")
ax[0, 1].set_ylabel("precision"); ax[0, 1].set_ylim(0.5, 1.02)

# C: recall
ax[0, 2].plot(x, g("bestR"), "o-", color=BLUE, label="best-F1 threshold")
ax[0, 2].plot(x, g("r90"), "s--", color=ORANGE, label="@10% FDR")
ax[0, 2].set_title("Recall = TP / total true edges")
ax[0, 2].set_ylabel("recall"); ax[0, 2].set_ylim(0, 1.0)

# D: F1
ax[1, 0].plot(x, g("bestF1"), "o-", color=GREEN)
ax[1, 0].set_title("Best-F1 vs recording duration")
ax[1, 0].set_ylabel("F1 (excitatory)"); ax[1, 0].set_ylim(0, 1.0)

# E: ranking quality (threshold-free)
ax[1, 1].plot(x, g("auc_e"), "o-", color=BLUE, label="exc AUC")
ax[1, 1].plot(x, g("ap_e"), "s-", color=PURPLE, label="exc AP")
ax[1, 1].plot(x, g("auc_i"), "^--", color=RED, label="inh AUC")
ax[1, 1].set_title("Ranking quality vs duration")
ax[1, 1].set_ylabel("AUC / AP"); ax[1, 1].set_ylim(0, 1.02)

# F: PR curves at a few sizes
for nr, col in zip(PR_SNAPSHOTS, (RED, ORANGE, BLUE)):
    if nr in pr_store:
        rc, pc = pr_store[nr]
        ax[1, 2].plot(rc, pc, color=col, lw=2, label="%d rec (%d min)" % (nr, int(nr * REC_MIN)))
ax[1, 2].set_title("Precision-Recall curve (excitatory)")
ax[1, 2].set_xlabel("recall"); ax[1, 2].set_ylabel("precision")
ax[1, 2].set_xlim(0, 1); ax[1, 2].set_ylim(0, 1.02)

for a in ax.ravel():
    a.grid(alpha=0.25)
    a.legend(fontsize=8, loc="best")
for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]):
    a.set_xlabel("recording duration (min)   [= number of 60 s recordings]")

fig.suptitle("GLM (sum4) excitatory edge recovery vs recording duration\n"
             "%s  |  N=%d, %d candidate pairs, %d true exc edges"
             % (os.path.basename(SESSION), N, int(cand.sum()), n_true_exc), fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_png = os.path.join(FIGS, "glm_scaling_vs_duration.png")
fig.savefig(out_png, dpi=130, facecolor="white", bbox_inches="tight")
print("figure -> %s" % out_png, flush=True)
print("TOTAL %.1fs" % (time.time() - t0), flush=True)
