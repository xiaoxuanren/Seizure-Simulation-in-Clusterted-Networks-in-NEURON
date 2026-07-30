"""Label-free recovery over the (target-FDR x recording-duration) grid.

For each duration: fit sum4 W + n_surrogates jitter nulls ONCE, then sweep every
target FDR (0.1..1.0) over the stored scores (cheap). Evaluate the resulting
excitatory edges against ground truth. Produces line families vs duration (one
curve per target) and heatmaps over (target x duration).
"""
import os, sys, json, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_glm as sg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Dataset location. Override with the DATASET_SESSION / DATASET_STATE env
# vars, or edit here. `python session_paths.py` lists what is available.
from session_paths import resolve  # noqa: E402
SESSION = resolve(os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec"),
                  os.environ.get("DATASET_STATE", "normal"))
SIZES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TARGETS = np.round(np.arange(0.1, 1.001, 0.1), 2)          # 0.1 .. 1.0
BIN_MS, MAX_LAG, L2, KSUM = 5.0, 6, 2.0, 4
JITTER_MS, N_SURR, SEED = 25.0, 8, 1
JBINS = max(1, int(round(JITTER_MS / BIN_MS)))


def fdr_threshold(obs, null, target):
    cand = np.unique(obs)
    obs_s = np.sort(obs); null_s = np.sort(null.ravel()); ns = null.shape[0]
    n_obs = len(obs_s) - np.searchsorted(obs_s, cand, side="left")
    n_null = (len(null_s) - np.searchsorted(null_s, cand, side="left")) / ns
    fdr = np.where(n_obs > 0, n_null / np.maximum(n_obs, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    return float(cand[ok[0]]) if len(ok) else float("inf")


def sum4_W(Mn, bndn):
    B = sg.fit_B(Mn, bndn, max_lag=MAX_LAG, l2=L2); W = B[:KSUM].sum(0); np.fill_diagonal(W, 0.0); return W


t0 = time.time()
M, bnd = sg.load_session(SESSION, bin_ms=BIN_MS)
gt = sg.load_ground_truth(SESSION)
N = M.shape[0]
cand = ~np.eye(N, dtype=bool)
ye = gt["A_exc"][cand].astype(bool)
n_true = int(ye.sum())
print("N=%d cand=%d true_exc=%d | %.1fs" % (N, cand.sum(), n_true, time.time() - t0), flush=True)

nS, nT = len(SIZES), len(TARGETS)
G = {k: np.full((nT, nS), np.nan) for k in ("realfdr", "recall", "precision", "f1", "tp", "npred")}
for si, n_rec in enumerate(SIZES):
    tf = time.time()
    Mn = M[:, :bnd[n_rec]].tocsr(); bndn = bnd[:n_rec + 1]
    W = sum4_W(Mn, bndn); se = W[cand]
    rng = np.random.default_rng(SEED)
    null = np.stack([sum4_W(sg.jitter(Mn, bndn, JBINS, rng), bndn)[cand] for _ in range(N_SURR)])
    for ti, tgt in enumerate(TARGETS):
        thr = fdr_threshold(se, null, tgt)
        pred = se >= thr if np.isfinite(thr) else np.zeros_like(se, bool)
        TP = int((pred & ye).sum()); FP = int((pred & ~ye).sum()); FN = int((~pred & ye).sum())
        P = TP / max(TP + FP, 1); R = TP / max(TP + FN, 1)
        G["tp"][ti, si] = TP; G["npred"][ti, si] = TP + FP
        G["precision"][ti, si] = P; G["recall"][ti, si] = R
        G["f1"][ti, si] = 2 * P * R / max(P + R, 1e-9)
        G["realfdr"][ti, si] = FP / max(TP + FP, 1)
    print("  n=%3d (%3d min) done %.1fs" % (n_rec, n_rec, time.time() - tf), flush=True)

json.dump({"sizes": SIZES, "targets": TARGETS.tolist(), "n_true_exc": n_true,
           "grids": {k: v.tolist() for k, v in G.items()}},
          open(os.path.join(SESSION, "glm_labelfree_fdr_duration_metrics.json"), "w"), indent=2)

# ---------------- figure ----------------
x = np.array(SIZES, float)
cmap = plt.cm.viridis
tcol = [cmap(i / (nT - 1)) for i in range(nT)]
fig, ax = plt.subplots(2, 3, figsize=(17, 10))

def lines(a, key, ylab, title, hlines=None):
    for ti, tgt in enumerate(TARGETS):
        a.plot(x, G[key][ti], "-o", ms=3, color=tcol[ti], lw=1.3)
    a.set_title(title); a.set_xlabel("recording duration (min)"); a.set_ylabel(ylab); a.grid(alpha=0.25)

lines(ax[0, 0], "realfdr", "realized FDR", "(a) realized FDR vs duration (per target)")
ax[0, 0].axhline(0.10, color="k", ls=":", lw=1)
lines(ax[0, 1], "recall", "recall (exc)", "(b) recall vs duration (per target)")
lines(ax[0, 2], "f1", "F1 (exc)", "(c) F1 vs duration (per target)")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.1, 1.0)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax[0, :], fraction=0.02, pad=0.01); cb.set_label("target FDR")

def heat(a, key, title, cmapname, contour10=False):
    Z = G[key]
    pm = a.pcolormesh(x, TARGETS, Z, cmap=cmapname, shading="nearest")
    fig.colorbar(pm, ax=a, fraction=0.046, pad=0.04)
    if contour10:
        cs = a.contour(x, TARGETS, Z, levels=[0.10], colors="k", linewidths=1.5)
        a.clabel(cs, fmt="realized 0.10", fontsize=8)
    # mark best-F1 target per duration
    if key == "f1":
        best = np.nanargmax(Z, axis=0)
        a.plot(x, TARGETS[best], "w*-", ms=9, lw=1, label="best-F1 target")
        a.legend(fontsize=8, loc="lower right")
    a.set_title(title); a.set_xlabel("recording duration (min)"); a.set_ylabel("target FDR")

heat(ax[1, 0], "realfdr", "(d) realized FDR  (black = true 10% locus)", "RdYlGn_r", contour10=True)
heat(ax[1, 1], "recall", "(e) recall (exc)", "viridis")
heat(ax[1, 2], "f1", "(f) F1 (exc)", "viridis")

fig.suptitle("GLM sum4 (LABEL-FREE) \u2014 recovery over target-FDR (0.1-1.0) \u00d7 recording duration (5-100 min)\n"
             "normal 100-rec flagship; threshold from spike-jitter null (no ground truth)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(SESSION, "glm_labelfree_fdr_duration.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("figure -> %s" % out, flush=True)
# quick text summary: target needed for realized 0.10 at each duration
print("\n target giving realized-FDR closest to 0.10, per duration:")
for si, n_rec in enumerate(SIZES):
    col = G["realfdr"][:, si]
    j = int(np.nanargmin(np.abs(col - 0.10)))
    print("  %3d min: target %.1f -> realFDR %.3f (recall %.2f, F1 %.2f)"
          % (n_rec, TARGETS[j], col[j], G["recall"][j, si], G["f1"][j, si]))
print("TOTAL %.1fs" % (time.time() - t0), flush=True)
