"""Target-FDR calibration curves for the sweep sessions.

One sum4 fit + jitter null per session (full data), then the label-free
threshold rule is evaluated at a grid of target FDRs; each target's REALIZED
FDR (1 - precision vs ground truth) exposes the calibration error and how it
varies with burstiness.

Writes results/<state>/glm/fdr_calibration.json per session and the sweep
figure fdr_calibration.png.

    python analysis/sweep_fdr_calibration.py [session ...]   # default: all sweep_*
"""
import importlib
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

TARGETS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def run_session(session_name):
    os.environ["DATASET_SESSION"] = session_name
    import burstexcl_glm_arm as bx
    importlib.reload(bx)
    import sparse_glm as sg
    from session_paths import results_dir

    sd = bx.SESSION
    M, bnd = sg.load_session(sd, bin_ms=bx.BIN_MS)
    n = M.shape[0]
    gt = sg.load_ground_truth(sd)
    off = ~np.eye(n, dtype=bool)
    true_adj = (gt["A_exc"] | gt["A_inh"]) & off

    W = bx.sum4_W(M, bnd)
    obs = np.abs(W)[off]
    rng = np.random.default_rng(bx.SEED)
    null = np.array([np.abs(bx.sum4_W(sg.jitter(M, bnd, bx.JBINS, rng), bnd))[off]
                     for _ in range(bx.N_SURR)])

    points = []
    for t in TARGETS:
        thr, est = bx.fdr_threshold(obs, null, t)
        pred = np.abs(W) > thr
        np.fill_diagonal(pred, False)
        tp = int((pred & true_adj).sum()); fp = int((pred & ~true_adj).sum())
        fn = int((~pred & true_adj).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        points.append(dict(target=t, est_fdr=est, threshold=thr, n_pred=int(pred.sum()),
                           realized_fdr=1 - P, precision=P, recall=R))
        print("  target %.2f -> realized %.3f (P %.3f R %.3f, %d edges)"
              % (t, 1 - P, P, R, int(pred.sum())), flush=True)

    out = os.path.join(results_dir(session_name, os.environ.get("DATASET_STATE", "normal"),
                                   "glm"), "fdr_calibration.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(session=session_name, targets=points), fh, indent=1)
    return points


def figure(results):
    from session_paths import DATA, resolve
    OUT = os.path.join(DATA, "sweep_summary")
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bursts_all = []
    for session, points in results.items():
        with open(os.path.join(resolve(session, "normal"), "session_metadata.json"),
                  encoding="utf-8") as fh:
            m = json.load(fh)
        bursts_all.append(np.mean([r.get("n_bursts", 0) for r in m["recordings"]]))
    bmax = max(bursts_all) or 1.0
    cmap = plt.get_cmap("viridis")
    for (session, points), b in zip(results.items(), bursts_all):
        ax.plot([p["target"] for p in points], [p["realized_fdr"] for p in points],
                "o-", color=cmap(b / bmax), lw=1.4, ms=4, alpha=0.8)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, bmax))
    fig.colorbar(sm, ax=ax, label="bursts per recording")
    ax.set_xlabel("target (estimated) FDR"); ax.set_ylabel("realized FDR")
    ax.set_title("Jitter-null FDR calibration across %d networks:\n"
                 "conservative when quiet, anti-conservative when bursting" % len(results))
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fdr_calibration.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)
    print("figure -> %s" % os.path.join(OUT, "fdr_calibration.png"))


def main():
    sys.path.insert(0, HERE)
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
