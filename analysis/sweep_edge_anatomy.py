"""Edge-level anatomy of the sweep's inference results (no refits needed).

From each session's saved glm_connectivity_sum4_5ms.npz + network npz:
  * detection probability vs true synaptic weight (recall by weight decile)
  * per-neuron degree recovery (true vs predicted in/out degree, Pearson r)
  * false-positive anatomy: pair distance and shared-common-input count of FP
    pairs vs recovered-TP pairs

Writes results/<state>/glm/edge_anatomy.json per session, plus the sweep
figures weight_vs_detection.png, degree_recovery.png, fp_anatomy.png into
"NEURON data parallel/sweep_summary/".

    python analysis/sweep_edge_anatomy.py [session ...]      # default: all sweep_*
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from session_paths import DATA, resolve, results_dir, list_sessions  # noqa: E402

STATE = os.environ.get("DATASET_STATE", "normal")
OUT = os.path.join(DATA, "sweep_summary")
C50, C40 = "#1f5fd0", "#c0392b"


def analyze(session):
    sd = resolve(session, STATE)
    glm_dir = results_dir(session, STATE, "glm")
    res = np.load(os.path.join(glm_dir, "glm_connectivity_sum4_5ms.npz"), allow_pickle=True)
    net = np.load(glob.glob(os.path.join(sd, "network_*.npz"))[0], allow_pickle=True)

    pred = res["pred_adjacency"].astype(bool)
    A = res["A_exc"].astype(bool) | res["A_inh"].astype(bool)
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    true = A & off
    np.fill_diagonal(pred, False)

    # --- detection vs |true weight|, per E/I class (deciles WITHIN class:
    #     inhibitory weights are larger in magnitude and detected worse, so a
    #     combined decile axis conflates weight with class) -------------------
    Wtrue = np.zeros((n, n))
    for row in net["connections"]:
        Wtrue[int(row[0]), int(row[1])] = abs(float(row[2]))

    def decile_detection(mask):
        w = Wtrue[mask]
        detected = pred[mask]
        edges = np.quantile(w, np.linspace(0, 1, 11))
        out = []
        for i in range(10):
            m = (w >= edges[i]) & (w <= edges[i + 1] if i == 9 else w < edges[i + 1])
            out.append(float(detected[m].mean()) if m.any() else float("nan"))
        return out

    det_exc = decile_detection(res["A_exc"].astype(bool) & off)
    det_inh = decile_detection(res["A_inh"].astype(bool) & off)

    # --- degree recovery ----------------------------------------------------
    r_out = float(np.corrcoef(true.sum(1), pred.sum(1))[0, 1])
    r_in = float(np.corrcoef(true.sum(0), pred.sum(0))[0, 1])

    # --- FP anatomy ---------------------------------------------------------
    pos = np.asarray(net["neuron_positions"], float)
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    fp = pred & ~true & off
    tp = pred & true
    # shared common input: number of common true presynaptic parents per pair
    Ai = A.astype(np.int32)
    shared = Ai.T @ Ai                     # shared[i, j] = # common parents of i and j
    stats = dict(
        session=session, n_neurons=int(n),
        detection_by_weight_decile_exc=det_exc,
        detection_by_weight_decile_inh=det_inh,
        degree_r_out=r_out, degree_r_in=r_in,
        fp_median_distance=float(np.median(D[fp])) if fp.any() else None,
        tp_median_distance=float(np.median(D[tp])) if tp.any() else None,
        fp_mean_shared_input=float(shared[fp].mean()) if fp.any() else None,
        tp_mean_shared_input=float(shared[tp].mean()) if tp.any() else None,
        n_fp=int(fp.sum()), n_tp=int(tp.sum()),
    )
    with open(os.path.join(glm_dir, "edge_anatomy.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    print("%s: exc det(low->high w) %.2f->%.2f | inh %.2f->%.2f | degree r_out %.2f "
          "r_in %.2f | shared-input FP %.1f vs TP %.1f" % (
              session, det_exc[0], det_exc[-1], det_inh[0], det_inh[-1], r_out, r_in,
              stats["fp_mean_shared_input"] or -1, stats["tp_mean_shared_input"] or -1))
    return stats


def figures(all_stats):
    meta = {}
    for s in all_stats:
        sess = s["session"]
        with open(os.path.join(resolve(sess, STATE), "session_metadata.json"),
                  encoding="utf-8") as fh:
            m = json.load(fh)
        bursts = np.mean([r.get("n_bursts", 0) for r in m["recordings"]])
        meta[sess] = dict(bursts=float(bursts), group="c50" if "_c50_" in sess else "c40")
    col = lambda s: C50 if meta[s["session"]]["group"] == "c50" else C40

    fig, (axe, axi) = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for s in all_stats:
        axe.plot(range(1, 11), s["detection_by_weight_decile_exc"], "-", color=col(s),
                 alpha=0.55, lw=1.4)
        axi.plot(range(1, 11), s["detection_by_weight_decile_inh"], "-", color=col(s),
                 alpha=0.55, lw=1.4)
    axe.set_title("Excitatory edges: the GLM finds strong synapses first")
    axi.set_title("Inhibitory edges: weaker, flatter detection")
    for ax in (axe, axi):
        ax.set_xlabel("true |weight| decile within class (weak -> strong)")
        ax.grid(alpha=0.3)
    axe.set_ylabel("detection probability (typed operating point)")
    fig.suptitle("%d networks; blue c50, red c40" % len(all_stats), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "weight_vs_detection.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for s in all_stats:
        b = meta[s["session"]]["bursts"]
        ax.plot(b, s["degree_r_out"], "o", color=col(s), ms=7)
        ax.plot(b, s["degree_r_in"], "^", color=col(s), ms=7, alpha=0.6)
    ax.plot([], [], "ko", label="out-degree r")
    ax.plot([], [], "k^", alpha=0.6, label="in-degree r")
    ax.set_xlabel("bursts per recording"); ax.set_ylabel("true-vs-predicted degree correlation")
    ax.set_title("Hub structure survives inference except in heavy burst regimes")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "degree_recovery.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for s in all_stats:
        if s["fp_mean_shared_input"] is None:
            continue
        ax.plot(meta[s["session"]]["bursts"],
                s["fp_mean_shared_input"] / max(s["tp_mean_shared_input"], 1e-9),
                "o", color=col(s), ms=8)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel("bursts per recording")
    ax.set_ylabel("mean shared common input: FP pairs / TP pairs")
    ax.set_title("False positives are common-input pairs (ratio > 1 = FPs share more parents)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fp_anatomy.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    sessions = sys.argv[1:] or sorted(s for s in list_sessions() if s.startswith("sweep_"))
    stats = [analyze(s) for s in sessions]
    if len(stats) > 1:
        figures(stats)
        print("figures -> %s" % OUT)


if __name__ == "__main__":
    main()
