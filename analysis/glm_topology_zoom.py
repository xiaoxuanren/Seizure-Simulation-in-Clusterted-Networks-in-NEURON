"""Zoomed predicted-vs-true wiring: individual neurons and edges made visible.

The whole-network figure draws 13,356 edges over 926 neurons, so the neurons
disappear under the lines. These zooms take one cluster at a time:

    left   spatial view of that cluster's neurons -- each one visible, excitatory
           filled grey, inhibitory ringed -- with every edge that touches the
           cluster drawn as TP / FP / FN
    right  the adjacency submatrix for the same neurons, one cell per pair, with
           the true edge outlined so hits and misses are countable by eye

One PNG per cluster, plus a combined figure covering several clusters at once.
Scores in each title are computed over the WITHIN-cluster pairs shown.

    python glm_topology_zoom.py                    # 3 largest clusters
    python glm_topology_zoom.py --clusters 4 2 35
"""

import argparse
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
from matplotlib.patches import Rectangle  # noqa: E402

_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SD = resolve(_S, _T)
RESULTS = results_dir(_S, _T, "glm")
FIGS = results_dir(_S, _T, "figures")

GREEN, RED, ORANGE = "#1a9850", "#d73027", "#f0a000"


def load():
    res = np.load(os.path.join(RESULTS, "glm_connectivity_sum4_5ms.npz"),
                  allow_pickle=True)
    net = np.load(sorted(glob.glob(os.path.join(SD, "network_*.npz")))[0],
                  allow_pickle=True)
    pred = res["pred_adjacency"].astype(bool)
    cand = res["candidates"].astype(bool)
    A = (res["A_exc"].astype(bool) | res["A_inh"].astype(bool))
    N = pred.shape[0]
    off = ~np.eye(N, dtype=bool)
    return dict(pred=pred, cand=cand, A=A & off, off=off, N=N,
                pos=np.asarray(net["neuron_positions"], float),
                ca=np.asarray(net["cluster_assignments"]).astype(int),
                inh=np.asarray(net["neuron_is_inhibitory"]).astype(bool),
                n_rec=int(res["n_recordings"]) if "n_recordings" in res.files else -1)


def panels(d, ids, ax_sp, ax_mx, title_prefix, show_ids=True):
    """Draw the spatial zoom and the adjacency submatrix for neurons ``ids``."""
    pos, inh = d["pos"], d["inh"]
    sub = np.ix_(ids, ids)
    A_s, P_s, C_s = d["A"][sub], d["pred"][sub], d["cand"][sub]
    tp, fp, fn = A_s & P_s, P_s & ~A_s & C_s, A_s & ~P_s
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    Pr = TP / (TP + FP) if TP + FP else 0.0
    Rc = TP / (TP + FN) if TP + FN else 0.0

    # --- spatial ---
    p = pos[ids]

    def segs(m):
        return [[p[i], p[j]] for i, j in np.argwhere(m)]

    ax_sp.add_collection(LineCollection(segs(fn), colors=ORANGE, lw=1.0,
                                        alpha=0.75, linestyles="dotted",
                                        zorder=1))
    ax_sp.add_collection(LineCollection(segs(tp), colors=GREEN, lw=1.0,
                                        alpha=0.6, zorder=2))
    ax_sp.add_collection(LineCollection(segs(fp), colors=RED, lw=1.0,
                                        alpha=0.7, zorder=3))
    e = ~inh[ids]
    ax_sp.scatter(p[e, 0], p[e, 1], s=90, c="#d9d9d9", edgecolors="#444",
                  lw=0.8, zorder=5)
    ax_sp.scatter(p[~e, 0], p[~e, 1], s=130, facecolors="white",
                  edgecolors="k", lw=2.0, zorder=6)
    if show_ids:
        for k, gid in enumerate(ids):
            ax_sp.annotate(str(gid), p[k], fontsize=5.5, ha="center",
                           va="center", zorder=7)
    ax_sp.set_title("%s - %d neurons (%d inhibitory)\nwithin-cluster: "
                    "P=%.2f R=%.2f  (TP %d, FP %d, FN %d)"
                    % (title_prefix, len(ids), int(inh[ids].sum()), Pr, Rc,
                       TP, FP, FN), fontsize=10)
    ax_sp.set_xticks([])
    ax_sp.set_yticks([])
    ax_sp.autoscale_view()
    ax_sp.margins(0.08)

    # --- adjacency submatrix ---
    M = np.zeros(A_s.shape)
    M[fn] = 1
    M[fp] = 2
    M[tp] = 3
    ax_mx.imshow(M, cmap=ListedColormap(["white", ORANGE, RED, GREEN]),
                 vmin=0, vmax=3, interpolation="nearest", aspect="equal")
    # outline every TRUE edge so misses are countable against the truth
    for i, j in np.argwhere(A_s):
        ax_mx.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                  edgecolor="#333", lw=0.35, zorder=4))
    n = len(ids)
    if n <= 45:
        ax_mx.set_xticks(range(n))
        ax_mx.set_yticks(range(n))
        ax_mx.set_xticklabels([str(g) for g in ids], fontsize=4.5, rotation=90)
        ax_mx.set_yticklabels([str(g) for g in ids], fontsize=4.5)
    ax_mx.set_xlabel("post")
    ax_mx.set_ylabel("pre")
    ax_mx.set_title("adjacency submatrix (black outline = a true edge)",
                    fontsize=10)
    return dict(n=len(ids), TP=TP, FP=FP, FN=FN, precision=Pr, recall=Rc)


def group_panels(d, clusters, ax_sp, ax_mx):
    """Several clusters at once, so BETWEEN-cluster edges are visible too."""
    pos, inh, ca = d["pos"], d["inh"], d["ca"]
    ids = np.concatenate([np.where(ca == c)[0] for c in clusters])
    owner = np.concatenate([np.full((ca == c).sum(), i)
                            for i, c in enumerate(clusters)])
    sub = np.ix_(ids, ids)
    A_s, P_s, C_s = d["A"][sub], d["pred"][sub], d["cand"][sub]
    tp, fp, fn = A_s & P_s, P_s & ~A_s & C_s, A_s & ~P_s

    same = owner[:, None] == owner[None, :]
    def stat(m):
        w, b = m & same, m & ~same
        return int(m.sum()), int(w.sum()), int(b.sum())
    (TP, TPw, TPb), (FP, _, _), (FN, FNw, FNb) = stat(tp), stat(fp), stat(fn)
    Pr = TP / (TP + FP) if TP + FP else 0.0
    Rc = TP / (TP + FN) if TP + FN else 0.0
    Rw = TPw / (TPw + FNw) if TPw + FNw else 0.0
    Rb = TPb / (TPb + FNb) if TPb + FNb else 0.0

    p = pos[ids]
    def segs(m):
        return [[p[i], p[j]] for i, j in np.argwhere(m)]

    ax_sp.add_collection(LineCollection(segs(fn), colors=ORANGE, lw=0.5,
                                        alpha=0.5, linestyles="dotted", zorder=1))
    ax_sp.add_collection(LineCollection(segs(tp), colors=GREEN, lw=0.5,
                                        alpha=0.45, zorder=2))
    ax_sp.add_collection(LineCollection(segs(fp), colors=RED, lw=0.6,
                                        alpha=0.6, zorder=3))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, c in enumerate(clusters):
        m = owner == i
        e = m & ~inh[ids]
        h = m & inh[ids]
        ax_sp.scatter(p[e, 0], p[e, 1], s=55, color=cmap[i % 10],
                      edgecolors="#333", lw=0.6, zorder=5,
                      label="cluster %d (n=%d)" % (c, m.sum()))
        ax_sp.scatter(p[h, 0], p[h, 1], s=110, facecolors="white",
                      edgecolors=cmap[i % 10], lw=2.4, zorder=6)
    ax_sp.set_title("%d clusters, %d neurons  |  P=%.2f R=%.2f\n"
                    "recall within-cluster %.2f vs between-cluster %.2f"
                    % (len(clusters), len(ids), Pr, Rc, Rw, Rb), fontsize=10)
    ax_sp.set_xticks([])
    ax_sp.set_yticks([])
    ax_sp.autoscale_view()
    ax_sp.margins(0.08)

    M = np.zeros(A_s.shape)
    M[fn] = 1
    M[fp] = 2
    M[tp] = 3
    ax_mx.imshow(M, cmap=ListedColormap(["white", ORANGE, RED, GREEN]),
                 vmin=0, vmax=3, interpolation="nearest", aspect="equal")
    edges = np.cumsum([0] + [int((owner == i).sum()) for i in range(len(clusters))])
    for b in edges[1:-1]:                      # cluster block separators
        ax_mx.axhline(b - .5, color="#222", lw=1.1)
        ax_mx.axvline(b - .5, color="#222", lw=1.1)
    mids = 0.5 * (edges[:-1] + edges[1:]) - .5
    ax_mx.set_xticks(mids)
    ax_mx.set_yticks(mids)
    ax_mx.set_xticklabels(["c%d" % c for c in clusters], fontsize=9)
    ax_mx.set_yticklabels(["c%d" % c for c in clusters], fontsize=9)
    ax_mx.set_xlabel("post")
    ax_mx.set_ylabel("pre")
    ax_mx.set_title("adjacency, blocks = clusters\n"
                    "diagonal blocks within-cluster, off-diagonal between",
                    fontsize=10)
    return dict(clusters=list(map(int, clusters)), n=len(ids), TP=TP, FP=FP,
                FN=FN, precision=Pr, recall=Rc, recall_within=Rw,
                recall_between=Rb)


def nearest_groups(d, size, n_groups):
    """Spatially adjacent cluster groups, so a zoom is geographically coherent."""
    ca, pos = d["ca"], d["pos"]
    ids = np.unique(ca)
    cent = np.array([pos[ca == c].mean(0) for c in ids])
    counts = np.array([(ca == c).sum() for c in ids])
    seeds = ids[np.argsort(counts)[::-1]]          # start from big clusters
    out, used = [], set()
    for s in seeds:
        if len(out) >= n_groups:
            break
        if s in used:
            continue
        k = np.where(ids == s)[0][0]
        order = np.argsort(((cent - cent[k]) ** 2).sum(1))
        grp = [int(ids[j]) for j in order if int(ids[j]) not in used][:size]
        if len(grp) < size:
            continue
        used.update(grp)
        out.append(grp)
    return out


def legend_handles():
    return [
        Line2D([0], [0], color=GREEN, lw=2, label="TP (recovered)"),
        Line2D([0], [0], color=RED, lw=2, label="FP (false)"),
        Line2D([0], [0], color=ORANGE, lw=2, ls=":", label="FN (missed)"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="#d9d9d9",
               markeredgecolor="#444", label="excitatory"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor="white",
               markeredgecolor="k", markeredgewidth=2, label="inhibitory"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=int, nargs="+", default=None,
                    help="cluster ids to zoom (default: the 3 largest)")
    ap.add_argument("--group-size", type=int, default=4,
                    help="clusters per group figure (0 disables group figures)")
    ap.add_argument("--n-groups", type=int, default=3)
    a = ap.parse_args()

    d = load()
    counts = np.bincount(d["ca"])
    clusters = a.clusters or [int(c) for c in np.argsort(counts)[::-1][:3]]

    # --- groups of neighbouring clusters: shows BETWEEN-cluster edges too ---
    if a.group_size > 1:
        for gi, grp in enumerate(nearest_groups(d, a.group_size, a.n_groups)):
            fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
            s = group_panels(d, grp, ax[0], ax[1])
            h = ax[0].get_legend_handles_labels()[0] + legend_handles()[:3]
            ax[0].legend(handles=h, fontsize=8, loc="upper right", framealpha=0.9)
            fig.suptitle("%s / %s, %d recordings - clusters %s"
                         % (_S, _T, d["n_rec"],
                            ", ".join(str(c) for c in grp)),
                         fontsize=13, weight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(FIGS, "glm_topology_zoom_group%d.png" % (gi + 1))
            fig.savefig(out, dpi=145, facecolor="white", bbox_inches="tight")
            plt.close(fig)
            print("  group %d %-18s n=%-4d P=%.3f R=%.3f (within %.3f / between "
                  "%.3f) -> %s"
                  % (gi + 1, str(grp), s["n"], s["precision"], s["recall"],
                     s["recall_within"], s["recall_between"],
                     os.path.basename(out)))

    # --- one figure per cluster ---
    for c in clusters:
        ids = np.where(d["ca"] == c)[0]
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        s = panels(d, ids, ax[0], ax[1], "cluster %d" % c)
        ax[0].legend(handles=legend_handles(), fontsize=8, loc="upper right",
                     framealpha=0.9)
        fig.suptitle("%s / %s, %d recordings - cluster %d zoom"
                     % (_S, _T, d["n_rec"], c), fontsize=13, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(FIGS, "glm_topology_zoom_cluster%02d.png" % c)
        fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("  cluster %-3d n=%-3d P=%.3f R=%.3f (TP %d FP %d FN %d) -> %s"
              % (c, s["n"], s["precision"], s["recall"], s["TP"], s["FP"],
                 s["FN"], os.path.basename(out)))

    # --- combined: several clusters side by side ---
    fig, ax = plt.subplots(len(clusters), 2, figsize=(15, 6.5 * len(clusters)))
    ax = np.atleast_2d(ax)
    for r, c in enumerate(clusters):
        ids = np.where(d["ca"] == c)[0]
        panels(d, ids, ax[r, 0], ax[r, 1], "cluster %d" % c, show_ids=False)
    ax[0, 0].legend(handles=legend_handles(), fontsize=8, loc="upper right",
                    framealpha=0.9)
    fig.suptitle("%s / %s, %d recordings - predicted vs true wiring, "
                 "per-cluster zoom" % (_S, _T, d["n_rec"]),
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGS, "glm_topology_zoom_combined.png")
    fig.savefig(out, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  combined -> %s" % os.path.basename(out))


if __name__ == "__main__":
    main()
