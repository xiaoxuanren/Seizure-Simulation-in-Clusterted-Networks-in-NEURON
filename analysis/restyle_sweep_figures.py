"""Restyle four sweep figures to the thesis style, rebuilt ONLY from the
persisted numbers (per-session JSONs + sweep_summary.csv + durgrid JSONs) --
no fit npz, no recordings. Each figure writes a sibling CSV so its claim
survives as numbers.

  splithalf_reliability.png  <- results/normal/glm/splithalf.json x20
  minutes_to_criterion.png   <- results/normal/glm/scaling.json  x20
  fp_anatomy.png             <- results/normal/glm/edge_anatomy.json x20
  durgrid_all_oracle.png     <- sweep_summary/durgrid/*.json (layer-labeled)

    python analysis/restyle_sweep_figures.py
"""
import csv
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
from session_paths import DATA, results_dir  # noqa: E402
import thesis_style as st  # noqa: E402

st.apply()
OUT = os.path.join(DATA, "sweep_summary")


def sessions():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(
        os.path.join(DATA, "sweep_*", "results")))


def sload(session, name):
    p = os.path.join(results_dir(session, "normal", "glm"), name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def bursts_per_rec():
    rows = csv.DictReader(open(os.path.join(OUT, "sweep_summary.csv"),
                               encoding="utf-8"))
    return {r["session"]: float(r["bursts_per_rec"]) for r in rows}


def col(session):
    return st.C50 if "_c50_" in session else st.C40


def newfig(w=3.2, h=2.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.92, bottom=0.17)
    st.lineax(ax)
    return fig, ax


def write_csv(name, cols, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(p)


def fig_splithalf():
    b = bursts_per_rec()
    fig, ax = newfig(3.4, 2.6)
    rows = []
    for s in sessions():
        d = sload(s, "splithalf.json")
        if not d:
            continue
        ax.plot(b[s], d["jaccard_topk"], "o", ms=4.5, color=col(s))
        ax.plot(b[s], d["spearman_topk"], "^", ms=4, color=col(s), alpha=0.55)
        rows.append(dict(session=s, group="c50" if "_c50_" in s else "c40",
                         bursts_per_rec=b[s], jaccard_topk=d["jaccard_topk"],
                         spearman_topk=d["spearman_topk"],
                         k_true_edges=d["k_true_edges"]))
    ax.plot([], [], "o", ms=4.5, color="0.2", label="top-K Jaccard")
    ax.plot([], [], "^", ms=4, color="0.2", alpha=0.55,
            label=u"top-K Spearman ρ")
    ax.set_xlabel("bursts per recording")
    ax.set_ylabel("split-half reliability (top-K of |W|)")
    ax.legend(loc="lower left")
    p = os.path.join(OUT, "splithalf_reliability.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)
    write_csv("splithalf_reliability.csv",
              ["session", "group", "bursts_per_rec", "jaccard_topk",
               "spearman_topk", "k_true_edges"], rows)


def fig_minutes():
    b = bursts_per_rec()
    fig, ax = newfig(3.4, 2.6)
    rows = []
    for s in sessions():
        d = sload(s, "scaling.json")
        if not d:
            continue
        xs = np.array([g["n_recordings"] for g in d["grid"]], float)
        ys = np.array([g["auc"] for g in d["grid"]], float)
        above = np.nonzero(ys >= 0.9)[0]
        if len(above):
            k = above[0]
            need = xs[k] if k == 0 else float(
                np.interp(0.9, [ys[k - 1], ys[k]], [xs[k - 1], xs[k]]))
            ax.plot(b[s], need, "o" if "_c50_" in s else "s", ms=4.5,
                    color=col(s))
        else:
            need = float("nan")
            ax.plot(b[s], xs.max(), "x", color=col(s), ms=6, mew=1.4)
        rows.append(dict(session=s, group="c50" if "_c50_" in s else "c40",
                         bursts_per_rec=b[s],
                         minutes_to_auc09=round(need, 2)))
    ax.plot([], [], "o", ms=4.5, color=st.C50, label="c50")
    ax.plot([], [], "s", ms=4.5, color=st.C40, label="c40")
    ax.set_xlabel("bursts per recording")
    ax.set_ylabel("minutes of data to reach AUC 0.9")
    ax.legend(loc="upper left")
    p = os.path.join(OUT, "minutes_to_criterion.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)
    write_csv("minutes_to_criterion.csv",
              ["session", "group", "bursts_per_rec", "minutes_to_auc09"],
              rows)


def fig_fp_anatomy():
    fig, axes = plt.subplots(1, 2, figsize=(st.FIGW * 0.62, 2.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.18,
                        wspace=0.35)
    rows = []
    for s in sessions():
        d = sload(s, "edge_anatomy.json")
        if not d:
            continue
        axes[0].plot(d["tp_median_distance"], d["fp_median_distance"], "o",
                     ms=4.5, color=col(s))
        axes[1].plot(d["tp_mean_shared_input"], d["fp_mean_shared_input"],
                     "o", ms=4.5, color=col(s))
        rows.append(dict(session=s, group="c50" if "_c50_" in s else "c40",
                         tp_median_distance=d["tp_median_distance"],
                         fp_median_distance=d["fp_median_distance"],
                         tp_mean_shared_input=d["tp_mean_shared_input"],
                         fp_mean_shared_input=d["fp_mean_shared_input"],
                         n_tp=d["n_tp"], n_fp=d["n_fp"]))
    for ax, xl, yl in ((axes[0], "TP median pair distance",
                        "FP median pair distance"),
                       (axes[1], "TP mean shared inputs",
                        "FP mean shared inputs")):
        lim = ax.get_xlim() + ax.get_ylim()
        lo, hi = min(lim), max(lim)
        ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        st.lineax(ax)
    st.letter(axes[0], "a", dx=-0.24)
    st.letter(axes[1], "b", dx=-0.24)
    p = os.path.join(OUT, "fp_anatomy.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)
    write_csv("fp_anatomy.csv",
              ["session", "group", "tp_median_distance", "fp_median_distance",
               "tp_mean_shared_input", "fp_mean_shared_input", "n_tp", "n_fp"],
              rows)


def fig_oracle():
    """Achieved (typed protocol, all edges) vs oracle ceiling; the ceiling is
    computed on the EXCITATORY layer -- the axes say so explicitly."""
    grid = {}
    for p in sorted(glob.glob(os.path.join(OUT, "durgrid", "durgrid_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        grid.setdefault(d["session"], {})[d["n_recordings"]] = d
    fig, ax = newfig(3.5, 2.9)
    fig.subplots_adjust(left=0.22)
    rows = []
    for s, pts in sorted(grid.items()):
        d = pts[max(pts)]
        orc_exc = d["layers"]["exc"]["oracle"]["best_f1"]
        orc_all = d["layers"]["all_absW"]["oracle"]["best_f1"]
        ach = d["typed_at_070"]["f1"]
        ax.plot(orc_exc, ach, "o" if "_c50_" in s else "s", ms=4.5,
                color=col(s), alpha=0.85)
        rows.append(dict(session=s, group="c50" if "_c50_" in s else "c40",
                         n_recordings=max(pts),
                         oracle_best_f1_exc=round(orc_exc, 4),
                         oracle_best_f1_all_absW=round(orc_all, 4),
                         achieved_f1_typed=round(ach, 4),
                         gap_vs_exc_oracle=round(orc_exc - ach, 4)))
    lim = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lim), max(lim)
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=0.7)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([], [], "o", ms=4.5, color=st.C50, label="c50")
    ax.plot([], [], "s", ms=4.5, color=st.C40, label="c40")
    ax.set_xlabel("oracle best F1 (excitatory layer,\nground-truth threshold)")
    ax.set_ylabel("achieved F1 (typed protocol, all edges,\n"
                  "label-free FDR 0.70)")
    ax.legend(loc="upper left")
    p = os.path.join(OUT, "durgrid_all_oracle.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)
    write_csv("durgrid_all_oracle.csv",
              ["session", "group", "n_recordings", "oracle_best_f1_exc",
               "oracle_best_f1_all_absW", "achieved_f1_typed",
               "gap_vs_exc_oracle"], rows)
    gaps = [r["gap_vs_exc_oracle"] for r in rows]
    print("mean achieved-vs-EXC-oracle gap: %.3f F1" % np.mean(gaps))


if __name__ == "__main__":
    fig_splithalf()
    fig_minutes()
    fig_fp_anatomy()
    fig_oracle()
