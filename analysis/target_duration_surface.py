"""Why the nominal FDR target is 0.70: the target x duration surface.

Input: sweep_summary/durgrid/durgrid_<session>_n<N>.json (260 files from the
CHTC duration grid), all_absW layer -- the layer the deployed operating point
thresholds. No recordings needed.

Panels: (a) realized FDR vs nominal target, one curve per duration, mean over
the 20 networks; (b) F1 vs nominal target at representative durations with
the maximizing target marked; (c) heatmap of mean realized FDR over
target x duration with contours at 0.05 / 0.10 / 0.20.

Writes sweep_summary/target_duration_surface.{png,json}. The JSON carries the
full mean surface plus, per duration, the F1-maximizing target and the
realized FDR there -- so the figure's claim survives as numbers.

    python analysis/target_duration_surface.py
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
from session_paths import DATA  # noqa: E402
import thesis_style as st  # noqa: E402

st.apply()

GRID = os.path.join(DATA, "sweep_summary", "durgrid")
OUT = os.path.join(DATA, "sweep_summary")
LAYER = "all_absW"
REP_DURATIONS = [10, 30, 60, 120, 200]     # panel (b)

# sanity anchors (recomputed 2026-08-31 from the durgrid JSONs themselves)
ANCHOR_AUC = {10: 0.836, 50: 0.944, 100: 0.967, 200: 0.980}
ANCHOR_P70 = {10: 0.698, 50: 0.780, 100: 0.794, 200: 0.794}


def load():
    grid = {}
    for p in sorted(glob.glob(os.path.join(GRID, "durgrid_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        grid.setdefault(d["session"], {})[d["n_recordings"]] = d
    return grid


def main():
    grid = load()
    sessions = sorted(grid)
    durations = sorted(next(iter(grid.values())))
    targets = [r["target"]
               for r in grid[sessions[0]][durations[0]]["layers"][LAYER]["targets"]]
    nS, nD, nT = len(sessions), len(durations), len(targets)
    print("%d sessions x %d durations x %d targets" % (nS, nD, nT))

    realized = np.full((nS, nD, nT), np.nan)
    f1 = np.full((nS, nD, nT), np.nan)
    prec = np.full((nS, nD, nT), np.nan)
    rec = np.full((nS, nD, nT), np.nan)
    empty_cells, violations = [], []
    for i, s in enumerate(sessions):
        for j, n in enumerate(durations):
            for k, r in enumerate(grid[s][n]["layers"][LAYER]["targets"]):
                if r["n_pred"] == 0:
                    empty_cells.append([s, n, r["target"]])
                    continue
                realized[i, j, k] = r["realized_fdr"]
                f1[i, j, k] = r["f1"]
                prec[i, j, k] = r["precision"]
                rec[i, j, k] = r["recall"]
                if r["realized_fdr"] > r["target"] + 1e-12:
                    violations.append([s, n, r["target"],
                                       round(r["realized_fdr"], 4)])

    # ---- sanity, before plotting ----------------------------------------
    for n, want in ANCHOR_AUC.items():
        got = np.mean([grid[s][n]["layers"][LAYER]["auc"] for s in sessions])
        assert abs(got - want) < 5e-4, "AUC anchor n=%d: %.4f != %.3f" % (n, got, want)
    k70 = targets.index(0.7)
    for n, want in ANCHOR_P70.items():
        got = np.nanmean(prec[:, durations.index(n), k70])
        assert abs(got - want) < 5e-4, "P@.70 anchor n=%d: %.4f != %.3f" % (n, got, want)
    print("sanity anchors: MATCH  (%d realized>nominal cells, %d empty)"
          % (len(violations), len(empty_cells)))

    m_real = np.nanmean(realized, 0)
    m_f1 = np.nanmean(f1, 0)

    fig, axes = plt.subplots(1, 3, figsize=(st.FIGW, 2.35),
                             gridspec_kw=dict(wspace=0.48, left=0.075,
                                              right=0.965, top=0.90,
                                              bottom=0.19))
    cmap = plt.cm.viridis

    axa = axes[0]
    for j, n in enumerate(durations):
        axa.plot(targets, m_real[j], "-", color=cmap(j / (nD - 1)), lw=0.9)
    axa.plot([0, 1], [0, 1], "--", color="0.5", lw=0.7)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(min(durations), max(durations)))
    cb = fig.colorbar(sm, ax=axa, pad=0.02, fraction=0.055)
    cb.ax.set_title("recs", fontsize=6, pad=3)
    cb.ax.tick_params(labelsize=6)
    axa.set_xlabel("nominal FDR target")
    axa.set_ylabel("realized FDR (all-|W| layer)")
    axa.set_xlim(0, 1)
    axa.set_ylim(0, 1)
    st.lineax(axa)
    st.letter(axa, "a")

    axb = axes[1]
    best = []
    for n in REP_DURATIONS:
        j = durations.index(n)
        c = cmap(j / (nD - 1))
        axb.plot(targets, m_f1[j], "-o", color=c, lw=0.9, ms=2.2,
                 label="n=%d" % n)
        kbest = int(np.nanargmax(m_f1[j]))
        best.append((n, targets[kbest]))
        axb.plot(targets[kbest], m_f1[j][kbest], "o", color=c, ms=5.5,
                 mfc="none", mew=1.2)
    axb.set_xlabel("nominal FDR target")
    axb.set_ylabel("F1 (all-|W| layer)")
    axb.legend(loc="lower center", ncol=2, handlelength=1.4)
    st.lineax(axb)
    st.letter(axb, "b")

    axc = axes[2]
    im = axc.imshow(m_real, aspect="auto", origin="lower", cmap="magma_r",
                    extent=[targets[0], targets[-1], 0, nD], vmin=0,
                    vmax=max(0.3, np.nanmax(m_real)))
    Tg, Dg = np.meshgrid(targets, np.arange(nD) + 0.5)
    cs = axc.contour(Tg, Dg, m_real, levels=[0.05, 0.10, 0.20],
                     colors="white", linewidths=0.7)
    axc.clabel(cs, fontsize=5.5, fmt="%.2f")
    axc.set_yticks(np.arange(nD) + 0.5)
    axc.set_yticklabels(durations, fontsize=5.5)
    axc.set_xlabel("nominal FDR target")
    axc.set_ylabel("recordings")
    cb2 = fig.colorbar(im, ax=axc, pad=0.02, fraction=0.055)
    cb2.set_label("mean realized FDR", fontsize=6.5)
    cb2.ax.tick_params(labelsize=6)
    st.letter(axc, "c")

    p = os.path.join(OUT, "target_duration_surface.png")
    fig.savefig(p, dpi=st.DPI, facecolor="white")
    plt.close(fig)
    print(p)

    per_duration = []
    for j, n in enumerate(durations):
        kbest = int(np.nanargmax(m_f1[j]))
        per_duration.append(dict(
            n_recordings=n, f1_max_target=targets[kbest],
            f1_at_max=float(m_f1[j][kbest]),
            realized_fdr_at_max=float(m_real[j][kbest]),
            f1_at_070=float(m_f1[j][k70]),
            realized_fdr_at_070=float(m_real[j][k70])))
    out = dict(
        layer=LAYER, sessions=sessions, durations=durations, targets=targets,
        mean_realized_fdr=np.round(m_real, 6).tolist(),
        mean_f1=np.round(m_f1, 6).tolist(),
        mean_precision=np.round(np.nanmean(prec, 0), 6).tolist(),
        mean_recall=np.round(np.nanmean(rec, 0), 6).tolist(),
        per_duration=per_duration,
        realized_gt_nominal_cells=violations,
        n_pred_zero_cells=empty_cells,
        n_cells=nS * nD * nT)
    q = os.path.join(OUT, "target_duration_surface.json")
    with open(q, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(q)
    for d in per_duration:
        if d["n_recordings"] in REP_DURATIONS:
            print("n=%3d  F1-max target %.2f (F1 %.3f, realized %.3f) | "
                  "at 0.70: F1 %.3f realized %.3f"
                  % (d["n_recordings"], d["f1_max_target"], d["f1_at_max"],
                     d["realized_fdr_at_max"], d["f1_at_070"],
                     d["realized_fdr_at_070"]))


if __name__ == "__main__":
    main()
