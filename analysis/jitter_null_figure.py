"""Thesis Figure 2 panels (a)-(c): the jitter-null threshold rule, regenerated
on the session the rest of the chapter uses.

Reuses the shipped pipeline verbatim -- sparse_glm.load_session / jitter /
fit_B, and sum4_W + fdr_threshold from analysis/burstexcl_glm_arm.py (the
same code path that produced results/<state>/glm/fdr_calibration.json, so the
worked numbers land on the recorded ones). Scores are |W| over all
off-diagonal pairs: the untyped layer of sweep_fdr_calibration.py.

Compute is cached in results/<state>/glm/jitter_null_bundle.npz; pass
--replot to redraw from the cache without refitting.

    python analysis/jitter_null_figure.py --session sweep_c50_seed09 --state normal
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sparse_glm as sg  # noqa: E402
from session_paths import DATA, resolve, results_dir  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
})

FIGW = 6.9
DPI = 400
CURVE_MAX_PTS = 2000          # thinning for the cached theta curve
WIN_MS = 2000.0               # panel (a) window
ZOOM_MS = 300.0               # panel (a) bottom strip
ZOOM_NEURONS = 7


def panel_letter(ax, letter, dx=-0.10, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- compute ---

def compute(session, state):
    # burstexcl_glm_arm resolves its session from these env vars at import
    # time; point it at ours BEFORE the import so its module-level code never
    # touches a session we are not using
    os.environ["DATASET_SESSION"] = session
    os.environ["DATASET_STATE"] = state
    import burstexcl_glm_arm as bx  # noqa: E402  (deliberate late import)

    sd = resolve(session, state)
    print("loading %s ..." % sd, flush=True)
    M, bnd = sg.load_session(sd, bin_ms=bx.BIN_MS)
    n = M.shape[0]
    off = ~np.eye(n, dtype=bool)
    gt = sg.load_ground_truth(sd)
    true_adj = (gt["A_exc"] | gt["A_inh"]) & off
    print("N=%d neurons, %d bins, %d recordings, %d true edges"
          % (n, M.shape[1], len(bnd) - 1, int(true_adj.sum())), flush=True)

    t0 = time.time()
    W = bx.sum4_W(M, bnd)
    obs = np.abs(W)[off]
    print("real fit: %.1f s" % (time.time() - t0), flush=True)

    rng = np.random.default_rng(bx.SEED)
    null = np.empty((bx.N_SURR, obs.size), np.float64)
    for k in range(bx.N_SURR):
        tk = time.time()
        null[k] = np.abs(bx.sum4_W(sg.jitter(M, bnd, bx.JBINS, rng), bnd))[off]
        print("surrogate %d/%d: %.1f s" % (k + 1, bx.N_SURR, time.time() - tk),
              flush=True)

    # -- threshold sweep over every unique observed score ------------------
    # estimated FDR uses >= (exactly fdr_threshold's convention); realized
    # uses strict > (exactly how predictions are thresholded downstream)
    cand = np.unique(obs)
    obs_s = np.sort(obs)
    null_s = np.sort(null.ravel())
    n_obs_ge = obs.size - np.searchsorted(obs_s, cand, side="left")
    n_null_ge = (null_s.size - np.searchsorted(null_s, cand, side="left")) \
        / bx.N_SURR
    est_curve = np.where(n_obs_ge > 0, n_null_ge / np.maximum(n_obs_ge, 1),
                         np.inf)
    order = np.argsort(obs)[::-1]
    ys = true_adj[off][order].astype(np.int64)
    cum_tp = np.cumsum(ys)
    n_pred_gt = obs.size - np.searchsorted(obs_s, cand, side="right")
    tp_gt = np.where(n_pred_gt > 0, cum_tp[np.maximum(n_pred_gt - 1, 0)], 0)
    real_curve = np.where(n_pred_gt > 0,
                          (n_pred_gt - tp_gt) / np.maximum(n_pred_gt, 1),
                          np.nan)

    thr, est = bx.fdr_threshold(obs, null, bx.TARGET_FDR)
    pred = np.abs(W) > thr
    np.fill_diagonal(pred, False)
    tp = int((pred & true_adj).sum())
    fp = int((pred & ~true_adj).sum())
    n_pred = int(pred.sum())
    realized = fp / max(n_pred, 1)
    i_thr = int(np.searchsorted(cand, thr))
    surr_mean_above = float(n_null_ge[min(i_thr, len(cand) - 1)])
    print("target %.2f -> threshold %.6f, %d edges, est FDR %.4f, "
          "realized %.4f" % (bx.TARGET_FDR, thr, n_pred, est, realized),
          flush=True)

    # -- thin the curve for the cache (always keep the selected theta) -----
    keep = np.unique(np.linspace(0, len(cand) - 1,
                                 min(CURVE_MAX_PTS, len(cand))).astype(int))
    keep = np.union1d(keep, [min(i_thr, len(cand) - 1)])

    # -- panel (a) spike window: a 2 s span around a saved burst window ----
    rec = sorted(p for p in glob.glob(os.path.join(sd, "recording*.npz"))
                 if "raster" not in os.path.basename(p))[0]
    d = np.load(rec, allow_pickle=True)
    bw = np.atleast_2d(np.asarray(d["burst_windows"], float))
    if bw.size:
        mid = 0.5 * (bw[0][0] + bw[0][1])
    else:                                   # no burst saved: use peak rate
        allt = np.concatenate([np.atleast_1d(np.asarray(t, float))
                               for t in d["spike_times"]])
        hist, edges = np.histogram(allt, bins=int(60000 / 100))
        mid = edges[np.argmax(hist)]
    win_t0 = float(np.clip(mid - WIN_MS / 2, 0,
                           float(d["duration"]) - WIN_MS))
    spk_neuron, spk_t = [], []
    for i, t in enumerate(d["spike_times"]):
        t = np.atleast_1d(np.asarray(t, float))
        m = (t >= win_t0) & (t < win_t0 + WIN_MS)
        spk_neuron.append(np.full(int(m.sum()), i, np.int32))
        spk_t.append(t[m])
    spk_neuron = np.concatenate(spk_neuron)
    spk_t = np.concatenate(spk_t).astype(np.float32)
    # the surrogate jitters each spike's BIN by a uniform integer in
    # [-JBINS, +JBINS]; mirror that operation on spike times for the drawing
    jrng = np.random.default_rng(bx.SEED)
    jit_t = (spk_t + bx.BIN_MS * jrng.integers(-bx.JBINS, bx.JBINS + 1,
                                               spk_t.size)).astype(np.float32)

    out = os.path.join(results_dir(session, state, "glm"),
                       "jitter_null_bundle.npz")
    np.savez_compressed(
        out,
        session=session, state=state, n_neurons=n,
        n_recordings=len(bnd) - 1,
        obs=obs.astype(np.float32),
        null=null.astype(np.float32),
        true_adj=true_adj,
        theta=cand[keep].astype(np.float32),
        est_fdr_curve=est_curve[keep].astype(np.float32),
        realized_fdr_curve=real_curve[keep].astype(np.float32),
        n_obs_ge=n_obs_ge[keep].astype(np.int64),
        target_fdr=bx.TARGET_FDR,
        threshold=thr, est_fdr=est, n_pred=n_pred, tp=tp, fp=fp,
        realized_fdr=realized, surr_mean_above=surr_mean_above,
        win_t0_ms=win_t0, win_ms=WIN_MS,
        spk_neuron=spk_neuron, spk_t_ms=spk_t, jit_t_ms=jit_t,
        jitter_ms=bx.JITTER_MS, bin_ms=bx.BIN_MS)
    print("bundle -> %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6),
          flush=True)
    return out


# ----------------------------------------------------------------- figure ---

def draw_a(fig, sub, b):
    """Three strips: original window, jittered window, joined zoom."""
    gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=sub,
                                          height_ratios=[1, 1, 1.25],
                                          hspace=0.35)
    t0, wms = float(b["win_t0_ms"]), float(b["win_ms"])
    spk_n, spk_t, jit_t = b["spk_neuron"], b["spk_t_ms"], b["jit_t_ms"]

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(spk_t - t0, spk_n, ".", color="#404040", ms=0.7, rasterized=True)
    ax1.set_title("original", fontsize=7, loc="left", pad=2)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(jit_t - t0, spk_n, ".", color="#c0392b", ms=0.7, rasterized=True)
    ax2.set_title(u"\u00b1%d ms jitter" % int(b["jitter_ms"]),
                  fontsize=7, loc="left", pad=2)
    for ax in (ax1, ax2):
        ax.set_xlim(0, wms)
        ax.set_yticks([])
        despine(ax)
        ax.set_xticks([])
    # time reference for the untick'd strips
    ymin, _ = ax1.get_ylim()
    ax1.plot([wms - 550, wms - 50], [ymin, ymin], "-", color="#202020", lw=1.2,
             clip_on=False)
    ax1.text(wms - 300, ymin, "500 ms", fontsize=6, ha="center", va="bottom")
    # zoom: densest 300 ms, ~7 most active neurons there
    hist, edges = np.histogram(spk_t - t0, bins=np.arange(0, wms + 1, 50.0))
    z0 = float(edges[np.argmax(hist)])
    z0 = min(max(z0 - ZOOM_MS / 3, 0), wms - ZOOM_MS)
    inz = (spk_t - t0 >= z0) & (spk_t - t0 < z0 + ZOOM_MS)
    ids, cnt = np.unique(spk_n[inz], return_counts=True)
    pick = ids[np.argsort(cnt)[::-1][:ZOOM_NEURONS]]
    ax3 = fig.add_subplot(gs[2])
    for row, nid in enumerate(sorted(pick)):
        m = inz & (spk_n == nid)
        for ts, tj in zip(spk_t[m] - t0, jit_t[m] - t0):
            ax3.plot([ts, tj], [row + 0.18, row + 0.82], "-",
                     color="#aaaaaa", lw=0.4, zorder=1)
        ax3.plot(spk_t[m] - t0, np.full(int(m.sum()), row + 0.18), ".",
                 color="#404040", ms=2.6, zorder=2)
        ax3.plot(jit_t[m] - t0, np.full(int(m.sum()), row + 0.82), ".",
                 color="#c0392b", ms=2.6, zorder=2)
    ax3.set_xlim(z0, z0 + ZOOM_MS)
    ax3.set_ylim(-0.3, len(pick))
    ax3.set_yticks([])
    ax3.set_xlabel("time in window (ms)")
    ax3.set_title("%d ms zoom" % int(ZOOM_MS), fontsize=7, loc="left", pad=2)
    despine(ax3)
    panel_letter(ax1, "a")
    return [ax1, ax2, ax3]


def draw_b(ax, b, compact=False):
    obs, null = b["obs"], b["null"]
    thr = float(b["threshold"])
    hi = max(obs.max(), null.max())
    bins = np.linspace(0.0, float(hi), 120)
    lab_null = ("jitter null" if compact
                else "jitter null (mean of %d)" % null.shape[0])
    ax.hist(null.ravel(), bins=bins, weights=np.full(null.size, 1.0 / null.shape[0]),
            color="#c9a0a0", label=lab_null)
    ax.hist(obs, bins=bins, histtype="step", color="#202020", lw=0.9,
            label="real fit")
    ax.axvline(thr, ls="--", color="#1f5fd0", lw=0.9,
               label=(u"\u03b8*" if compact
                      else u"\u03b8 = %.4f (FDR 0.70)" % thr))
    ax.set_yscale("log")
    ax.set_xlabel(u"edge score  |W|")
    ax.set_ylabel("pair count")
    if compact:
        ax.legend(frameon=False, loc="upper right", fontsize=5.6,
                  handlelength=1.1, labelspacing=0.25, borderaxespad=0.1)
    else:
        ax.legend(frameon=False, loc="upper right")
    despine(ax)
    panel_letter(ax, "b")


def draw_c(ax, b):
    th = b["theta"]
    # the count-ratio estimator exceeds 1 where the null has more mass above
    # theta than the data; clip for display, as q-values conventionally are
    ax.plot(th, np.minimum(b["est_fdr_curve"], 1.0), "-", color="#c0392b",
            lw=1.0, label="estimated (jitter null)")
    ax.plot(th, b["realized_fdr_curve"], "-", color="#404040",
            lw=1.0, label="true (ground truth)")
    tgt, thr = float(b["target_fdr"]), float(b["threshold"])
    ax.axhline(tgt, ls=":", color="#1f5fd0", lw=0.8)
    ax.axvline(thr, ls="--", color="#1f5fd0", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(u"threshold \u03b8")
    ax.set_ylabel("FDR")
    ax.legend(frameon=False, loc="upper right")
    box = (u"at \u03b8 = %.4f:\n"
           "edges above: %d\n"
           "null expectation: %.0f\n"
           "estimated FDR: %.3f\n"
           "true FDR: %.3f"
           % (thr, int(b["n_pred"]), float(b["surr_mean_above"]),
              float(b["est_fdr"]), float(b["realized_fdr"])))
    ax.text(0.03, 0.05, box, transform=ax.transAxes, fontsize=6.2,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#999999",
                      lw=0.5))
    despine(ax)
    panel_letter(ax, "c")


def render(bundle_path, session):
    b = dict(np.load(bundle_path, allow_pickle=True))
    outdir = os.path.join(DATA, "sweep_summary")

    fig = plt.figure(figsize=(FIGW, 2.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.42,
                           left=0.06, right=0.985, top=0.90, bottom=0.17)
    draw_a(fig, gs[0], b)
    draw_b(fig.add_subplot(gs[1]), b)
    draw_c(fig.add_subplot(gs[2]), b)
    p = os.path.join(outdir, "thesis_fig2_panels_abc_%s.png"
                     % session.replace("sweep_", ""))
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(p)

    # separate panels for the existing composite
    fa = plt.figure(figsize=(FIGW * 0.38, 2.6))
    ga = gridspec.GridSpec(1, 1, left=0.08, right=0.97, top=0.90, bottom=0.17)
    draw_a(fa, ga[0], b)
    for fig_i, drawer, tag, w in ((None, draw_b, "b", 0.33),
                                  (None, draw_c, "c", 0.33)):
        f = plt.figure(figsize=(FIGW * w, 2.6))
        ax = f.add_axes([0.17, 0.17, 0.78, 0.73])
        drawer(ax, b)
        q = os.path.join(outdir, "thesis_fig2_panel_%s_%s.png"
                         % (tag, session.replace("sweep_", "")))
        f.savefig(q, dpi=DPI, facecolor="white")
        plt.close(f)
        print(q)
    qa = os.path.join(outdir, "thesis_fig2_panel_a_%s.png"
                      % session.replace("sweep_", ""))
    fa.savefig(qa, dpi=DPI, facecolor="white")
    plt.close(fa)
    print(qa)


def _sessions():
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(
        os.path.join(DATA, "sweep_c*_seed*", "results")))


def _bundle(session, state="normal"):
    return os.path.join(results_dir(session, state, "glm"),
                        "jitter_null_bundle.npz")


def draw_d(ax):
    """Calibration across the study: realized vs nominal FDR, all 20 nets."""
    import thesis_style as ts
    for s in _sessions():
        p = os.path.join(results_dir(s, "normal", "glm"),
                         "fdr_calibration.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p, encoding="utf-8"))["targets"]
        ax.plot([r["target"] for r in rows],
                [r["realized_fdr"] for r in rows], "-o", ms=1.8, lw=0.8,
                color=ts.C50 if "_c50_" in s else ts.C40, alpha=0.65)
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8)
    ax.plot([], [], "-", color=st_c50(), label="50-cluster")
    ax.plot([], [], "-", color=st_c40(), label="40-cluster")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal FDR target")
    ax.set_ylabel("realized FDR")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax.grid(lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    despine(ax)
    panel_letter(ax, "d")


def st_c50():
    import thesis_style as ts
    return ts.C50


def st_c40():
    import thesis_style as ts
    return ts.C40


def composite(rep="sweep_c50_seed09"):
    """thesis_fig2_v3.png: (a)-(c) on the representative network, (d) sweep."""
    b = dict(np.load(_bundle(rep), allow_pickle=True))
    fig = plt.figure(figsize=(FIGW, 2.45))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.18, 1.0, 1.0, 1.0],
                           wspace=0.50, left=0.045, right=0.99,
                           top=0.89, bottom=0.18)
    draw_a(fig, gs[0], b)
    draw_b(fig.add_subplot(gs[1]), b, compact=True)
    draw_c(fig.add_subplot(gs[2]), b)
    draw_d(fig.add_subplot(gs[3]))
    p = os.path.join(DATA, "sweep_summary", "thesis_fig2_v3.png")
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(p)


def across_networks():
    """Panel-(b) tail comparison for all 20 networks: survival fraction of
    pair scores vs theta normalized by each network's selected threshold."""
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.94, bottom=0.15)
    for s in _sessions():
        b = np.load(_bundle(s), allow_pickle=True)
        obs, null = b["obs"], b["null"]
        thr = float(b["threshold"])
        xs = np.logspace(-1.3, 1.0, 140) * thr
        obs_s = np.sort(obs)
        null_s = np.sort(null.ravel())
        fo = (obs.size - np.searchsorted(obs_s, xs)) / obs.size
        fn_ = (null.size - np.searchsorted(null_s, xs)) / null.size
        col = st_c50() if "_c50_" in s else st_c40()
        ax.plot(xs / thr, np.maximum(fo, 1e-8), "-", color=col, lw=0.8,
                alpha=0.65)
        ax.plot(xs / thr, np.maximum(fn_, 1e-8), "-", color="0.65", lw=0.6,
                alpha=0.5)
    ax.axvline(1.0, ls="--", color="0.4", lw=0.8)
    ax.plot([], [], "-", color=st_c50(), label="real fit, 50-cluster")
    ax.plot([], [], "-", color=st_c40(), label="real fit, 40-cluster")
    ax.plot([], [], "-", color="0.65", label="jitter null (all networks)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 1.2)
    ax.set_xlabel(u"θ / θ*  (selected threshold)")
    ax.set_ylabel(u"fraction of pairs with score > θ")
    ax.legend(frameon=False, loc="lower left", fontsize=6.5)
    ax.grid(lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    despine(ax)
    p = os.path.join(DATA, "sweep_summary", "jitter_null_across_networks.png")
    fig.savefig(p, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(p)


def table():
    print("\n%-18s %10s %8s %10s %8s %8s %8s"
          % ("session", "threshold", "edges", "null_exp", "est", "true",
             "ratio"))
    ratios = []
    for s in _sessions():
        b = np.load(_bundle(s), allow_pickle=True)
        ratio = float(b["surr_mean_above"]) / max(int(b["fp"]), 1)
        ratios.append(ratio)
        print("%-18s %10.6f %8d %10.1f %8.4f %8.4f %7.1fx"
              % (s, float(b["threshold"]), int(b["n_pred"]),
                 float(b["surr_mean_above"]), float(b["est_fdr"]),
                 float(b["realized_fdr"]), ratio))
    ratios = np.array(ratios)
    print("conservatism ratio: mean %.1fx  median %.1fx  range %.1f-%.1fx"
          % (ratios.mean(), np.median(ratios), ratios.min(), ratios.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="sweep_c50_seed09")
    ap.add_argument("--state", default="normal")
    ap.add_argument("--replot", action="store_true",
                    help="redraw from the cached bundle without refitting")
    ap.add_argument("--composite", action="store_true",
                    help="build thesis_fig2_v3 + across-networks overlay + "
                         "per-network table from the existing bundles")
    a = ap.parse_args()
    if a.composite:
        composite(a.session)
        across_networks()
        table()
        return
    bundle = os.path.join(results_dir(a.session, a.state, "glm"),
                          "jitter_null_bundle.npz")
    if not (a.replot and os.path.exists(bundle)):
        bundle = compute(a.session, a.state)
    render(bundle, a.session)

    b = np.load(bundle, allow_pickle=True)
    print("\nworked numbers at target %.2f:" % float(b["target_fdr"]))
    print("  threshold        %.6f" % float(b["threshold"]))
    print("  edges above      %d" % int(b["n_pred"]))
    print("  null expectation %.1f" % float(b["surr_mean_above"]))
    print("  estimated FDR    %.4f" % float(b["est_fdr"]))
    print("  true FDR         %.4f" % float(b["realized_fdr"]))
    print("  null/trueFP      %.2f"
          % (float(b["surr_mean_above"]) / max(int(b["fp"]), 1)))

    cal = os.path.join(results_dir(a.session, a.state, "glm"),
                       "fdr_calibration.json")
    if os.path.exists(cal):
        rows = json.load(open(cal, encoding="utf-8"))["targets"]
        ref = [r for r in rows
               if abs(r["target"] - float(b["target_fdr"])) < 1e-9]
        if ref:
            r = ref[0]
            same = (abs(r["threshold"] - float(b["threshold"])) < 1e-12
                    and r["n_pred"] == int(b["n_pred"]))
            print("\nsanity vs fdr_calibration.json: %s"
                  % ("MATCH" if same else
                     "MISMATCH  (recorded thr %.6f n_pred %d est %.4f "
                     "realized %.4f)"
                     % (r["threshold"], r["n_pred"], r["est_fdr"],
                        r["realized_fdr"])))


if __name__ == "__main__":
    main()
