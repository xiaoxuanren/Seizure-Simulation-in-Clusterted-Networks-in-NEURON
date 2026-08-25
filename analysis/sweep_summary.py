"""Cross-network summary of the cluster-count x topology-seed sweep.

Aggregates every sweep_* session's saved artifacts -- session_metadata.json
(burst/rate stats, seeds), glm_lag_sweep.json (per-lag AUC/AP),
glm_connectivity_sum4_5ms.npz (typed-protocol edge prediction),
burstexcl_arms.json (FULL vs burst-excluded |W| arms) -- into

    notebooks/NEURON data parallel/sweep_summary/sweep_summary.csv

plus the cross-network figures:

    dose_response.png        precision & AUC vs burst rate, both groups
    exclusion_effect.png     FULL -> EXCL precision arrows for all networks
    size_vs_bursting.png     bursts/rec vs realized N (the critical-mass plot)
    typed_vs_untyped.png     typed-protocol precision vs |W|-protocol precision
                             (exposes the E/I-typing collapse in quiet nets)

    python analysis/sweep_summary.py
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
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from session_paths import DATA, resolve, results_dir, list_sessions  # noqa: E402

STATE = os.environ.get("DATASET_STATE", "normal")
OUT = os.path.join(DATA, "sweep_summary")

C50, C40 = "#1f5fd0", "#c0392b"


def typed_metrics(sd, glm_dir, n):
    """Typed-protocol P/R/F1 recomputed from the saved fit npz + ground truth."""
    path = os.path.join(glm_dir, "glm_connectivity_sum4_5ms.npz")
    if not os.path.exists(path):
        return {}
    res = np.load(path, allow_pickle=True)
    pred = res["pred_adjacency"].astype(bool)
    off = ~np.eye(n, dtype=bool)
    true = (res["A_exc"].astype(bool) | res["A_inh"].astype(bool)) & off
    tp = int((pred & true).sum()); fp = int((pred & ~true & off).sum())
    fn = int((~pred & true).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(typed_precision=P, typed_recall=R,
                typed_f1=2 * P * R / max(P + R, 1e-12))


def gather():
    rows = []
    for session in sorted(s for s in list_sessions() if s.startswith("sweep_")):
        sd = resolve(session, STATE)
        glm_dir = results_dir(session, STATE, "glm", create=False)
        with open(os.path.join(sd, "session_metadata.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        net = np.load(glob.glob(os.path.join(sd, "network_*.npz"))[0], allow_pickle=True)
        n = int(net["neuron_positions"].shape[0])
        spikes = np.array([r["num_spikes"] for r in meta["recordings"]])
        bursts = np.array([r.get("n_bursts", 0) for r in meta["recordings"]])
        row = dict(
            session=session,
            group="c%d" % int(net["num_clusters"]),
            topology_seed=int(net["topology_seed"]),
            noise_seed_base=int(net["noise_seed_base"]),
            n_neurons=n, n_edges=int(len(net["connections"])),
            density=float(net["density"]), space_size=float(net["space_size"]),
            mean_rate_hz=float((spikes / n / 60.0).mean()),
            bursts_per_rec=float(bursts.mean()),
            frac_recs_bursting=float((bursts > 0).mean()),
        )
        lag_path = os.path.join(sd, "glm_lag_sweep.json")
        if os.path.exists(lag_path):
            with open(lag_path, encoding="utf-8") as fh:
                sweep = json.load(fh)
            best = max(sweep["per_lag"], key=lambda r: r["exc_auc"])
            row.update(best_lag=best["lag"], exc_auc=best["exc_auc"],
                       exc_ap=best["exc_ap"],
                       inh_auc=max(r["inh_auc"] for r in sweep["per_lag"]),
                       inh_ap=max(r["inh_ap"] for r in sweep["per_lag"]))
        row.update(typed_metrics(sd, glm_dir, n))
        arms_path = os.path.join(glm_dir, "burstexcl_arms.json")
        if os.path.exists(arms_path):
            with open(arms_path, encoding="utf-8") as fh:
                arms = json.load(fh)
            row.update(
                wprot_full_precision=arms["full"]["precision"],
                wprot_full_recall=arms["full"]["recall"],
                wprot_excl_precision=arms["excl"]["precision"],
                wprot_excl_recall=arms["excl"]["recall"],
                excl_dropped_frac=arms["dropped_bins"] / arms["total_bins"])
        rows.append(row)
    return rows


def figures(rows):
    g = {r["session"]: r for r in rows}
    color = lambda r: C50 if r["group"] == "c50" else C40
    have_arms = [r for r in rows if "wprot_full_precision" in r]

    # 1. dose-response
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for grp, col in (("c50", C50), ("c40", C40)):
        rr = [r for r in rows if r["group"] == grp and "typed_precision" in r]
        ax1.plot([r["bursts_per_rec"] for r in rr], [r["typed_precision"] for r in rr],
                 "o", color=col, ms=8, label="%s (%d nets)" % (grp, len(rr)))
        rr2 = [r for r in rows if r["group"] == grp and "exc_auc" in r]
        ax2.plot([r["bursts_per_rec"] for r in rr2], [r["exc_auc"] for r in rr2],
                 "o", color=col, ms=8, label=grp)
    ax1.set_xlabel("network bursts per 60 s recording"); ax1.set_ylabel("edge precision (typed, FDR 0.70)")
    ax1.set_title("Precision falls with burstiness -- both groups, one curve")
    ax2.set_xlabel("network bursts per 60 s recording"); ax2.set_ylabel("excitatory AUC")
    ax2.set_title("Ranking quality (AUC) degrades too, more gently")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dose_response.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)

    # 2. exclusion effect
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in have_arms:
        x = r["bursts_per_rec"]
        ax.annotate("", xy=(x, r["wprot_excl_precision"]), xytext=(x, r["wprot_full_precision"]),
                    arrowprops=dict(arrowstyle="-|>", color=color(r), lw=1.6, alpha=0.85))
        ax.plot(x, r["wprot_full_precision"], "o", color=color(r), ms=6)
        ax.plot(x, r["wprot_excl_precision"], "^", color="#2e8b57", ms=7)
    ax.plot([], [], "o", color=C50, label="full recording (c50)")
    ax.plot([], [], "o", color=C40, label="full recording (c40)")
    ax.plot([], [], "^", color="#2e8b57", label="burst windows excluded")
    ax.set_xlabel("network bursts per 60 s recording"); ax.set_ylabel("edge precision (|W|, FDR 0.70)")
    ax.set_title("Excluding <2%% of bins repairs the burst common-input damage (%d networks)"
                 % len(have_arms))
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "exclusion_effect.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)

    # 3. size vs bursting
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for grp, col in (("c50", C50), ("c40", C40)):
        rr = [r for r in rows if r["group"] == grp]
        ax.plot([r["n_neurons"] for r in rr], [r["bursts_per_rec"] for r in rr],
                "o", color=col, ms=8, label=grp)
    ax.set_xlabel("realized network size N"); ax.set_ylabel("bursts per 60 s recording")
    ax.set_title("Bursting needs critical mass: burst rate vs realized N\n"
                 "(identical parameters and neuron density everywhere)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "size_vs_bursting.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)

    # 4. typed vs untyped precision
    both = [r for r in have_arms if "typed_precision" in r]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for r in both:
        ax.plot(r["wprot_full_precision"], r["typed_precision"], "o",
                color=color(r), ms=8)
        if r["wprot_full_precision"] - r["typed_precision"] > 0.12:
            ax.annotate("seed%02d" % r["topology_seed"],
                        (r["wprot_full_precision"], r["typed_precision"]),
                        textcoords="offset points", xytext=(6, -4), fontsize=8)
    lim = (0.4, 1.0)
    ax.plot(lim, lim, "--", color="grey", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("untyped |W| precision"); ax.set_ylabel("typed (E/I) precision")
    ax.set_title("E/I typing costs precision only in near-burstless networks")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "typed_vs_untyped.png"),
                                    dpi=140, facecolor="white"); plt.close(fig)


def scaling_by_group(rows):
    """Performance vs recording duration, one column per cluster-size group
    (10 curves each, colored by burstiness), plus a minutes-to-criterion
    summary panel."""
    data = {}
    for r in rows:
        p = os.path.join(results_dir(r["session"], STATE, "glm", create=False),
                         "scaling.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                data[r["session"]] = (r, json.load(fh)["grid"])
    if not data:
        return
    bmax = max(r["bursts_per_rec"] for r, _ in data.values()) or 1.0
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True)
    for j, grp in enumerate(("c50", "c40")):
        for session, (r, grid) in sorted(data.items()):
            if r["group"] != grp:
                continue
            c = cmap(r["bursts_per_rec"] / bmax)
            mins = [g["n_recordings"] for g in grid]        # 60 s each -> minutes
            axes[0][j].plot(mins, [g["auc"] for g in grid], "o-", color=c,
                            ms=3.5, lw=1.3, alpha=0.85)
            axes[1][j].plot(mins, [g["precision"] for g in grid], "o-", color=c,
                            ms=3.5, lw=1.3, alpha=0.85)
        for i, metric in enumerate(("auc", "precision")):
            grids = [g for r, g in data.values() if r["group"] == grp]
            xs = [p["n_recordings"] for p in grids[0]]
            mean = [np.mean([g[k][metric] for g in grids if k < len(g)])
                    for k in range(len(xs))]
            axes[i][j].plot(xs, mean, "-", color="black", lw=2.5, alpha=0.75)
        axes[0][j].set_title("%s (%d networks; black = group mean)"
                             % (grp, sum(1 for r, _ in data.values() if r["group"] == grp)))
        axes[1][j].set_xlabel("minutes of recording (60 s each)")
        axes[0][j].set_ylim(0.55, 1.0); axes[1][j].set_ylim(0, 1.0)
        for i in (0, 1):
            axes[i][j].grid(alpha=0.3)
            axes[i][j].set_xscale("log")
            axes[i][j].set_xticks([10, 20, 50, 100, 200])
            axes[i][j].set_xticklabels([10, 20, 50, 100, 200])
    axes[0][0].set_ylabel("excitatory+inhibitory AUC")
    axes[1][0].set_ylabel("precision @ FDR 0.70")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, bmax))
    fig.colorbar(sm, ax=axes, label="bursts per recording", shrink=0.85)
    fig.suptitle("Inference performance vs recording duration -- %s state" % STATE)
    fig.savefig(os.path.join(OUT, "scaling_by_group.png"), dpi=140,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # minutes to reach AUC >= 0.9 (linear interpolation on the grid)
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for session, (r, grid) in sorted(data.items()):
        xs = np.array([g["n_recordings"] for g in grid], float)
        ys = np.array([g["auc"] for g in grid], float)
        above = np.nonzero(ys >= 0.9)[0]
        col = C50 if r["group"] == "c50" else C40
        if len(above):
            k = above[0]
            need = xs[k] if k == 0 else np.interp(0.9, [ys[k-1], ys[k]], [xs[k-1], xs[k]])
            ax.plot(r["bursts_per_rec"], need, "o" if r["group"] == "c50" else "s",
                    color=col, ms=8)
        else:
            ax.plot(r["bursts_per_rec"], 200, "x", color=col, ms=10, mew=2)
    ax.plot([], [], "o", color=C50, label="c50")
    ax.plot([], [], "s", color=C40, label="c40")
    ax.plot([], [], "kx", label="never reaches 0.9 (plotted at 200)")
    ax.set_xlabel("bursts per recording")
    ax.set_ylabel("minutes of data to reach AUC 0.9")
    ax.set_title("How much recording does inference need? -- %s state" % STATE)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "minutes_to_criterion.png"), dpi=140,
                facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = gather()
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "session", k))
    csv_path = os.path.join(OUT, "sweep_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("%s: %d networks, %d columns" % (csv_path, len(rows), len(keys)))
    figures(rows)
    scaling_by_group(rows)
    for f in ("dose_response", "exclusion_effect", "size_vs_bursting", "typed_vs_untyped"):
        print("  %s.png" % os.path.join(OUT, f))


if __name__ == "__main__":
    main()
