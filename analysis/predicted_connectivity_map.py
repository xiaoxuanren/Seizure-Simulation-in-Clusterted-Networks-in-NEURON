"""Predicted-connectivity maps with neurons colored by rate/participation.

For every sweep session and state this draws the GLM-predicted wiring
(``pred_adjacency`` from ``results/<state>/glm/glm_connectivity_sum4_5ms.npz``,
the label-free sum4 @ FDR 0.70 operating point) on a white background, with
every edge colored by correctness against the saved ground truth (same
convention as ``glm_predicted_topology.py``: TP green, FP red, FN orange
dotted; FP restricted to the candidate mask) and the TP/FP/FN counts plus
precision/recall/F1 in the on-figure legend. Each neuron is colored by its
pooled mean firing rate (jet, 5th-95th percentile colorbar in Hz) and, in a
second figure, by its network-burst participation (fixed 0-100 % scale).
The rate/participation values are identical to the FOV map figures (same
``load_session`` pooling), and positions use the same space-units -> 1000 um
mapping, so the 100 um scale bar means the same thing across the suite.

Outputs per (sweep, state)::

    <sweep>/results/<state>/figures/predicted_connectivity_rate.png
    <sweep>/results/<state>/figures/predicted_connectivity_participation.png

plus collected copies in ``<root>/sweep_summary/predicted_connectivity/``.

Usage (from repo root)::

    python analysis/predicted_connectivity_map.py                # all sweeps
    python analysis/predicted_connectivity_map.py --only c50_seed02 --workers 1
"""

import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_ROOT = os.path.join("notebooks", "NEURON data parallel")

# same correctness palette as glm_predicted_topology.py
GREEN, RED, ORANGE = "#1a9850", "#d73027", "#f0a000"

# node colormaps, chosen to share NO hue with the green/red/orange
# correctness edges (jet would collide with all three), and to differ from
# each other: rate = cyan->magenta, participation = light->dark blue
NODE_CMAP = "cool"
PART_CMAP = "Blues"


def _node_cmap(name, truncate_lo=0.0):
    """Colormap for nodes; ``truncate_lo`` cuts a too-pale light end."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    base = plt.get_cmap(name)
    if truncate_lo <= 0:
        return base
    return ListedColormap(base(np.linspace(truncate_lo, 1.0, 256)))


def render_pair(sweep_dir, state, collect_dir, cmap_name=NODE_CMAP,
                cmap_part_name=PART_CMAP):
    """Worker: rate + participation connectivity maps for one (sweep, state)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from fov_rate_map import load_session, nice_limits

    label = f"{os.path.basename(sweep_dir)}/{state}"
    try:
        sess = os.path.join(sweep_dir, state)
        glm_file = os.path.join(sweep_dir, "results", state, "glm",
                                "glm_connectivity_sum4_5ms.npz")
        res = np.load(glm_file, allow_pickle=True)
        pred = res["pred_adjacency"].astype(bool)
        cand = res["candidates"].astype(bool)
        A_edge = res["A_exc"].astype(bool) | res["A_inh"].astype(bool)
        np.fill_diagonal(pred, False)
        np.fill_diagonal(A_edge, False)

        tp = pred & A_edge
        fp = pred & ~A_edge & cand
        fn = A_edge & ~pred
        TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
        Pr = TP / (TP + FP) if TP + FP else 0.0
        Rc = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * Pr * Rc / (Pr + Rc) if Pr + Rc else 0.0

        (positions, _conn, rates, participation,
         meta) = load_session(sess)

        fov_um = 1000.0
        span = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))
        pos_um = (positions - positions.min(axis=0)) * (fov_um / span)
        pad = 35.0
        side = max(pos_um[:, 0].max(), pos_um[:, 1].max()) + 2 * pad
        cx, cy = pos_um[:, 0].max() / 2, pos_um[:, 1].max() / 2
        extent = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)

        segs_tp = pos_um[np.argwhere(tp)]  # (n, 2, 2)
        segs_fp = pos_um[np.argwhere(fp)]
        segs_fn = pos_um[np.argwhere(fn)]
        n_edges = int(pred.sum())

        outs = []
        for metric, values in (("rate", rates),
                               ("participation", participation)):
            if metric == "participation":
                vmin, vmax = 0.0, 100.0
                cbar_title = "(%)"
            else:
                vmin, vmax = nice_limits(values)
                cbar_title = "(Hz)"
            if metric == "participation":
                cmap = _node_cmap(cmap_part_name, truncate_lo=0.2)
            else:
                cmap = _node_cmap(cmap_name)
            norm = plt.Normalize(vmin, vmax)

            # keep dense predicted graphs legible: thinner, fainter edges
            lw, alpha = (0.35, 0.60) if n_edges < 15000 else (0.25, 0.40)

            fig = plt.figure(figsize=(9.6, 8.4), facecolor="white")
            ax = fig.add_axes([0.01, 0.02, 0.80, 0.96])
            ax.set_facecolor("white")
            ax.add_collection(LineCollection(
                segs_fn, colors=ORANGE, lw=0.9 * lw, alpha=0.7 * alpha,
                linestyles="dotted", zorder=1))
            ax.add_collection(LineCollection(
                segs_tp, colors=GREEN, lw=lw, alpha=alpha, zorder=1.5))
            ax.add_collection(LineCollection(
                segs_fp, colors=RED, lw=lw, alpha=alpha, zorder=2))
            ax.legend(handles=[
                Line2D([0], [0], color=GREEN, lw=2,
                       label=f"TP ({TP:,} correct)"),
                Line2D([0], [0], color=RED, lw=2,
                       label=f"FP ({FP:,} false)"),
                Line2D([0], [0], color=ORANGE, lw=2, ls=":",
                       label=f"FN ({FN:,} missed)"),
            ], loc="upper right", fontsize=9, framealpha=0.9,
                title=f"P={Pr:.2f}  R={Rc:.2f}  F1={F1:.2f}",
                title_fontsize=9)
            ax.scatter(pos_um[:, 0], pos_um[:, 1], s=26,
                       c=np.clip(values, vmin, vmax), cmap=cmap, norm=norm,
                       edgecolors="white", linewidths=0.4, zorder=3)
            bx, by = extent[0] + 25, extent[2] + 25
            ax.plot([bx, bx + 100], [by, by], color="black", lw=3,
                    solid_capstyle="butt", zorder=4)
            ax.text(bx + 2, by + 10, "100 μm", color="black", fontsize=10)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_axis_off()

            cax = fig.add_axes([0.865, 0.28, 0.022, 0.42])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cb = fig.colorbar(sm, cax=cax)
            cb.locator = matplotlib.ticker.MaxNLocator(6)
            cb.update_ticks()
            cb.ax.tick_params(labelsize=10)
            cax.set_title(cbar_title, fontsize=11, pad=12)

            fig_dir = os.path.join(sweep_dir, "results", state, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            out = os.path.join(fig_dir,
                               f"predicted_connectivity_{metric}.png")
            fig.savefig(out, dpi=200, facecolor="white")
            plt.close(fig)
            shutil.copyfile(out, os.path.join(
                collect_dir,
                f"{os.path.basename(sweep_dir)}_{state}_{metric}.png"))
            outs.append(out)

        print(f"{label}: {len(pos_um)} neurons, {n_edges} predicted edges, "
              f"TP={TP} FP={FP} FN={FN} P={Pr:.2f} R={Rc:.2f} F1={F1:.2f} "
              f"-> {outs[0]}")
        return label, None
    except Exception as exc:  # noqa: BLE001 - report and keep the batch going
        return label, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cmap", default=NODE_CMAP,
                    help="node colormap for the rate figure (default 'cool'; "
                         "avoid green/red maps -- those hues mean TP/FP)")
    ap.add_argument("--cmap-participation", default=PART_CMAP,
                    help="node colormap for the participation figure "
                         "(default 'Blues', light end truncated)")
    args = ap.parse_args()

    sweeps = sorted(glob.glob(os.path.join(args.root, "sweep_c[45]0_seed*")))
    sweeps = [s for s in sweeps if os.path.isdir(s)]
    if args.only:
        sweeps = [s for s in sweeps if args.only in os.path.basename(s)]
    if not sweeps:
        raise SystemExit(f"no sweep dirs found under {args.root}")

    collect_dir = os.path.join(args.root, "sweep_summary",
                               "predicted_connectivity")
    os.makedirs(collect_dir, exist_ok=True)

    jobs = [(s, st) for s in sweeps for st in ("normal", "seizure")
            if os.path.isdir(os.path.join(s, st))]
    print(f"{len(jobs)} (sweep, state) pairs x 2 metrics "
          f"({args.workers} workers)")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_pair, s, st, collect_dir, args.cmap,
                            args.cmap_participation): (s, st)
                for s, st in jobs}
        for fut in as_completed(futs):
            label, err = fut.result()
            if err:
                failures.append((label, err))
                print(f"FAIL  {label}: {err}")
    print(f"\ncollected in {collect_dir}")
    for label, err in failures:
        print(f"  FAILED {label}: {err}")


if __name__ == "__main__":
    main()
