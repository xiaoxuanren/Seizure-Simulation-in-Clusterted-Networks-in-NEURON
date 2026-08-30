"""Composite imaging-style figure: FOV map + zoom insets + colored raster.

Reproduces the classic calcium-imaging composite layout: a large
field-of-view panel (synthetic two-photon-style background, ROIs colored by
firing rate or network-burst participation) with four dashed boxes marking
zoom regions, four matching zoom panels with color-coded dashed borders, a
colorbar, and a spike-raster panel in which every neuron's dots carry its
map color. Builds on :mod:`fov_rate_map` (same data loading, background
synthesis and caveats -- the gray morphology is cosmetic, the ROI positions
and colors are real data).

Zoom regions are picked automatically from the cluster structure: the
lowest-participation cluster, the highest, the most heterogeneous, and the
largest, subject to staying inside the field and apart from each other.

Usage (from repo root)::

    python analysis/fov_composite.py "notebooks/NEURON data parallel/sweep_c50_seed02/seizure"
    python analysis/fov_composite.py <session> --metric rate --recording 5
"""

import argparse
import glob
import os
import zlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from fov_rate_map import load_session, render_background, nice_limits

BOX_COLORS = ("magenta", "limegreen", "gold", "deepskyblue")
DASH = (0, (5, 3))


def neuron_polygons(pos_um, seed, r_um=4.2):
    """Deterministic irregular ROI outline per neuron (independent of draw
    order, so main and zoom panels show identical shapes)."""
    polys = []
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for i, p in enumerate(pos_um):
        rng = np.random.default_rng([seed, i])
        r = r_um * (1.0
                    + 0.22 * np.cos(2 * theta + rng.uniform(0, 6.28))
                    + 0.15 * np.cos(3 * theta + rng.uniform(0, 6.28))
                    + 0.10 * np.cos(5 * theta + rng.uniform(0, 6.28)))
        r *= rng.uniform(0.75, 1.25)
        polys.append(np.column_stack([p[0] + r * np.cos(theta),
                                      p[1] + r * np.sin(theta)]))
    return polys


def pick_zoom_regions(pos_um, values, cluster_ids, half_um, extent, n=4):
    """Choose ``n`` zoom centers from cluster stats (lowest mean value,
    highest mean, most heterogeneous, largest), non-overlapping, in-field."""
    stats = []
    for cid in np.unique(cluster_ids):
        m = cluster_ids == cid
        if m.sum() < 10:
            continue
        stats.append((np.mean(pos_um[m], axis=0), values[m].mean(),
                      values[m].std(), m.sum()))
    if not stats:
        raise ValueError("no clusters with >=10 neurons")
    centers = np.array([s[0] for s in stats])
    means = np.array([s[1] for s in stats])
    stds = np.array([s[2] for s in stats])
    counts = np.array([s[3] for s in stats])

    lo = np.array([extent[0], extent[2]]) + half_um + 5
    hi = np.array([extent[1], extent[3]]) - half_um - 5
    ranked = [int(np.argmin(means)), int(np.argmax(means)),
              int(np.argmax(stds)), int(np.argmax(counts))]
    ranked += [int(i) for i in np.argsort(-stds)]
    chosen = []
    for idx in ranked:
        c = np.clip(centers[idx], lo, hi)
        if all(np.hypot(*(c - p)) > 2.3 * half_um for p in chosen):
            chosen.append(c)
        if len(chosen) == n:
            break
    return chosen


def _strip_axes(ax, color="black", lw=1.0, ls="solid"):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(lw)
        sp.set_linestyle(ls)


def make_composite(session_dir, metric="participation", recording=0,
                   zoom_um=130.0, out=None, fov_um=1000.0, n_decor=400,
                   seed=None):
    (positions, connections, rates, participation,
     meta) = load_session(session_dir)
    values = participation if metric == "participation" else rates
    if metric == "participation":
        vmin, vmax = 0.0, 100.0
        cbar_title = "(%)"
    else:
        vmin, vmax = nice_limits(values)
        cbar_title = "(Hz)"

    if seed is None:  # same derivation as fov_rate_map_sweep -> same background
        sweep_name = os.path.basename(os.path.dirname(
            os.path.abspath(session_dir)))
        seed = zlib.crc32(sweep_name.encode()) % (2 ** 31)
    rng = np.random.default_rng(seed)

    span = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))
    um_per_unit = fov_um / span
    pos_um = (positions - positions.min(axis=0)) * um_per_unit
    pad = 35.0
    side = max(pos_um[:, 0].max(), pos_um[:, 1].max()) + 2 * pad
    cx, cy = pos_um[:, 0].max() / 2, pos_um[:, 1].max() / 2
    extent = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)

    px_per_um = 1.4
    img = render_background(pos_um, connections, extent, px_per_um, rng,
                            n_decor=n_decor)

    net_file = glob.glob(os.path.join(session_dir, "network_*.npz"))[0]
    cluster_ids = np.load(net_file, allow_pickle=True)["cluster_assignments"]

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin, vmax)
    colors = cmap(norm(np.clip(values, vmin, vmax)))
    polys = neuron_polygons(pos_um, seed)
    half = zoom_um / 2.0
    regions = pick_zoom_regions(pos_um, values, cluster_ids, half, extent)

    fig = plt.figure(figsize=(11.4, 9.0), facecolor="white")

    # ---- main FOV -------------------------------------------------------
    axM = fig.add_axes([0.02, 0.355, 0.4912, 0.622])
    axM.imshow(img, cmap="gray", origin="lower",
               extent=list(extent), vmin=0, vmax=1, interpolation="bilinear")
    for poly, col in zip(polys, colors):
        axM.add_patch(Polygon(poly, closed=True, facecolor=col,
                              edgecolor="white", linewidth=0.3, alpha=0.95))
    for (rx, ry), bc in zip(regions, BOX_COLORS):
        axM.add_patch(Rectangle((rx - half, ry - half), zoom_um, zoom_um,
                                fill=False, edgecolor=bc, linewidth=2.0,
                                linestyle=DASH))
    bx, by = extent[0] + 30, extent[2] + 30
    axM.plot([bx, bx + 200], [by, by], color="white", lw=3,
             solid_capstyle="butt")
    axM.text(bx, by + 12, "200 μm", color="white", fontsize=11)
    axM.set_xlim(extent[0], extent[1])
    axM.set_ylim(extent[2], extent[3])
    _strip_axes(axM)

    # ---- zoom panels ----------------------------------------------------
    zoom_rects = ([0.525, 0.672, 0.2386, 0.302],   # right column, top
                  [0.525, 0.355, 0.2386, 0.302],   # right column, bottom
                  [0.02, 0.030, 0.2386, 0.302],    # bottom row, left
                  [0.264, 0.030, 0.2386, 0.302])   # bottom row, mid
    for k, ((rx, ry), rect, bc) in enumerate(
            zip(regions, zoom_rects, BOX_COLORS)):
        ax = fig.add_axes(rect)
        ax.imshow(img, cmap="gray", origin="lower", extent=list(extent),
                  vmin=0, vmax=1, interpolation="bilinear")
        near = np.where(
            (np.abs(pos_um[:, 0] - rx) < half + 12) &
            (np.abs(pos_um[:, 1] - ry) < half + 12))[0]
        for i in near:
            ax.add_patch(Polygon(polys[i], closed=True, facecolor=colors[i],
                                 edgecolor="white", linewidth=1.0,
                                 alpha=0.95))
        if k == 2:  # one 20-um bar, bottom-left zoom (as in the reference)
            zx, zy = rx - half + 8, ry - half + 10
            ax.plot([zx, zx + 20], [zy, zy], color="white", lw=3,
                    solid_capstyle="butt")
            ax.text(zx, zy + 4, "20 μm", color="white", fontsize=10)
        ax.set_xlim(rx - half, rx + half)
        ax.set_ylim(ry - half, ry + half)
        _strip_axes(ax, color=bc, lw=2.2, ls=DASH)

    # ---- colorbar -------------------------------------------------------
    cax = fig.add_axes([0.785, 0.42, 0.017, 0.46])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.locator = matplotlib.ticker.MaxNLocator(6)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=10)
    cax.set_title(cbar_title, fontsize=11, pad=12)

    # ---- raster (rows sorted by cluster, dots in map colors) ------------
    axR = fig.add_axes([0.525, 0.062, 0.30, 0.270])
    rec = np.load(os.path.join(session_dir,
                               f"recording{recording:03d}.npz"),
                  allow_pickle=True)
    st = rec["spike_times"]
    dur_s = float(rec["duration"]) / 1000.0
    order = np.argsort(cluster_ids, kind="stable")
    xs, ys, cs = [], [], []
    for row, i in enumerate(order):
        s = np.asarray(st[i], dtype=float)
        if s.size:
            xs.append(s / 1000.0)
            ys.append(np.full(s.size, row))
            cs.append(np.tile(colors[i], (s.size, 1)))
    axR.scatter(np.concatenate(xs), np.concatenate(ys), s=1.5,
                c=np.concatenate(cs), marker=".", linewidths=0,
                rasterized=True)
    axR.set_xlim(0, dur_s)
    axR.set_ylim(-2, len(order) + 1)
    axR.set_yticks([])
    axR.set_xticks(np.arange(0, dur_s + 1, 20))
    axR.tick_params(labelsize=8)
    axR.set_xlabel("Time (s)", fontsize=9)
    for sp in axR.spines.values():
        sp.set_linewidth(1.0)

    state = meta.get("state", {}).get("state_name", "unknown")
    if out is None:
        out = os.path.join(session_dir, f"fov_{metric}_composite.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)

    unit = "%" if metric == "participation" else "Hz"
    print(f"state={state}  neurons={len(values)}  "
          f"bursts={meta['_total_bursts']}  raster=recording{recording:03d}")
    print(f"{metric} {unit}: min={values.min():.2f} mean={values.mean():.2f} "
          f"max={values.max():.2f}  colorbar=[{vmin:.0f}, {vmax:.0f}]")
    print(f"zoom regions (um): "
          + ", ".join(f"({r[0]:.0f}, {r[1]:.0f})" for r in regions))
    print(f"saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", help="session directory")
    ap.add_argument("--metric", choices=("rate", "participation"),
                    default="participation")
    ap.add_argument("--recording", type=int, default=0,
                    help="recording index shown in the raster panel")
    ap.add_argument("--zoom-um", type=float, default=130.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fov-um", type=float, default=1000.0)
    ap.add_argument("--decor-cells", type=int, default=400)
    ap.add_argument("--seed", type=int, default=None,
                    help="background seed (default: derived from the sweep "
                         "folder name, matching fov_rate_map_sweep)")
    args = ap.parse_args()
    make_composite(args.session, metric=args.metric,
                   recording=args.recording, zoom_um=args.zoom_um,
                   out=args.out, fov_um=args.fov_um,
                   n_decor=args.decor_cells, seed=args.seed)


if __name__ == "__main__":
    main()
