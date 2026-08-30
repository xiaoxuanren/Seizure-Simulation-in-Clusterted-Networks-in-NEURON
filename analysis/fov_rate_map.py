"""Render a two-photon-style field-of-view firing-rate map for a session.

Produces a figure in the style of an in-vivo calcium-imaging FOV: a synthetic
grayscale background with neuron-like morphology (somata + dendritic
processes rendered at the *real* simulated neuron positions), overlaid with
one irregular colored ROI per neuron, color-coded by its mean firing rate
(jet colormap, colorbar in Hz), plus a 100-um scale bar.

The grayscale background is SYNTHESIZED for presentation purposes -- the
model has no imaging channel. Everything quantitative (ROI positions, firing
rates, colorbar) comes from the saved session data:

* ``network_<ts>.npz``   -> neuron_positions, connections (used to route
  faint neurite fascicles between connected cells)
* ``recording###.npz``   -> spike_times (ms) + duration (ms); rates are
  pooled across all recordings of the session unless ``--recording`` is given.

Positions are stored in abstract "space units" (space_size = 15); the field
is mapped to ``--fov-um`` micrometers (default 1000) for display, which sets
the meaning of the scale bar.

Usage::

    python analysis/fov_rate_map.py "notebooks/NEURON data/normal/20260710_182039"
    python analysis/fov_rate_map.py <session_dir> --recording 0 --decor-cells 0
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(session_dir, recording=None):
    """Return (positions, connections, rates_hz, participation_pct, meta).

    ``participation_pct`` is the percentage of detected network bursts
    (pooled ``burst_windows`` across the recordings used) in which each
    neuron fired at least one spike; all zeros if no bursts were detected.
    """
    net_files = glob.glob(os.path.join(session_dir, "network_*.npz"))
    if not net_files:
        raise FileNotFoundError(f"no network_*.npz in {session_dir}")
    net = np.load(net_files[0], allow_pickle=True)
    positions = net["neuron_positions"]
    connections = net["connections"]

    rec_files = sorted(glob.glob(os.path.join(session_dir, "recording*.npz")))
    rec_files = [f for f in rec_files if "raster" not in os.path.basename(f)]
    if recording is not None:
        rec_files = [f for f in rec_files
                     if f.endswith(f"recording{recording:03d}.npz")]
    if not rec_files:
        raise FileNotFoundError(
            f"no recording npz files in {session_dir} "
            f"(recording filter: {recording})")

    n = len(positions)
    counts = np.zeros(n)
    part_counts = np.zeros(n)
    total_ms = 0.0
    total_bursts = 0
    for f in rec_files:
        rec = np.load(f, allow_pickle=True)
        st = rec["spike_times"]
        counts += np.array([len(s) for s in st], dtype=float)
        total_ms += float(rec["duration"])
        bw = np.asarray(rec["burst_windows"], dtype=float)
        if bw.size:
            total_bursts += len(bw)
            for i, s in enumerate(st):
                if len(s):
                    arr = np.asarray(s, dtype=float)
                    in_win = (np.searchsorted(arr, bw[:, 1]) -
                              np.searchsorted(arr, bw[:, 0])) > 0
                    part_counts[i] += in_win.sum()
    rates = counts / (total_ms / 1000.0)
    participation = (part_counts / total_bursts * 100.0 if total_bursts
                     else np.zeros(n))

    meta = {}
    meta_file = os.path.join(session_dir, "session_metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file) as fh:
            meta = json.load(fh)
    meta["_n_recordings_used"] = len(rec_files)
    meta["_total_seconds"] = total_ms / 1000.0
    meta["_total_bursts"] = total_bursts
    return positions, connections, rates, participation, meta


# ---------------------------------------------------------------------------
# Synthetic two-photon background
# ---------------------------------------------------------------------------

def _splat(img, xs, ys, amps):
    """Bilinear-splat point samples (px coords) into ``img``."""
    h, w = img.shape
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx
        yi = y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(img, (yi[ok], xi[ok]), amps[ok] * wgt[ok])


def _wiggly_path(p0, p1, rng, sag_um=25.0, n=None):
    """Quadratic Bezier from p0 to p1 (um coords) with a random lateral sag."""
    d = p1 - p0
    length = np.hypot(*d)
    if n is None:
        n = max(8, int(length * 1.5))
    perp = np.array([-d[1], d[0]]) / (length + 1e-9)
    ctrl = (p0 + p1) / 2 + perp * rng.normal(0, sag_um)
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    # small high-frequency wobble so fibres don't look vector-drawn
    wob = rng.normal(0, 0.6, (n, 2))
    wob = gaussian_filter(wob, (3, 0))
    return pts + wob


def _random_branch(origin, rng, length_um):
    """Random-walk-with-momentum dendrite starting at ``origin`` (um)."""
    step = 1.2
    n = int(length_um / step)
    ang = rng.uniform(0, 2 * np.pi)
    angs = ang + np.cumsum(rng.normal(0, 0.18, n))
    steps = np.stack([np.cos(angs), np.sin(angs)], axis=1) * step
    return origin + np.cumsum(steps, axis=0)


def _stamp_soma(img, cx, cy, r_px, brightness, rng):
    """Draw one irregular soma (bright rim + dimmer interior) in px coords."""
    pad = int(r_px * 2.5) + 2
    x0, x1 = int(cx) - pad, int(cx) + pad + 1
    y0, y1 = int(cy) - pad, int(cy) + pad + 1
    h, w = img.shape
    if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
        return
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x1, w), min(y1, h)
    yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
    dx = xx - cx
    dy = yy - cy
    d = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    # irregular radius: low-order angular harmonics
    r_t = r_px * (1.0
                  + 0.13 * np.cos(2 * theta + rng.uniform(0, 6.28))
                  + 0.09 * np.cos(3 * theta + rng.uniform(0, 6.28)))
    rim = 0.7 * np.exp(-((d - r_t) / (0.55 * r_px)) ** 2)
    interior = 0.6 * np.exp(-(d / r_t) ** 2)
    halo = 0.25 * np.exp(-(d / (2.2 * r_t)) ** 2)
    img[ys0:ys1, xs0:xs1] += brightness * (rim + interior + halo)


def render_background(positions_um, connections, extent, px_per_um, rng,
                      n_decor=400):
    """Synthesize a 2p-microscopy-style grayscale image. Returns float image."""
    x0, x1, y0, y1 = extent
    w = int((x1 - x0) * px_per_um)
    h = int((y1 - y0) * px_per_um)

    def to_px(pts_um):
        return ((pts_um[:, 0] - x0) * px_per_um,
                (pts_um[:, 1] - y0) * px_per_um)

    img = np.zeros((h, w))

    # --- neuropil: multi-scale blurred noise ------------------------------
    for sigma, amp in ((2, 0.040), (6, 0.060), (18, 0.080)):
        img += amp * gaussian_filter(rng.standard_normal((h, w)), sigma)
    img -= img.min()

    fibres = np.zeros((h, w))

    # --- fascicles along a sample of real connections ---------------------
    n_conn = len(connections)
    idx = rng.permutation(n_conn)[:min(1500, n_conn)]
    for i in idx:
        pre, post = int(connections[i][0]), int(connections[i][1])
        p0, p1 = positions_um[pre], positions_um[post]
        if np.hypot(*(p1 - p0)) < 8:
            continue
        pts = _wiggly_path(p0, p1, rng)
        xs = (pts[:, 0] - x0) * px_per_um
        ys = (pts[:, 1] - y0) * px_per_um
        amp = rng.uniform(0.06, 0.18)
        amps = np.full(len(pts), amp) * rng.uniform(0.6, 1.4, len(pts))
        _splat(fibres, xs, ys, amps)

    # --- decor cells: unlabeled background neurons (cosmetic only) --------
    if n_decor:
        n_near = int(0.35 * n_decor)
        near = (positions_um[rng.integers(0, len(positions_um), n_near)]
                + rng.normal(0, 55.0, (n_near, 2)))
        far = rng.uniform([x0, y0], [x1, y1], (n_decor - n_near, 2))
        decor = np.vstack([near, far])
    else:
        decor = np.zeros((0, 2))

    # --- dendrites from every real (and decor) soma -----------------------
    for pts_um, gain in ((positions_um, 1.0), (decor, 0.55)):
        for p in pts_um:
            for _ in range(rng.integers(3, 7)):
                br = _random_branch(p, rng, rng.uniform(40, 160))
                xs = (br[:, 0] - x0) * px_per_um
                ys = (br[:, 1] - y0) * px_per_um
                n = len(br)
                fade = np.linspace(1.0, 0.3, n)
                amps = gain * rng.uniform(0.12, 0.35) * fade
                _splat(fibres, xs, ys, amps)

    fibres = gaussian_filter(fibres, 1.1)
    img += fibres

    # --- somata (fuzzy blob + modest rim + local glow) --------------------
    xs, ys = to_px(positions_um)
    for cx, cy in zip(xs, ys):
        r_px = rng.uniform(4.5, 6.5) * px_per_um
        _stamp_soma(img, cx, cy, r_px, rng.lognormal(-0.45, 0.35), rng)
    if n_decor:
        dxs, dys = to_px(decor)
        for cx, cy in zip(dxs, dys):
            r_px = rng.uniform(3.5, 5.5) * px_per_um
            _stamp_soma(img, cx, cy, r_px, 0.45 * rng.lognormal(-0.45, 0.5),
                        rng)

    # --- diffuse glow where cells/fibres are dense ------------------------
    img += 0.30 * gaussian_filter(img, 25 * px_per_um)

    img = gaussian_filter(img, 0.9)

    # --- shot noise, vignette, tone curve ---------------------------------
    img += rng.normal(0, 0.030, (h, w)) * (0.6 + np.sqrt(np.clip(img, 0, None)))
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    img *= 1.0 - 0.15 * np.clip(rr, 0, 1.2) ** 2
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / (hi - lo), 0, 1) ** 0.60
    return img


# ---------------------------------------------------------------------------
# ROI overlay
# ---------------------------------------------------------------------------

def roi_polygon(center, rng, r_um=4.2):
    """Irregular blob outline (um coords) mimicking a segmented soma ROI."""
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = r_um * (1.0
                + 0.22 * np.cos(2 * theta + rng.uniform(0, 6.28))
                + 0.15 * np.cos(3 * theta + rng.uniform(0, 6.28))
                + 0.10 * np.cos(5 * theta + rng.uniform(0, 6.28)))
    r *= rng.uniform(0.75, 1.25)
    return np.column_stack([center[0] + r * np.cos(theta),
                            center[1] + r * np.sin(theta)])


def nice_limits(rates, q=(5, 95), step=0.05):
    """Rate colour limits rounded outward to ``step``."""
    lo, hi = np.percentile(rates, q)
    vmin = np.floor(lo / step) * step
    vmax = np.ceil(hi / step) * step
    if vmax <= vmin:
        vmax = vmin + step
    return float(vmin), float(vmax)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(session_dir, out=None, recording=None, fov_um=1000.0,
                seed=0, n_decor=400, vmin=None, vmax=None, metric="rate"):
    """Render one map. ``metric`` is ``"rate"`` (Hz, auto-scaled colorbar)
    or ``"participation"`` (% of network bursts joined, fixed 0-100)."""
    (positions, connections, rates, participation,
     meta) = load_session(session_dir, recording)
    if metric not in ("rate", "participation"):
        raise ValueError(f"unknown metric {metric!r}")
    values = rates if metric == "rate" else participation
    rng = np.random.default_rng(seed)

    # map space units -> um: the position bounding square spans ~fov_um
    span = float(max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])))
    um_per_unit = fov_um / span
    pos_um = (positions - positions.min(axis=0)) * um_per_unit
    # square field of view centred on the cells
    pad = 35.0
    side = max(pos_um[:, 0].max(), pos_um[:, 1].max()) + 2 * pad
    cx, cy = pos_um[:, 0].max() / 2, pos_um[:, 1].max() / 2
    extent = (cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)

    px_per_um = 1.4
    img = render_background(pos_um, connections, extent, px_per_um, rng,
                            n_decor=n_decor)

    if metric == "participation":
        vmin = 0.0 if vmin is None else vmin
        vmax = 100.0 if vmax is None else vmax
        cbar_title = "(%)"
    else:
        if vmin is None or vmax is None:
            auto = nice_limits(values)
            vmin = auto[0] if vmin is None else vmin
            vmax = auto[1] if vmax is None else vmax
        cbar_title = "(Hz)"
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin, vmax)

    fig = plt.figure(figsize=(9.6, 8.4), facecolor="white")
    ax = fig.add_axes([0.01, 0.02, 0.80, 0.96])
    ax.imshow(img, cmap="gray", origin="lower",
              extent=[extent[0], extent[1], extent[2], extent[3]],
              vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()

    for p, v in zip(pos_um, values):
        poly = roi_polygon(p, rng)
        ax.add_patch(Polygon(poly, closed=True,
                             facecolor=cmap(norm(np.clip(v, vmin, vmax))),
                             edgecolor="white", linewidth=0.35, alpha=0.95))

    # scale bar (100 um), bottom-left
    bx = extent[0] + 25
    by = extent[2] + 25
    ax.plot([bx, bx + 100], [by, by], color="white", lw=3,
            solid_capstyle="butt")
    ax.text(bx + 2, by + 8, "100 μm", color="white", fontsize=10)

    # colorbar, short and vertically centred like the reference figure
    cax = fig.add_axes([0.865, 0.28, 0.022, 0.42])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.locator = matplotlib.ticker.MaxNLocator(6)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=10)
    cax.set_title(cbar_title, fontsize=11, pad=12)

    state = meta.get("state", {}).get("state_name", "unknown")
    if out is None:
        tag = "all" if recording is None else f"{recording:03d}"
        out = os.path.join(session_dir, f"fov_{metric}_map_rec{tag}.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)

    unit = "Hz" if metric == "rate" else "%"
    print(f"state={state}  neurons={len(values)}  "
          f"recordings={meta['_n_recordings_used']}  "
          f"pooled={meta['_total_seconds']:.0f}s  "
          f"bursts={meta['_total_bursts']}")
    print(f"{metric} {unit}: min={values.min():.3f} mean={values.mean():.3f} "
          f"max={values.max():.3f}  colorbar=[{vmin:.2f}, {vmax:.2f}]")
    print(f"saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", help="session directory (contains "
                    "network_*.npz and recording###.npz)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--recording", type=int, default=None,
                    help="use only this recording index (default: pool all)")
    ap.add_argument("--fov-um", type=float, default=1000.0,
                    help="physical width assigned to the position field")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the synthetic background")
    ap.add_argument("--decor-cells", type=int, default=400,
                    help="cosmetic unlabeled background cells (0 disables)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--metric", choices=("rate", "participation"),
                    default="rate",
                    help="color by firing rate (Hz) or network-burst "
                         "participation (%%, fixed 0-100 scale)")
    args = ap.parse_args()
    make_figure(args.session, out=args.out, recording=args.recording,
                fov_um=args.fov_um, seed=args.seed,
                n_decor=args.decor_cells, vmin=args.vmin, vmax=args.vmax,
                metric=args.metric)


if __name__ == "__main__":
    main()
