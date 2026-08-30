"""Batch-render FOV firing-rate maps for the parallel sweep sessions.

For every ``sweep_c40_*`` / ``sweep_c50_*`` directory under
``notebooks/NEURON data parallel`` this renders one map per state
(normal + seizure) with :mod:`fov_rate_map`. The synthetic background uses
the same RNG seed for both states of a sweep, so within a pair the gray
morphology is pixel-identical and only the ROI colors (firing rates)
change. Color limits are chosen per state (5th/95th percentile, like the
reference figure) unless ``--shared-scale`` is given, which uses common
limits across the pair for direct normal-vs-seizure comparison.

Outputs land next to the data (``<sweep>/<state>/fov_rate_map.png``) and are
also collected in ``<root>/sweep_summary/fov_rate_maps/<sweep>_<state>.png``.

Usage (from repo root)::

    python analysis/fov_rate_map_sweep.py            # all 20 sweeps
    python analysis/fov_rate_map_sweep.py --only sweep_c50_seed01 --workers 1
"""

import argparse
import glob
import os
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_ROOT = os.path.join("notebooks", "NEURON data parallel")


def render_one(sweep_dir, state, collect_dir, shared_scale, metric):
    """Worker: render one (sweep, state) map. Returns (label, out, err)."""
    from fov_rate_map import load_session, make_figure, nice_limits
    import numpy as np

    # participation always uses the fixed 0-100 scale, shared by definition
    shared_scale = shared_scale and metric == "rate"
    suffix = {"rate": "_shared" if shared_scale else "",
              "participation": "_participation"}[metric]
    stem = "fov_rate_map" if metric == "rate" else "fov_participation_map"
    label = f"{os.path.basename(sweep_dir)}/{state}{suffix}"
    try:
        sess = os.path.join(sweep_dir, state)
        vmin = vmax = None
        if shared_scale:
            rates = np.concatenate([
                load_session(os.path.join(sweep_dir, s))[2]
                for s in ("normal", "seizure")])
            vmin, vmax = nice_limits(rates)
        seed = zlib.crc32(os.path.basename(sweep_dir).encode()) % (2 ** 31)
        out = os.path.join(sess, f"{stem}{'_shared' if shared_scale else ''}.png")
        make_figure(sess, out=out, seed=seed, vmin=vmin, vmax=vmax,
                    metric=metric)
        dst = os.path.join(
            collect_dir, f"{os.path.basename(sweep_dir)}_{state}{suffix}.png")
        shutil.copyfile(out, dst)
        return label, out, None
    except Exception as exc:  # noqa: BLE001 - report and keep the batch going
        return label, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None,
                    help="render only sweep dirs containing this substring")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shared-scale", action="store_true",
                    help="one colour scale per sweep (normal+seizure pooled)")
    ap.add_argument("--metric", choices=("rate", "participation"),
                    default="rate")
    args = ap.parse_args()

    sweeps = sorted(glob.glob(os.path.join(args.root, "sweep_c[45]0_seed*")))
    sweeps = [s for s in sweeps if os.path.isdir(s)]
    if args.only:
        sweeps = [s for s in sweeps if args.only in os.path.basename(s)]
    if not sweeps:
        raise SystemExit(f"no sweep dirs found under {args.root}")

    collect_dir = os.path.join(args.root, "sweep_summary", "fov_rate_maps")
    os.makedirs(collect_dir, exist_ok=True)

    jobs = [(s, st) for s in sweeps for st in ("normal", "seizure")
            if os.path.isdir(os.path.join(s, st))]
    print(f"{len(jobs)} maps to render from {len(sweeps)} sweeps "
          f"({args.workers} workers)")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_one, s, st, collect_dir,
                            args.shared_scale, args.metric): (s, st)
                for s, st in jobs}
        for fut in as_completed(futs):
            label, out, err = fut.result()
            if err:
                failures.append((label, err))
                print(f"FAIL  {label}: {err}")
            else:
                print(f"done  {label}")
    print(f"\ncollected in {collect_dir}")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for label, err in failures:
            print(f"  {label}: {err}")


if __name__ == "__main__":
    main()
