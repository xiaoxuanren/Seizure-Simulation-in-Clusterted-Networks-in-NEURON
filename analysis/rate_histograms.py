"""Overlaid normal-vs-seizure firing-rate histograms per sweep session.

For each ``sweep_c40_*`` / ``sweep_c50_*`` pair this plots the distribution
of per-neuron mean firing rates (pooled across all recordings, same numbers
as the FOV rate maps) for both states in one panel: semi-transparent
histograms on shared bins with Gaussian fits, legend "normal state" /
"seizure state" -- the classic before/after-drug comparison layout.

Output: ``<sweep>/results/rate_histogram.png`` plus a collected copy in
``<root>/sweep_summary/rate_histograms/<sweep>.png``.

Usage (from repo root)::

    python analysis/rate_histograms.py                 # all sweeps
    python analysis/rate_histograms.py --only c50_seed02 --workers 1
"""

import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_ROOT = os.path.join("notebooks", "NEURON data parallel")

NORMAL_FILL = "#a0a0ee"
NORMAL_LINE = "#0000cc"
SEIZURE_FILL = "#ff9c8a"
SEIZURE_LINE = "#e01010"


def plot_pair(sweep_dir, collect_dir):
    """Worker: one histogram figure per sweep pair. Returns (label, err)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fov_rate_map import load_session

    label = os.path.basename(sweep_dir)
    try:
        rates = {}
        for state in ("normal", "seizure"):
            rates[state] = load_session(os.path.join(sweep_dir, state))[2]

        styles = (("normal", NORMAL_FILL, NORMAL_LINE, "normal state"),
                  ("seizure", SEIZURE_FILL, SEIZURE_LINE, "seizure state"))

        def styled(ax):
            ax.set_ylim(bottom=0)
            ax.set_xlabel("firing rate (Hz)", fontsize=11)
            ax.set_ylabel("number of neurons", fontsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=10)
            ax.legend(loc="upper left", frameon=False, fontsize=11,
                      handlelength=1.6)

        # -- linear axis (reference style; shared bins, honest widths) ----
        hi = max(r.max() for r in rates.values()) * 1.10
        bins = np.linspace(0, hi, 100)
        width = bins[1] - bins[0]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
        for state, fill, line, name in styles:
            r = rates[state]
            ax.hist(r, bins=bins, color=fill, alpha=0.65, label=name,
                    zorder=2)
            mu, sigma = r.mean(), r.std()
            x = np.linspace(max(0, mu - 4 * sigma), mu + 4 * sigma, 300)
            pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi))
            ax.plot(x, pdf * len(r) * width, color=line, lw=3, zorder=3,
                    solid_capstyle="round")
        ax.set_xlim(0, hi)
        styled(ax)
        fig.tight_layout()
        out = os.path.join(sweep_dir, "results", "rate_histogram.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        shutil.copyfile(out, os.path.join(collect_dir, f"{label}.png"))

        # -- log axis companion (both humps comparable; lognormal fits) ---
        logr = {s: np.log10(rates[s][rates[s] > 0]) for s in rates}
        lo10 = min(v.min() for v in logr.values()) - 0.08
        hi10 = max(v.max() for v in logr.values()) + 0.08
        lbins = np.linspace(lo10, hi10, 60)
        lwidth = lbins[1] - lbins[0]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
        for state, fill, line, name in styles:
            v = logr[state]
            ax.hist(v, bins=lbins, color=fill, alpha=0.65, label=name,
                    zorder=2)
            mu, sigma = v.mean(), v.std()
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi))
            ax.plot(x, pdf * len(v) * lwidth, color=line, lw=3, zorder=3,
                    solid_capstyle="round")
        ticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        ticks = ticks[(np.log10(ticks) >= lo10) & (np.log10(ticks) <= hi10)]
        ax.set_xticks(np.log10(ticks))
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlim(lo10, hi10)
        styled(ax)
        fig.tight_layout()
        out2 = os.path.join(sweep_dir, "results", "rate_histogram_logx.png")
        fig.savefig(out2, dpi=200, facecolor="white")
        plt.close(fig)
        shutil.copyfile(out2, os.path.join(collect_dir, f"{label}_logx.png"))

        n, s = rates["normal"], rates["seizure"]
        print(f"{label}: normal {n.mean():.3f}±{n.std():.3f} Hz, "
              f"seizure {s.mean():.3f}±{s.std():.3f} Hz  -> {out}")
        return label, None
    except Exception as exc:  # noqa: BLE001 - report and keep the batch going
        return label, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    sweeps = sorted(glob.glob(os.path.join(args.root, "sweep_c[45]0_seed*")))
    sweeps = [s for s in sweeps if os.path.isdir(s)]
    if args.only:
        sweeps = [s for s in sweeps if args.only in os.path.basename(s)]
    if not sweeps:
        raise SystemExit(f"no sweep dirs found under {args.root}")

    collect_dir = os.path.join(args.root, "sweep_summary", "rate_histograms")
    os.makedirs(collect_dir, exist_ok=True)
    print(f"{len(sweeps)} histograms to render ({args.workers} workers)")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(plot_pair, s, collect_dir): s for s in sweeps}
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
