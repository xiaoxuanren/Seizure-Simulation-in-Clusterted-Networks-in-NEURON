"""Re-tuning harness for the Ho gK network -- a figure on every run.

STANDING RULE: no simulation without a diagnostic figure + a CSV row.

    run_and_diagnose(params, label, outdir) -> metrics dict
        Build the network from ``params``, run it (main window + burn-in, burn-in
        excluded from all metrics), compute metrics, and ALWAYS save one 5-panel
        figure ({outdir}/run_{label}.png) and append one row to {outdir}/log.csv.

    make_montage(run_labels, axis_name, outdir) -> path
        Tile the per-run figures of a sweep into one montage image.

Figure panels: (1) raster E=blue/I=red cluster-sorted; (2) population active
fraction + pop rate; (3) per-population mean [K+]o (E blue / I red) with a dashed
baseline reference; (4) Welch PSD 0-10 Hz with the 1-5 Hz ictal band; (5)
histogram of participation-per-10-ms-bin (the twitchiness adjudicator).

CLI (one isolated run, used by the parallel sweep drivers):
    python -m neuron_simulation.tuning <params.json> <label> <outdir>
"""
import os
import sys
import csv
import json
import time
import math

import numpy as np
import matplotlib
if "matplotlib.pyplot" not in sys.modules:      # headless in subprocesses; keep inline in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal as _sig
from scipy import stats as _stats
from scipy.ndimage import gaussian_filter1d
from neuron import h

from .topology import build_topology_lognormal
from .network_builder import build_network
from .simulation import run_simulation

BIN_MS = 10.0                     # population-activity bin
SMOOTH_SIGMA = 0.5                # light (5 ms) smoothing only; larger over-low-passes
                                  # the PSD and falsely inflates the 1-5 Hz fraction
KO_DT = 5.0                       # [K+]o sampling interval (ms)
SYNC_BIN_MS = 1.0                 # fine bin for the spike-density synchrony signal
SYNC_GAUSS_MS = 5.0               # Gaussian smoothing (ms) of each spike train (Ho's Kspike kernel)
KO_DRIFT_FLAG = 0.05              # |ko_drift| (mM/s) above which the reference is "leaking"
PSD_XLIM = (0.0, 10.0)
ICTAL_BAND = (1.0, 5.0)
KO_YLIM = (2.5, 9.0)             # fixed for comparability (rest ~3 -> ictal ~8 mM)

CSV_FIELDS = [
    "label", "gK_exc", "gK_inh", "iext_exc", "iext_inh", "iext_sigma", "noise_rate",
    "noise_weight", "exc_weight_scale", "inh_weight_scale", "tau_k",
    "num_clusters", "seed", "N", "burn_in_ms", "main_ms",
    "exc_rate_hz", "inh_rate_hz", "mean_participation", "pop_cv",
    "peak_freq_hz", "band_power_frac_1_5Hz", "participation_bimodality", "sync_S",
    "ko_base_exc", "ko_base_inh", "ko_drift_exc", "ko_drift_inh", "ko_flag", "figure",
]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _bin_edges(total_ms):
    return np.arange(0.0, total_ms + BIN_MS, BIN_MS)


def _population_series(spike_data, n_neurons, edges):
    """Per-bin population spike count and active-neuron fraction."""
    nb = len(edges) - 1
    popcount = np.zeros(nb)
    active = np.zeros(nb)
    for gid, times in spike_data.items():
        if not len(times):
            continue
        t = np.asarray(times, float)
        idx = np.clip(((t - edges[0]) / BIN_MS).astype(int), 0, nb - 1)
        popcount += np.bincount(idx, minlength=nb)
        active[np.unique(idx)] += 1.0
    return popcount, active / max(1, n_neurons)


def _bimodality(x):
    """Sarle's bimodality coefficient of the per-bin participation distribution.
    ~0.555 = uniform; > ~0.555 -> bimodal (all-or-none); < = unimodal."""
    x = np.asarray(x, float)
    n = x.size
    if n < 8 or np.std(x) < 1e-9:
        return 0.0
    g = _stats.skew(x)
    k = _stats.kurtosis(x, fisher=True)              # excess kurtosis
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = k + corr
    return float((g * g + 1.0) / denom) if denom > 1e-9 else 0.0


def compute_metrics(spike_data, n_neurons, is_inhibitory, burn_ms, total_ms):
    """Spike-based metrics on the post-burn-in window [burn_ms, total_ms]."""
    inh = np.asarray(is_inhibitory, bool)
    win_s = (total_ms - burn_ms) / 1000.0
    exc_c, inh_c = [], []
    for gid in range(n_neurons):
        t = np.asarray(spike_data.get(gid, []), float)
        n = int(np.count_nonzero(t >= burn_ms))
        (inh_c if inh[gid] else exc_c).append(n / win_s)
    exc_rate = float(np.mean(exc_c)) if exc_c else 0.0
    inh_rate = float(np.mean(inh_c)) if inh_c else 0.0

    edges = _bin_edges(total_ms)
    b0 = int(round(burn_ms / BIN_MS))
    popcount_full, active_full = _population_series(spike_data, n_neurons, edges)
    popcount = popcount_full[b0:]
    active = active_full[b0:]

    active_bins = active[active > 0]
    mean_participation = float(active_bins.mean()) if active_bins.size else 0.0
    pm = popcount.mean()
    pop_cv = float(popcount.std() / pm) if pm > 0 else 0.0
    bimod = _bimodality(active)

    peak_freq, band_frac = 0.0, 0.0
    if popcount.size >= 16 and pm > 0:
        sig = gaussian_filter1d(popcount.astype(float), sigma=SMOOTH_SIGMA)
        f, pxx = _sig.welch(sig, fs=1000.0 / BIN_MS, nperseg=int(min(256, len(sig))), detrend="constant")
        pos = f > 0
        tot = float(pxx[pos].sum())
        if tot > 0:
            band = (f >= ICTAL_BAND[0]) & (f <= ICTAL_BAND[1])
            band_frac = float(pxx[band].sum() / tot)
            lo = (f >= 0.5) & (f <= 8.0)
            if lo.any():
                peak_freq = float(f[lo][int(np.argmax(pxx[lo]))])
    return dict(exc_rate_hz=round(exc_rate, 3), inh_rate_hz=round(inh_rate, 3),
                mean_participation=round(mean_participation, 4), pop_cv=round(pop_cv, 3),
                peak_freq_hz=round(peak_freq, 3), band_power_frac_1_5Hz=round(band_frac, 4),
                participation_bimodality=round(bimod, 3))


def compute_sync(spike_data, is_inhibitory, burn_ms, total_ms):
    """Ho variance-ratio synchrony S over the PYRAMIDAL (excitatory) population.

    Smooth each PY cell's spike train with a short Gaussian (Ho's Kspike kernel),
    form the population-mean signal, then
        S = Var_t(<smoothed>(t)) / mean_i(Var_t(smoothed_i(t)))    (ratio, not sqrt)
    over the post-burn-in window. S in [0,1]: ~0 asynchronous, -> 1 synchronized.
    """
    inh = np.asarray(is_inhibitory, bool)
    py = np.where(~inh)[0]
    nb = int(round((total_ms - burn_ms) / SYNC_BIN_MS))
    if py.size < 2 or nb < 8:
        return 0.0
    sig = SYNC_GAUSS_MS / SYNC_BIN_MS
    sm = np.zeros((py.size, nb))
    for k, g in enumerate(py):
        t = np.asarray(spike_data.get(int(g), []), float)
        t = t[t >= burn_ms]
        if t.size:
            idx = np.clip(((t - burn_ms) / SYNC_BIN_MS).astype(int), 0, nb - 1)
            sm[k] = gaussian_filter1d(np.bincount(idx, minlength=nb).astype(float), sigma=sig)
    var_pop = float(sm.mean(axis=0).var())
    mean_var_i = float(sm.var(axis=1).mean())
    return float(var_pop / mean_var_i) if mean_var_i > 0 else 0.0


def compute_ko(ko_arr, ko_t, is_inhibitory, burn_ms):
    """Per-population mean [K+]o series + baseline level and least-squares drift (mM/s)."""
    inh = np.asarray(is_inhibitory, bool)
    m = ko_t >= burn_ms
    tsec = (ko_t[m] - ko_t[m][0]) / 1000.0 if m.any() else np.array([0.0])
    out, series = {}, {}
    for tag, sel in (("exc", ~inh), ("inh", inh)):
        if sel.sum() == 0 or ko_arr.size == 0:
            out[f"ko_base_{tag}"] = float("nan"); out[f"ko_drift_{tag}"] = float("nan")
            series[tag] = None; continue
        mean_ko = ko_arr[sel].mean(0)
        seg = mean_ko[m]
        out[f"ko_base_{tag}"] = round(float(seg.mean()), 3)
        slope = float(np.polyfit(tsec, seg, 1)[0]) if tsec.size >= 2 and np.ptp(tsec) > 0 else 0.0
        out[f"ko_drift_{tag}"] = round(slope, 4)
        series[tag] = mean_ko
    return out, series


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def _diagnostic_figure(spike_data, n_neurons, is_inhibitory, cluster_assignments,
                       burn_ms, total_ms, params, label, metrics, part_thresh,
                       ko_t, ko_series, ko_ref):
    inh = np.asarray(is_inhibitory, bool)
    ca = np.asarray(cluster_assignments, int)
    order = np.argsort(ca, kind="stable")
    row_of = np.empty(n_neurons, int)
    row_of[order] = np.arange(n_neurons)

    ex_t, ex_y, in_t, in_y = [], [], [], []
    for gid, times in spike_data.items():
        if not len(times):
            continue
        y = row_of[gid]
        (in_t if inh[gid] else ex_t).extend(times)
        (in_y if inh[gid] else ex_y).extend([y] * len(times))

    edges = _bin_edges(total_ms)
    centers = edges[:-1] + BIN_MS / 2.0
    popcount, active = _population_series(spike_data, n_neurons, edges)
    poprate = popcount / max(1, n_neurons) / (BIN_MS / 1000.0)
    b0 = int(round(burn_ms / BIN_MS))

    ko_flag = max(abs(metrics.get("ko_drift_exc", 0) or 0),
                  abs(metrics.get("ko_drift_inh", 0) or 0)) > KO_DRIFT_FLAG

    fig = plt.figure(figsize=(10, 11.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[3.0, 1.3, 1.3, 1.4], hspace=0.45, wspace=0.25)
    axr = fig.add_subplot(gs[0, :])
    axp = fig.add_subplot(gs[1, :])
    axk = fig.add_subplot(gs[2, :])
    axf = fig.add_subplot(gs[3, 0])
    axh = fig.add_subplot(gs[3, 1])

    # (1) raster
    axr.scatter(ex_t, ex_y, s=1.5, c="tab:blue", marker="|", linewidths=0.5, rasterized=True)
    axr.scatter(in_t, in_y, s=1.5, c="tab:red", marker="|", linewidths=0.5, rasterized=True)
    axr.axvline(burn_ms, color="green", ls="--", lw=1.0)
    axr.set(xlim=(0, total_ms), ylim=(-1, n_neurons), ylabel="neuron (cluster-sorted)")
    axr.set_title("raster (E=blue, I=red)", fontsize=9)

    # (2) active fraction + pop rate
    axp.fill_between(centers, active, step="mid", color="0.4", alpha=0.6)
    axp.axhline(part_thresh, color="k", ls=":", lw=0.8)
    axp.axvline(burn_ms, color="green", ls="--", lw=1.0)
    axp.set(xlim=(0, total_ms), ylim=(0, 1), ylabel="active frac")
    axp2 = axp.twinx()
    axp2.plot(centers, poprate, color="tab:purple", lw=0.6, alpha=0.8)
    axp2.set_ylabel("pop rate\n(Hz/cell)", color="tab:purple", fontsize=8)

    # (3) per-population [K+]o
    if ko_series.get("exc") is not None:
        axk.plot(ko_t, ko_series["exc"], color="tab:blue", lw=1.0, label="E")
    if ko_series.get("inh") is not None:
        axk.plot(ko_t, ko_series["inh"], color="tab:red", lw=1.0, label="I")
    axk.axhline(ko_ref, color="0.5", ls="--", lw=0.8)
    axk.axvline(burn_ms, color="green", ls="--", lw=1.0)
    axk.set(xlim=(0, total_ms), ylim=KO_YLIM, ylabel="[K+]o (mM)")
    axk.legend(loc="upper left", fontsize=8)
    axk.set_title(f"[K+]o  drift E={metrics.get('ko_drift_exc')} I={metrics.get('ko_drift_inh')} mM/s"
                  + ("   ⚠ DRIFT" if ko_flag else "   (flat)"),
                  fontsize=9, color=("red" if ko_flag else "black"))

    # (4) PSD
    pc = popcount[b0:]
    if pc.size >= 16 and pc.mean() > 0:
        s = gaussian_filter1d(pc.astype(float), sigma=SMOOTH_SIGMA)
        f, pxx = _sig.welch(s, fs=1000.0 / BIN_MS, nperseg=int(min(256, len(s))), detrend="constant")
        tot = pxx[f > 0].sum()
        axf.plot(f, pxx / tot if tot > 0 else pxx, color="tab:orange", lw=1.2)
        axf.axvspan(*ICTAL_BAND, color="orange", alpha=0.15)
    axf.set(xlim=PSD_XLIM, xlabel="freq (Hz)", ylabel="norm PSD")
    axf.set_title(f"PSD  peak {metrics['peak_freq_hz']:.2f}Hz  band15 {metrics['band_power_frac_1_5Hz']:.2f}",
                  fontsize=8)

    # (5) participation-per-bin histogram (twitchiness adjudicator)
    ph = active[b0:]
    bimod = metrics.get("participation_bimodality", 0.0)
    axh.hist(ph, bins=np.linspace(0, 1, 21), color=("tab:red" if bimod > 0.555 else "tab:green"), alpha=0.8)
    axh.axvline(part_thresh, color="k", ls=":", lw=0.8)
    axh.set(xlim=(0, 1), xlabel="participation / 10ms bin", ylabel="count")
    axh.set_yscale("log")
    axh.set_title(f"bimodality {bimod:.2f} " + ("(BIMODAL)" if bimod > 0.555 else "(unimodal)"),
                  fontsize=8, color=("red" if bimod > 0.555 else "black"))

    b = params.get("build", {})
    fig.suptitle(
        f"{label}"
        + ("   ⚠ KO DRIFT" if ko_flag else "")
        + f"\ngK=({b.get('gK_exc')},{b.get('gK_inh')}) iext=({b.get('iext_exc')},{b.get('iext_inh')})"
        f" sig={b.get('iext_sigma', 0)} noise={b.get('noise_rate')}Hz/{b.get('noise_weight')}"
        f" g_ee={b.get('exc_weight_scale')} g_ie={b.get('inh_weight_scale')} tau_k={b.get('tau_k')}\n"
        f"S(sync) {metrics.get('sync_S', 0):.3f} | "
        f"exc {metrics['exc_rate_hz']:.1f}Hz | inh {metrics['inh_rate_hz']:.1f}Hz | "
        f"part {metrics['mean_participation']:.2f} | CV {metrics['pop_cv']:.2f} | "
        f"bimod {metrics['participation_bimodality']:.2f} | "
        f"fpeak {metrics['peak_freq_hz']:.2f}Hz | band15 {metrics['band_power_frac_1_5Hz']:.2f}",
        fontsize=9)
    return fig


# --------------------------------------------------------------------------- #
# CSV (cross-process-safe append)
# --------------------------------------------------------------------------- #
def _append_row(csv_path, row):
    lock = csv_path + ".lock"
    for _ in range(3000):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.01)
    try:
        new = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
def _topology_from_cfg(topo_cfg):
    """Build a topology, reconstructing weight_params from a plain dict if given."""
    cfg = dict(topo_cfg)
    wpd = cfg.pop("weight_params", None)
    if isinstance(wpd, dict):
        from .topology import NeuronWeightParameters
        wp = NeuronWeightParameters()
        for k, v in wpd.items():
            setattr(wp, k, tuple(v) if isinstance(v, list) else v)
        cfg["weight_params"] = wp
    return build_topology_lognormal(**cfg)


def run_and_diagnose(params, label, outdir):
    os.makedirs(outdir, exist_ok=True)
    topo = _topology_from_cfg(params["topology"])
    build_kw = dict(params["build"])
    build_kw.setdefault("noise_seed", params.get("noise_seed", 1))
    burn = float(params.get("burn_in_ms", 500.0))
    main = float(params.get("main_ms", 3000.0))
    dt = float(params.get("dt", 0.025))
    part_thresh = float(params.get("participation_thresh", 0.2))
    total = burn + main
    inh = np.asarray(topo["neuron_is_inhibitory"], bool)

    net = build_network(topo, **build_kw)
    ko_vecs = [h.Vector().record(c.soma(0.5).kdyn._ref_ko, KO_DT) for c in net.cells]
    spikes, _v, _ko = run_simulation(net, duration=total, dt=dt,
                                     discard_transient_ms=0.0, record_ko=False)
    N = net.n_neurons
    ko_arr = np.array([np.asarray(v) for v in ko_vecs]) if ko_vecs else np.empty((0, 0))
    ko_t = np.arange(ko_arr.shape[1]) * KO_DT if ko_arr.size else np.array([0.0])
    ko_ref = float(ko_arr[:, 0].mean()) if ko_arr.size else 3.0

    metrics = compute_metrics(spikes, N, inh, burn, total)
    metrics["sync_S"] = round(compute_sync(spikes, inh, burn, total), 4)
    ko_metrics, ko_series = compute_ko(ko_arr, ko_t, inh, burn)
    metrics.update(ko_metrics)
    ko_flag = max(abs(ko_metrics["ko_drift_exc"] or 0) if not np.isnan(ko_metrics["ko_drift_exc"]) else 0,
                  abs(ko_metrics["ko_drift_inh"] or 0) if not np.isnan(ko_metrics["ko_drift_inh"]) else 0) > KO_DRIFT_FLAG

    fig = _diagnostic_figure(spikes, N, inh, topo["cluster_assignments"], burn, total,
                             params, label, metrics, part_thresh, ko_t, ko_series, ko_ref)
    figpath = os.path.join(outdir, f"run_{label}.png")
    fig.savefig(figpath, dpi=105, facecolor="white")
    plt.close(fig)

    row = {"label": label, "N": N, "num_clusters": params["topology"].get("num_clusters"),
           "seed": params["topology"].get("seed"), "burn_in_ms": burn, "main_ms": main,
           "ko_flag": int(ko_flag), "figure": os.path.basename(figpath)}
    for k in ("gK_exc", "gK_inh", "iext_exc", "iext_inh", "iext_sigma", "noise_rate",
              "noise_weight", "exc_weight_scale", "inh_weight_scale", "tau_k"):
        row[k] = build_kw.get(k)
    row.update(metrics)
    _append_row(os.path.join(outdir, "log.csv"), row)
    metrics["_figure"] = figpath
    return metrics


def make_montage(run_labels, axis_name, outdir):
    paths = [os.path.join(outdir, f"run_{l}.png") for l in run_labels]
    pairs = [(l, p) for l, p in zip(run_labels, paths) if os.path.exists(p)]
    if not pairs:
        raise FileNotFoundError(f"no run figures found in {outdir} for {run_labels}")
    n = len(pairs)
    ncol = int(math.ceil(math.sqrt(n)))
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.0, nrow * 4.6))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (lab, p) in zip(axes, pairs):
        ax.imshow(mpimg.imread(p))
        ax.set_title(lab, fontsize=6)
    fig.suptitle(f"sweep over: {axis_name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(outdir, f"montage_{axis_name}.png")
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    _pj, _label, _outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(_pj, encoding="utf-8") as _f:
        _params = json.load(_f)
    _m = run_and_diagnose(_params, _label, _outdir)
    print(json.dumps({k: v for k, v in _m.items() if not k.startswith("_")}))
