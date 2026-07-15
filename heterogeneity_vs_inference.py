"""A/B(/C) test: does bioplausible heterogeneity + [K+]o afterdischarge change
connectivity-inference difficulty?

ONE network (fixed topology = fixed ground truth), simulated three ways in the
SAME pharmacological state, then run through the SAME lag-1 GLM + spike-jitter
FDR pipeline. The only thing that differs between conditions is the intrinsic
dynamics, so any change in AUC/AP/FDR is attributable to the dynamics, not the
wiring.

  A  homogeneous              (current model: identical intrinsic params)
  B  heterogeneous            (per-cell sAHP increments, mean-preserving spread)
  C  heterogeneous + afterdischarge   (B, plus slower [K+]o clearance -> tail)

Reports, per condition:
  - dynamics:  mean rate, population Fano (burstiness), #network bursts, %spikes-in-burst
  - ranking:   exc/inh AUC and AP (whole-map)
  - operating point (LABEL-FREE): recovered-wiring TP/FP/FN/precision/recall/F1
  - REALIZED FDR at the target  <- the key check: does the jitter null still calibrate
                                    once the tail spikes are present?

Run from the repo root (NEURON + compiled mechanisms required):
    cd neuron_simulation && nrnivmodl mechanisms       # once
    python heterogeneity_vs_inference.py
"""
from __future__ import annotations
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo import (NEURON-backed); adjust if you run from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neuron_simulation import topology, states, build_network, run_simulation  # noqa: E402

# ----------------------------------------------------------------------------- #
# knobs
# ----------------------------------------------------------------------------- #
CONFIG = {
    # --- synced from the notebook's CONFIG['topology'] (single-knob, seed 1).
    #     weight_params omitted: the notebook's wp == NeuronWeightParameters()
    #     defaults, which build_topology_lognormal builds when weight_params=None.
    "topology": dict(num_clusters=40, neurons_per_cluster_range=(4, 40),
        inhibitory_probability=0.2, cluster_radius=1.0, space_size=15.0, seed=1,
        decay_sigma=3.0, max_connection_distance=6.0, cell_type_specific=True,
        p_ee_within=0.2, p_ee_between=0.06, p_ei_within=0.30, p_ei_between=0.01,
        p_ie_within=0.40, p_ii_within=0.50,
        within_cluster_prob=0.25, between_cluster_prob=0.06,
        ln_sigma=0.5, target_density=None),
    "build": dict(synapse_model="ampa_nmda", exc_tau=5.0, tau_nmda=350.0,
        nmda_ratio=3.0, exc_weight_scale=2.0, inh_weight_scale=2.5,   # exc_weight_scale synced from notebook (was 3.0)
        depression_d=0.2, tau_d=500.0, noise_rate=5.0, noise_weight=0.004,   # noise_rate synced from notebook (was 18.0)
        adapt=True, sahp_ainc_fast=0.005, sahp_tau_fast=300.0,        # sahp_ainc_fast NOT synced (study knob; heterogeneity jitters it)
        sahp_ainc_slow=0.009, sahp_tau_slow=6500.0, delay_per_distance=2.0),   # sahp_ainc_slow NOT synced (set per-state via STATE_SAHP_SLOW); tau_k NOT synced (afterdischarge knob)
    "sim": dict(dt=0.05, discard_transient_ms=1000.0),
}
STATE_SAHP_SLOW = 0.009          # 0.009 normal (matches your rasters) / 0.003 seizure
N_RECORDINGS   = 1               # (was 3) one recording per condition
DURATION_MS    = 20000.0         # (was 60000.0) 20 s per recording
BIN_MS         = 5.0
TARGET_FDR     = 0.10
JITTER_MS      = 25.0
N_SURROGATES   = 8
HETERO_SIGMA   = 0.30            # lognormal spread on sAHP increments (CV ~0.30; in 0.2-0.5)
TAU_K_AFTER_MS = 1000.0          # afterdischarge clearance (200ms default -> ~1s, still physiological)
HETERO_SEED    = 7               # fixed: heterogeneity is a property of the NETWORK, not the recording
NOISE_SEED0    = 1000
CONDITIONS = ["homogeneous", "heterogeneous", "heterogeneous+afterdischarge"]

# ----------------------------------------------------------------------------- #
# embedded inference (lag-1 GLM + jitter-null FDR; no torch)
# ----------------------------------------------------------------------------- #
def build_spike_matrix(spike_lists, n, bin_ms):
    T = int(DURATION_MS / bin_ms)
    M = np.zeros((n, T), np.float32)
    for i in range(n):
        t = np.atleast_1d(np.asarray(spike_lists[i], float))
        if len(t):
            M[i, np.clip((t / bin_ms).astype(int), 0, T - 1)] = 1.0
    return M

def _lag1(M, bnd, max_lag=4, l2=2.0):
    n = M.shape[0]
    F = np.zeros((max_lag * n, M.shape[1]), np.float32)
    for k in range(max_lag):
        lag = k + 1; sh = F[k * n:(k + 1) * n]
        sh[:, lag:] = M[:, :-lag]
        for b in bnd[1:-1]: sh[:, b:b + lag] = 0.0
    B = np.linalg.solve(F @ F.T + l2 * np.eye(F.shape[0]), F @ M.T.astype(np.float32))
    W = B.reshape(max_lag, n, n)[0].copy(); np.fill_diagonal(W, 0.0); return W

def _jitter(M, bnd, jbins, rng):
    out = np.zeros_like(M)
    for a, b in zip(bnd[:-1], bnd[1:]):
        L = b - a
        for i in range(M.shape[0]):
            idx = np.flatnonzero(M[i, a:b])
            if len(idx):
                out[i, a + np.clip(idx + rng.integers(-jbins, jbins + 1, len(idx)), 0, L - 1)] = 1.0
    return out

def _fdr_threshold(obs, null, target):
    cand = np.unique(obs); os_ = np.sort(obs); nz = np.sort(null.ravel())
    no = len(os_) - np.searchsorted(os_, cand, "left")
    nn = (len(nz) - np.searchsorted(nz, cand, "left")) / null.shape[0]
    fdr = np.where(no > 0, nn / np.maximum(no, 1), np.inf)
    ok = np.where(fdr <= target)[0]
    return float(cand[ok[0]]) if len(ok) else np.inf

def jitter_threshold(M, bnd, W, cand, sign, target, jbins, nsurr, seed=1):
    rng = np.random.default_rng(seed)
    obs = (sign * W)[cand]
    null = np.stack([(sign * _lag1(_jitter(M, bnd, jbins, rng), bnd))[cand] for _ in range(nsurr)])
    thr = _fdr_threshold(obs, null, target)
    pred = (sign * W >= thr) & cand if np.isfinite(thr) else np.zeros_like(cand)
    return thr, pred

# ----------------------------------------------------------------------------- #
# network-spike detector (cluster-based, your verified definition) + dynamics
# ----------------------------------------------------------------------------- #
def detect_network_spikes(spike_lists, clusters, max_isi_ms=100., cluster_frac=0.35, min_spikes=50):
    ncl = int(clusters.max()) + 1
    T, C = [], []
    for i, st in enumerate(spike_lists):
        t = np.atleast_1d(np.asarray(st, float)); T.append(t); C.append(np.full(len(t), clusters[i]))
    T = np.concatenate(T); C = np.concatenate(C)
    if len(T) == 0: return []
    o = np.argsort(T, kind="mergesort"); T, C = T[o], C[o]
    br = np.flatnonzero(np.diff(T) >= max_isi_ms)
    starts, ends = np.r_[0, br + 1], np.r_[br, len(T) - 1]; need = cluster_frac * ncl
    ev = []
    for s, e in zip(starts, ends):
        nsp = e - s + 1; nc = len(np.unique(C[s:e + 1]))
        if nsp > min_spikes and nc > need: ev.append((float(T[s]), float(T[e]), int(nsp)))
    return ev

def dynamics_metrics(spike_lists, clusters, n):
    total = sum(len(np.atleast_1d(x)) for x in spike_lists)
    rate = total / n / (DURATION_MS / 1000.0)
    M = build_spike_matrix(spike_lists, n, BIN_MS); pop = M.sum(0)
    fano = pop.var() / pop.mean() if pop.mean() > 0 else 0.0
    ev = detect_network_spikes(spike_lists, clusters)
    allt = np.concatenate([np.atleast_1d(x) for x in spike_lists if len(np.atleast_1d(x))]) if total else np.array([])
    inb = 0
    for (t0, t1, _) in ev: inb += int(((allt >= t0) & (allt <= t1)).sum())
    return dict(rate=rate, fano=fano, n_bursts=len(ev),
                burst_per_min=len(ev) / (DURATION_MS / 60000.0),
                frac_in_burst=(inb / total if total else 0.0))

# ----------------------------------------------------------------------------- #
# ground truth from the topology
# ----------------------------------------------------------------------------- #
def ground_truth(topo):
    n = topo["n_neurons"]; inh = topo["neuron_is_inhibitory"].astype(bool)
    Ae = np.zeros((n, n), bool); Ai = np.zeros((n, n), bool)
    for r in topo["connections"]:
        pre, post = int(r[0]), int(r[1])
        (Ai if (str(r[3]) == "inh" or inh[pre]) else Ae)[pre, post] = True
    return Ae, Ai, inh

# ----------------------------------------------------------------------------- #
# build + (optional) inject heterogeneity + run
# ----------------------------------------------------------------------------- #
def inject_heterogeneity(network, sigma, seed):
    """Mean-preserving per-cell spread on sAHP increments (CV ~ sigma)."""
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma ** 2  # so E[lognormal(mu,sigma)] = 1
    for cell in network.cells:
        if getattr(cell, "sahp", None) is not None:
            cell.sahp.ainc_fast = float(cell.sahp.ainc_fast * rng.lognormal(mu, sigma))
            cell.sahp.ainc_slow = float(cell.sahp.ainc_slow * rng.lognormal(mu, sigma))

def set_tau_k(network, tau_k_ms):
    for cell in network.cells:
        try:
            cell.soma(0.5).kdyn.tau_k = float(tau_k_ms)
        except AttributeError:
            pass

def simulate(topo, condition, noise_seed):
    bk = dict(CONFIG["build"]); bk["sahp_ainc_slow"] = STATE_SAHP_SLOW; bk["noise_seed"] = noise_seed
    for k in ("gbar_kA_exc", "gbar_kA_inh", "tau_k"):        # inherit normal_state defaults
        v = states.normal_state().get(k)
        if v is not None: bk[k] = v
    net = build_network(topo, **bk)
    if condition in ("heterogeneous", "heterogeneous+afterdischarge"):
        inject_heterogeneity(net, HETERO_SIGMA, HETERO_SEED)
    if condition == "heterogeneous+afterdischarge":
        set_tau_k(net, TAU_K_AFTER_MS)
    spikes, _, _ = run_simulation(net, duration=DURATION_MS, record_ko=False, **CONFIG["sim"])
    return spikes

# ----------------------------------------------------------------------------- #
# run
# ----------------------------------------------------------------------------- #
def run():
    topo = topology.build_topology_lognormal(**CONFIG["topology"])
    n = topo["n_neurons"]; clusters = topo["cluster_assignments"]
    Ae, Ai, inh = ground_truth(topo)
    off = ~np.eye(n, dtype=bool); allE, allI, allEdge = Ae & off, Ai & off, (Ae | Ai) & off
    from sklearn.metrics import roc_auc_score, average_precision_score

    results = {}; first_rec_spikes = {}
    for cond in CONDITIONS:
        print(f"\n=== {cond} ===")
        mats = []; dyn_accum = []
        for r in range(N_RECORDINGS):
            sp = simulate(topo, cond, NOISE_SEED0 + r)
            if r == 0: first_rec_spikes[cond] = sp
            mats.append(build_spike_matrix(sp, n, BIN_MS))
            dyn_accum.append(dynamics_metrics(sp, clusters, n))
        M = np.concatenate(mats, 1)
        bnd = [0]
        for m in mats: bnd.append(bnd[-1] + m.shape[1])
        dyn = {k: float(np.mean([d[k] for d in dyn_accum])) for k in dyn_accum[0]}

        W = _lag1(M, bnd)
        inferred = W.sum(1) < 0
        preI = np.zeros((n, n), bool); preI[np.where(inferred)[0], :] = True
        jbins = max(1, int(round(JITTER_MS / BIN_MS)))
        te, Pe = jitter_threshold(M, bnd, W, off, +1, TARGET_FDR, jbins, N_SURROGATES)
        ti, Pi = jitter_threshold(M, bnd, W, off & preI, -1, TARGET_FDR, jbins, N_SURROGATES)
        Pall = Pe | Pi

        def conf(pred, truth):
            TP = int((pred & truth).sum()); FP = int((pred & ~truth & off).sum()); FN = int((truth & ~pred).sum())
            P = TP / (TP + FP) if TP + FP else 0.0; R = TP / (TP + FN) if TP + FN else 0.0
            return TP, FP, FN, P, R, (2 * P * R / (P + R) if P + R else 0.0)
        ce, ci, ca = conf(Pe, allE), conf(Pi, allI), conf(Pall, allEdge)

        results[cond] = dict(
            **dyn,
            exc_auc=roc_auc_score(Ae[off], W[off]), exc_ap=average_precision_score(Ae[off], W[off]),
            inh_auc=roc_auc_score(Ai[off], -W[off]), inh_ap=average_precision_score(Ai[off], -W[off]),
            type_auc=roc_auc_score(inh, -W.sum(1)),
            exc_realized_fdr=1 - ce[3], inh_realized_fdr=1 - ci[3],
            all_TP=ca[0], all_FP=ca[1], all_FN=ca[2], all_P=ca[3], all_R=ca[4], all_F1=ca[5],
            n_true=int(allEdge.sum()),
        )
        d = results[cond]
        print(f"  dynamics: rate {d['rate']:.2f}Hz  Fano {d['fano']:.1f}  bursts {d['burst_per_min']:.1f}/min  "
              f"{d['frac_in_burst']*100:.0f}% in-burst")
        print(f"  ranking:  exc AUC {d['exc_auc']:.3f} AP {d['exc_ap']:.3f} | inh AUC {d['inh_auc']:.3f} AP {d['inh_ap']:.3f}")
        print(f"  wiring @FDR{TARGET_FDR}: TP {d['all_TP']}/{d['n_true']} FP {d['all_FP']} FN {d['all_FN']} "
              f"(P {d['all_P']:.2f} R {d['all_R']:.2f} F1 {d['all_F1']:.2f}) | realized FDR exc {d['exc_realized_fdr']:.2f} inh {d['inh_realized_fdr']:.2f}")

    # ---- comparison table (printed + csv) ----
    cols = ["rate", "fano", "burst_per_min", "frac_in_burst", "exc_auc", "exc_ap",
            "inh_auc", "inh_ap", "type_auc", "all_TP", "all_FP", "all_FN", "all_F1",
            "exc_realized_fdr", "inh_realized_fdr"]
    print("\n================ SUMMARY (target FDR = %.2f) ================" % TARGET_FDR)
    print("metric".ljust(18) + "".join(c[:15].rjust(16) for c in CONDITIONS))
    for k in cols:
        print(k.ljust(18) + "".join(f"{results[c][k]:16.3f}" for c in CONDITIONS))
    with open("heterogeneity_vs_inference.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric"] + CONDITIONS)
        for k in cols: w.writerow([k] + [results[c][k] for c in CONDITIONS])

    # ---- figure: rasters (random order, bursts shaded) + metric bars ----
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1.6], hspace=0.45, wspace=0.28)
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    for j, cond in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[0 if j < 2 else 1, j % 3] if False else gs[0, j])
        sp = first_rec_spikes[cond]
        for i in range(n):
            t = np.atleast_1d(sp[i]) / 1000.0; m = t <= 20
            if m.any(): ax.plot(t[m], np.full(m.sum(), int(np.flatnonzero(perm == i)[0])),
                                "|", ms=2, mew=0.5, color=("#1b6b3a" if inh[i] else "#333"), alpha=0.8)
        for (t0, t1, _) in detect_network_spikes(sp, clusters):
            if t0 / 1000 < 20: ax.axvspan(t0 / 1000, t1 / 1000, color="#c0392b", alpha=0.13)
        ax.set_title(cond, fontsize=9); ax.set_xlim(0, 20); ax.set_ylabel("neuron (random)" if j == 0 else "")
        ax.set_xlabel("time (s)")
    x = np.arange(len(CONDITIONS)); w = 0.8
    def barpanel(ax, keys, title, ylabel, hlines=None):
        nk = len(keys); ww = w / nk
        for ki, k in enumerate(keys):
            ax.bar(x + (ki - (nk - 1) / 2) * ww, [results[c][k] for c in CONDITIONS], ww, label=k)
        if hlines:
            for hv, hl in hlines: ax.axhline(hv, ls="--", color="k", lw=1, label=hl)
        ax.set_xticks(x); ax.set_xticklabels([c.replace("+", "\n+") for c in CONDITIONS], fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylabel); ax.legend(fontsize=7)
    barpanel(fig.add_subplot(gs[2, 0]), ["exc_auc", "exc_ap", "inh_auc", "inh_ap"], "ranking", "AUC / AP")
    barpanel(fig.add_subplot(gs[2, 1]), ["exc_realized_fdr", "inh_realized_fdr"], "realized FDR vs target",
             "realized FDR", hlines=[(TARGET_FDR, f"target {TARGET_FDR}")])
    barpanel(fig.add_subplot(gs[2, 2]), ["all_R", "all_P", "all_F1"], "recovered wiring", "score")
    fig.suptitle(f"Homogeneous vs heterogeneous(+afterdischarge) — same network (seed {CONFIG['topology']['seed']}), "
                 f"same lag-1+jitter pipeline, {N_RECORDINGS}x{DURATION_MS/1000:.0f}s", fontsize=11)
    fig.savefig("heterogeneity_vs_inference.png", dpi=120, bbox_inches="tight")
    print("\nsaved heterogeneity_vs_inference.png / .csv")

if __name__ == "__main__":
    run()
