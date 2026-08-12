"""Voltage-trace figures for recording000 of the normal 100-rec flagship.

Fig 1: example Vm during a network burst vs outside (inter-burst) periods.
Fig 2: connected-neuron traces + spike-triggered average (pre-spike -> post EPSP
       for excitatory pairs, IPSP for inhibitory pairs, flat for unconnected).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_paths import resolve, results_dir  # noqa: E402
_S = os.environ.get("DATASET_SESSION", "IC-locked_flagship_200rec")
_T = os.environ.get("DATASET_STATE", "normal")
SD = resolve(_S, _T)
FIGDIR = results_dir(_S, _T, "figures")

d = np.load(os.path.join(SD, "recording000.npz"), allow_pickle=True)
V = np.asarray(d["voltage_traces"], np.float32)          # [N, T] mV, 1 ms bins
N, T = V.shape
st = d["spike_times"]                                     # per-neuron ms
bw = np.asarray(d["burst_windows"], float)               # [[lo,hi], ...]
burst = bw[0]                                             # the network burst
def sp(i): return np.atleast_1d(np.asarray(st[i], float))

net = np.load([os.path.join(SD, f) for f in os.listdir(SD) if f.startswith("network_") and f.endswith(".npz")][0], allow_pickle=True)
conns = net["connections"]
is_inh = net["neuron_is_inhibitory"].astype(bool)
exc_pairs = [(int(r[0]), int(r[1])) for r in conns if str(r[3]) == "exc"]
inh_pairs = [(int(r[0]), int(r[1])) for r in conns if str(r[3]) == "inh"]

# ---- choose example neurons: active in the burst, mix exc/inh ----
counts = np.array([len(sp(i)) for i in range(N)])
in_burst = np.array([np.any((sp(i) >= burst[0]) & (sp(i) < burst[1])) for i in range(N)])
exc_ids = [i for i in np.argsort(-counts) if in_burst[i] and not is_inh[i]][:4]
inh_ids = [i for i in np.argsort(-counts) if in_burst[i] and is_inh[i]][:1]
sel = exc_ids + inh_ids
labels = ["exc %d" % i for i in exc_ids] + ["inh %d" % i for i in inh_ids]
COL = ["#1f5fd0", "#2e8b57", "#7d3c98", "#008b8b", "#c0392b"]

def plot_stack(ax, t0, t1, title):
    lo, hi = int(t0), int(t1)
    x = np.arange(lo, hi)
    offset = 130.0
    for k, i in enumerate(sel):
        ax.plot(x, V[i, lo:hi] + k * offset, color=COL[k], lw=0.7)
        ax.text(lo, k * offset + 25, labels[k], color=COL[k], fontsize=8, va="bottom")
    if lo <= burst[1] and hi >= burst[0]:
        ax.axvspan(max(lo, burst[0]), min(hi, burst[1]), color="orange", alpha=0.15, label="network burst")
    ax.set_title(title); ax.set_xlabel("time (ms)"); ax.set_yticks([])
    ax.set_xlim(lo, hi)

# ================= FIGURE 1 =================
fig1, ax = plt.subplots(1, 3, figsize=(18, 7),
                        gridspec_kw={"width_ratios": [2.4, 1, 1]})
plot_stack(ax[0], burst[0] - 2000, burst[1] + 3500, "(a) overview: 5 neurons across a network burst (shaded)")
ax[0].legend(loc="upper right", fontsize=8)
plot_stack(ax[1], burst[0] - 40, burst[1] + 60, "(b) DURING burst (zoom)")
# pick a quiet inter-burst window far from the burst
qz = 20000
plot_stack(ax[2], qz, qz + 400, "(c) OUTSIDE burst (inter-burst zoom)")
fig1.suptitle("Membrane potential during vs outside a network burst \u2014 recording000, normal flagship\n"
              "(each trace offset for clarity; units mV)", fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.95])
f1 = os.path.join(FIGDIR, "voltage_burst_vs_interburst.png")
fig1.savefig(f1, dpi=130, facecolor="white", bbox_inches="tight"); print("fig1 ->", f1)

# ================= FIGURE 2: connected neurons =================
# spike-triggered average of POST Vm aligned to PRE spikes (inter-burst spikes only)
LAG_LO, LAG_HI = -10, 40                                  # ms around pre spike
lags = np.arange(LAG_LO, LAG_HI)
margin = 150                                              # exclude spikes near burst
def interburst_spikes(i):
    s = sp(i)
    return s[(s > LAG_HI + 5) & (s < T - LAG_HI - 5) & ~((s > burst[0] - margin) & (s < burst[1] + margin))]

def sta(pairs, max_pairs=400, max_snips=25000):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))[:max_pairs]
    acc = []
    for j in idx:
        pre, post = pairs[j]
        ts = interburst_spikes(pre)
        if len(ts) == 0:
            continue
        for t in ts.astype(int):
            snip = V[post, t + LAG_LO:t + LAG_HI].astype(float)
            if len(snip) == len(lags):
                snip = snip - snip[:abs(LAG_LO) - 1].mean()   # baseline = pre-spike window
                acc.append(snip)
        if len(acc) >= max_snips:
            break
    A = np.array(acc)
    return A.mean(0), A.std(0) / max(np.sqrt(len(A)), 1), len(A)

# unconnected control pairs (random, not in the true adjacency)
trueset = set(exc_pairs) | set(inh_pairs)
rng = np.random.default_rng(2)
ctrl = []
while len(ctrl) < 400:
    a, b = int(rng.integers(N)), int(rng.integers(N))
    if a != b and (a, b) not in trueset:
        ctrl.append((a, b))

sta_e, se_e, ne = sta(exc_pairs)
sta_i, se_i, ni = sta(inh_pairs)
sta_c, se_c, nc = sta(ctrl)
print("STA snippets: exc=%d inh=%d ctrl=%d" % (ne, ni, nc))

# a raw connected-pair example: exc pair whose pre fires in a quiet window
example = None
for pre, post in exc_pairs:
    s = interburst_spikes(pre)
    if len(s) >= 4:
        w0 = int(s[len(s) // 3]) - 200
        nsp = np.sum((s >= w0) & (s < w0 + 1500))
        if nsp >= 3:
            example = (pre, post, w0, w0 + 1500, s)
            break

fig2, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})

if example:
    pre, post, w0, w1, s = example
    xp = np.arange(w0, w1)
    ax[0].plot(xp, V[pre, w0:w1] + 130, color="#1f5fd0", lw=0.8)
    ax[0].plot(xp, V[post, w0:w1], color="#2e8b57", lw=0.8)
    ax[0].text(w0, 130 + 25, "pre  (exc %d)" % pre, color="#1f5fd0", fontsize=9)
    ax[0].text(w0, 25, "post (%d)" % post, color="#2e8b57", fontsize=9)
    for t in s[(s >= w0) & (s < w1)]:
        ax[0].axvline(t, color="#1f5fd0", ls=":", lw=0.7, alpha=0.6)
    ax[0].set_title("(a) connected exc pair \u2014 pre spikes (dotted) drive post EPSPs")
    ax[0].set_xlabel("time (ms)"); ax[0].set_yticks([]); ax[0].set_xlim(w0, w1)

ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].axvline(0, color="0.6", lw=0.8, ls=":")
for m, se, n, c, lab in [(sta_e, se_e, ne, "#2e8b57", "excitatory pairs (EPSP)"),
                         (sta_i, se_i, ni, "#c0392b", "inhibitory pairs (IPSP)"),
                         (sta_c, se_c, nc, "0.5", "unconnected control")]:
    ax[1].plot(lags, m, color=c, lw=2, label="%s  n=%d" % (lab, n))
    ax[1].fill_between(lags, m - se, m + se, color=c, alpha=0.2)
ax[1].set_title("(b) spike-triggered average: POST Vm aligned to PRE spike")
ax[1].set_xlabel("lag from pre spike (ms)"); ax[1].set_ylabel("post \u0394Vm (mV)")
ax[1].legend(fontsize=9)

fig2.suptitle("Connected-neuron voltage \u2014 monosynaptic signature (recording000, normal flagship)",
              fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
f2 = os.path.join(FIGDIR, "voltage_connected_pairs.png")
fig2.savefig(f2, dpi=130, facecolor="white", bbox_inches="tight"); print("fig2 ->", f2)
print("EPSP peak=%.3f mV @ lag %d | IPSP trough=%.3f mV @ lag %d"
      % (sta_e.max(), lags[sta_e.argmax()], sta_i.min(), lags[sta_i.argmin()]))
