"""Does a single pre-spike trigger a post-spike?

Measures pre->post spike-transmission probability for ground-truth excitatory
pairs: P(post spikes 1-8 ms after a pre spike), inter-burst (isolated) vs
during-burst (summation), against a chance baseline. Also estimates rest,
threshold, single-EPSP amplitude -> how many summed EPSPs are needed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD = os.path.join(REPO, "notebooks", "NEURON data parallel", "normal", "20260721_163430")

d = np.load(os.path.join(SD, "recording000.npz"), allow_pickle=True)
V = np.asarray(d["voltage_traces"], np.float32)
N, T = V.shape
st = d["spike_times"]
burst = np.asarray(d["burst_windows"], float)[0]
def sp(i): return np.atleast_1d(np.asarray(st[i], float))

net = np.load([os.path.join(SD, f) for f in os.listdir(SD) if f.startswith("network_") and f.endswith(".npz")][0], allow_pickle=True)
conns = net["connections"]
exc_pairs = [(int(r[0]), int(r[1])) for r in conns if str(r[3]) == "exc"]

MARGIN = 150.0
def is_ib(t): return not (burst[0] - MARGIN < t < burst[1] + MARGIN)
WIN = (1.0, 8.0)                      # monosynaptic post-spike window (ms)

# ---- transmission probability + cross-correlogram ----
lags = np.arange(-15, 25)
ccg = np.zeros(len(lags))
n_pre_ib = hit_ib = 0
n_pre_bu = hit_bu = 0
exp_ib = 0.0
pair_eff = []                                     # per-pair inter-burst efficacy (excess)
for pre, post in exc_pairs:
    ps = sp(post)
    if len(ps) == 0:
        continue
    post_rate = len(ps) / T                       # spikes/ms
    p_n = p_hit = 0; p_exp = 0.0
    for t in sp(pre):
        lo = np.searchsorted(ps, t + WIN[0]); hi = np.searchsorted(ps, t + WIN[1])
        got = hi > lo
        if is_ib(t):
            n_pre_ib += 1; hit_ib += got; exp_ib += post_rate * (WIN[1] - WIN[0])
            p_n += 1; p_hit += got; p_exp += post_rate * (WIN[1] - WIN[0])
            a = np.searchsorted(ps, t + lags[0]); b = np.searchsorted(ps, t + lags[-1] + 1)
            for dt in (ps[a:b] - t):
                k = int(np.floor(dt)) - lags[0]
                if 0 <= k < len(lags):
                    ccg[k] += 1
        elif burst[0] <= t < burst[1]:
            n_pre_bu += 1; hit_bu += got
    if p_n >= 20:                                  # stable per-pair estimate only
        pair_eff.append(max(0.0, p_hit / p_n - p_exp / p_n))
pair_eff = np.array(pair_eff)

p_ib = hit_ib / max(n_pre_ib, 1)
p_base = exp_ib / max(n_pre_ib, 1)
p_bu = hit_bu / max(n_pre_bu, 1)
excess = p_ib - p_base
ccg_prob = ccg / max(n_pre_ib, 1)                 # per-pre-spike prob per 1-ms bin

print("inter-burst: P(post spike | pre spike, 1-8ms) = %.3f  (baseline %.3f, excess %.3f) over %d pre spikes"
      % (p_ib, p_base, excess, n_pre_ib))
print("during-burst: P = %.3f over %d pre spikes" % (p_bu, n_pre_bu))
print("per-pair efficacy: %d pairs, median=%.3f  90th pct=%.3f  max=%.3f"
      % (len(pair_eff), np.median(pair_eff), np.percentile(pair_eff, 90), pair_eff.max()))

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].bar(lags + 0.5, ccg_prob, width=1.0, color="#2e8b57")
ax[0].axvspan(WIN[0], WIN[1], color="orange", alpha=0.15, label="1-8 ms window")
ax[0].axvline(0, color="0.5", ls=":")
ax[0].set_title("(a) pre\u2192post cross-correlogram (exc pairs, inter-burst)")
ax[0].set_xlabel("lag: post spike - pre spike (ms)"); ax[0].set_ylabel("P(post spike) per 1-ms bin"); ax[0].legend(fontsize=8)

bars = ax[1].bar([0, 1, 2], [p_base, p_ib, p_bu],
                 color=["0.6", "#2e8b57", "#c0392b"])
ax[1].set_xticks([0, 1, 2])
ax[1].set_xticklabels(["chance\nbaseline", "isolated\npre spike\n(inter-burst)", "pre spike\nin a burst"])
ax[1].set_ylabel("P(post spike within 1-8 ms)")
ax[1].set_title("(b) spike-transmission probability")
for b, v in zip(bars, [p_base, p_ib, p_bu]):
    ax[1].text(b.get_x() + b.get_width()/2, v + .01, "%.1f%%" % (100*v), ha="center", fontsize=10)

# (c) per-pair efficacy distribution (data-driven; no threshold assumptions)
ax[2].hist(100 * pair_eff, bins=np.arange(0, 20.5, 1.0), color="#2e8b57", edgecolor="white")
ax[2].axvline(100 * np.median(pair_eff), color="k", ls="--",
              label="median %.1f%%" % (100 * np.median(pair_eff)))
ax[2].set_xlabel("per-pair transmission efficacy (%, excess over chance)")
ax[2].set_ylabel("number of exc pairs")
ax[2].set_title("(c) single-connection efficacy is weak & variable")
ax[2].legend(fontsize=9)

fig.suptitle("Is one pre-spike enough to trigger a post-spike?  \u2014 recording000, normal flagship\n"
             "NO: isolated pre spike -> post fires only %.1f%% of the time (chance %.1f%%); "
             "coincident inputs in a burst -> %.0f%%"
             % (100*p_ib, 100*p_base, 100*p_bu), fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(REPO, "figures", "spike_transmission_probability.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print("figure ->", out)
