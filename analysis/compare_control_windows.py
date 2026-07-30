"""Window-matched participation comparison for the zero-recurrence control.

The flagship's saved ``mean_participation`` (0.86) counts distinct cells firing
anywhere inside a detected BURST, which lasts far longer than 50 ms. Comparing
it against a 50 ms bin peak understates the coupled arm. This recomputes
participation over sliding windows of several widths for both arms so the
coupled/decoupled contrast is like-for-like, and reports it separately for the
t=0 initialization event and for everything after the 1 s discard.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS_MS = [50.0, 200.0, 500.0, 1000.0]


def load(tag):
    return np.load(os.path.join(HERE, "decoupled_control_%s.npz" % tag),
                   allow_pickle=True)


def sliding_participation(spikes, n, duration_ms, win_ms, step_ms=25.0):
    """Fraction of DISTINCT cells firing in each sliding window."""
    n_steps = int((duration_ms - win_ms) / step_ms) + 1
    starts = np.arange(n_steps) * step_ms
    counts = np.zeros(n_steps, dtype=np.int32)
    for times in spikes:
        t = np.asarray(times, dtype=float)
        if t.size == 0:
            continue
        # A cell counts once per window it has any spike in.
        lo = np.maximum(0, np.ceil((t - win_ms) / step_ms)).astype(int)
        hi = np.minimum(n_steps - 1, np.floor(t / step_ms)).astype(int)
        hit = np.zeros(n_steps, dtype=bool)
        for a, b in zip(lo, hi):
            if b >= a:
                hit[a:b + 1] = True
        counts += hit
    return starts, counts / float(n)


def report(tag, duration_ms=60000.0):
    d = load(tag)
    spikes = list(d["spike_times"])
    n = len(spikes)
    print("\n=== %s  (exc x%.1f, inh x%.1f)  rate %.3f Hz ==="
          % (tag, d["exc_weight_scale"], d["inh_weight_scale"], d["rate"]))
    for win in WINDOWS_MS:
        starts, part = sliding_participation(spikes, n, duration_ms, win)
        # t=0 initialization event vs everything after the standard 1 s discard.
        init = starts < 1000.0
        post = starts >= 1000.0
        i_init = int(np.argmax(part[init]))
        i_post = int(np.argmax(part[post]))
        print("  win %5.0f ms | init(<1s) %5.1f%% @ %.2f s | post(>1s) %5.1f%% @ %.2f s"
              % (win,
                 100 * part[init][i_init], starts[init][i_init] / 1000.0,
                 100 * part[post][i_post], starts[post][i_post] / 1000.0))
    # Peak-locked detail at the flagship-like 500 ms width.
    starts, part = sliding_participation(spikes, n, duration_ms, 500.0)
    post = starts >= 1000.0
    order = np.argsort(part[post])[::-1]
    seen, top = [], []
    for i in order:  # greedy non-overlapping peaks
        t = starts[post][i]
        if all(abs(t - s) > 1000.0 for s in seen):
            seen.append(t)
            top.append((t, part[post][i]))
        if len(top) >= 5:
            break
    # Baseline: how exceptional is a peak? Chance level for a Poisson cell at
    # the observed rate is 1 - exp(-rate * win); the empirical median says what
    # a typical (non-burst) window looks like.
    chance = 1.0 - np.exp(-float(d["rate"]) * 0.5)
    print("  baseline (500 ms, post-transient): median %.1f%%  p95 %.1f%%  "
          "| Poisson chance %.1f%%"
          % (100 * np.median(part[post]), 100 * np.percentile(part[post], 95),
             100 * chance))
    print("  top 5 non-overlapping post-transient events (500 ms window):")
    prev = None
    for t, p in top:
        gap = "" if prev is None else "   (+%.2f s)" % ((t - prev) / 1000.0)
        print("      %6.2f s   %5.1f%%%s" % (t / 1000.0, 100 * p, gap))
        prev = t
    # Events in time order, to expose a regular ring-down ladder if present.
    ordered = sorted(top)
    if len(ordered) > 1:
        gaps = np.diff([t for t, _ in ordered]) / 1000.0
        print("  in time order: %s" % ", ".join("%.2f s (%.0f%%)" % (t / 1000.0, 100 * p)
                                                for t, p in ordered))
        print("  inter-event gaps: %s  (mean %.2f s, sd %.2f s)"
              % (", ".join("%.2f" % g for g in gaps), gaps.mean(), gaps.std()))


if __name__ == "__main__":
    report("coupled")
    report("decoupled")
