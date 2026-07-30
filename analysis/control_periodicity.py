"""Period and decay of the population events in each control arm.

The greedy top-5 peak list can miss events and so mis-estimate the period. This
uses (a) all local maxima above a baseline-referenced threshold and (b) the
autocorrelation of the participation series, which needs no peak picking.

Question it answers: is the coupled network's burst rhythm the SAME rhythm the
shared initial condition rings at in the decoupled arm, or a different one?
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from compare_control_windows import load, sliding_participation  # noqa: E402


def events(starts, part, thresh, min_sep_ms=2000.0):
    """Local maxima above ``thresh``, separated by at least ``min_sep_ms``."""
    out = []
    order = np.argsort(part)[::-1]
    for i in order:
        if part[i] < thresh:
            break
        if all(abs(starts[i] - t) >= min_sep_ms for t, _ in out):
            out.append((starts[i], part[i]))
    return sorted(out)


def autocorr_period(part, step_ms=25.0, min_lag_s=1.5, max_lag_s=20.0):
    """First autocorrelation peak in [min_lag, max_lag] -- a peak-free period."""
    x = part - part.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    lo = int(min_lag_s * 1000.0 / step_ms)
    hi = min(len(ac) - 1, int(max_lag_s * 1000.0 / step_ms))
    seg = ac[lo:hi]
    j = int(np.argmax(seg))
    return (lo + j) * step_ms / 1000.0, seg[j]


def report(tag, win_ms=500.0, duration_ms=60000.0):
    d = load(tag)
    spikes = list(d["spike_times"])
    starts, part = sliding_participation(spikes, len(spikes), duration_ms, win_ms)
    post = starts >= 1000.0
    s, p = starts[post], part[post]

    p95 = np.percentile(p, 95)
    med = np.median(p)
    thresh = med + 0.5 * (p95 - med)          # halfway between typical and high
    ev = events(s, p, thresh)

    print("\n=== %s (exc x%.1f) | rate %.3f Hz | median %.1f%% p95 %.1f%% "
          "| threshold %.1f%% ==="
          % (tag, d["exc_weight_scale"], d["rate"], 100 * med, 100 * p95,
             100 * thresh))
    print("  %d events above threshold:" % len(ev))
    for t, v in ev:
        print("      %6.2f s   %5.1f%%" % (t / 1000.0, 100 * v))
    if len(ev) > 1:
        gaps = np.diff([t for t, _ in ev]) / 1000.0
        print("  inter-event gaps (s): %s" % ", ".join("%.2f" % g for g in gaps))
        print("  period from peaks: %.2f +/- %.2f s (n=%d)"
              % (gaps.mean(), gaps.std(), gaps.size))

    lag, r = autocorr_period(p)
    print("  period from autocorrelation: %.2f s (r = %.3f)" % (lag, r))

    # Amplitude decay: does the event series ring down toward baseline?
    if len(ev) >= 3:
        amps = np.array([v for _, v in ev])
        first_half = amps[:max(1, len(amps) // 2)].mean()
        second_half = amps[len(amps) // 2:].mean()
        print("  amplitude: first half %.1f%% -> second half %.1f%%  (%s)"
              % (100 * first_half, 100 * second_half,
                 "ringing down" if second_half < 0.75 * first_half else "sustained"))
        # Where does it fall back into the noise?
        below = [t for t, v in ev if v <= p95]
        if below:
            print("  first event at/below p95 baseline: %.2f s" % (below[0] / 1000.0))
        else:
            print("  no event falls back to the p95 baseline within 60 s")


if __name__ == "__main__":
    report("coupled")
    report("decoupled")
