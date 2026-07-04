"""Participation-based network-burst detection and burst statistics.

This is the biophysical analogue of the LIF project's analysis helpers, but the
burst definition is stricter and matches the project requirement:

    A **network burst** is an event window during which **> 80% of neurons
    fire** (measured post burn-in). This is NOT the same as "the peak active
    fraction in a 10 ms bin exceeds 80%": a burst can spread its participants
    across tens of milliseconds, so participation is counted over the whole
    event window, not within a single narrow bin.

Detection has two stages:

1. Bracket candidate events from a coarse population-activity signal (contiguous
   bins whose active fraction clears a low onset threshold, merged across short
   gaps).
2. For each candidate window, count the *unique* neurons that fired anywhere in
   the window and keep it as a network burst only if that fraction clears the
   participation threshold.

The first ``burn_in_ms`` milliseconds are always excluded (startup transient).
"""

import numpy as np


def population_activity(spike_data, n_neurons, duration_ms, bin_ms=10.0):
    """Compute binned population activity for rasters and burst bracketing.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        bin_ms: Bin width in milliseconds.

    Returns:
        A tuple ``(bin_centers, active_fraction, pop_rate_hz)`` where
        ``active_fraction[i]`` is the fraction of *distinct* neurons that fired
        in bin ``i`` and ``pop_rate_hz[i]`` is the mean per-neuron firing rate in
        that bin (Hz).
    """
    n_bins = max(1, int(np.ceil(duration_ms / bin_ms)))
    counts = np.zeros(n_bins)
    active = np.zeros(n_bins)
    for spikes in spike_data.values():
        if len(spikes) == 0:
            continue
        idx = np.floor(np.asarray(spikes, dtype=float) / bin_ms).astype(int)
        idx = idx[(idx >= 0) & (idx < n_bins)]
        if idx.size == 0:
            continue
        np.add.at(counts, idx, 1)
        active[np.unique(idx)] += 1
    bin_centers = (np.arange(n_bins) + 0.5) * bin_ms
    active_fraction = active / max(1, n_neurons)
    pop_rate_hz = counts / max(1, n_neurons) / (bin_ms / 1000.0)
    return bin_centers, active_fraction, pop_rate_hz


def detect_network_bursts(
    spike_data,
    n_neurons,
    duration_ms,
    participation_threshold=0.8,
    burn_in_ms=1000.0,
    activity_bin_ms=5.0,
    onset_active_frac=0.05,
    merge_gap_ms=50.0,
    min_event_ms=8.0,
    pad_ms=10.0,
):
    """Detect network bursts by per-event participation (post burn-in).

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        participation_threshold: Minimum fraction of *distinct* neurons that
            must fire within an event window for it to count as a network burst.
        burn_in_ms: Startup transient (ms) excluded before detection.
        activity_bin_ms: Bin width (ms) for the coarse population-activity signal
            used to bracket candidate events.
        onset_active_frac: Active-neuron fraction per bin used to bracket the
            start/end of a candidate event (kept low so the whole event is
            captured before participation is measured).
        merge_gap_ms: Candidate events separated by less than this are merged.
        min_event_ms: Candidate events shorter than this are discarded.
        pad_ms: Symmetric padding (ms) added to each bracketed window before
            counting participants.

    Returns:
        A list of burst dicts sorted by time, each with keys ``start_ms``,
        ``end_ms``, ``duration_ms``, ``peak_time_ms``, ``n_participants``, and
        ``participation`` (fraction of neurons that fired in the window).
    """
    # Restrict to the post-transient window and re-zero the clock.
    trimmed = {}
    for nid, spikes in spike_data.items():
        s = np.asarray(spikes, dtype=float)
        s = s[s >= burn_in_ms] - burn_in_ms
        trimmed[nid] = s
    eff_duration = max(0.0, duration_ms - burn_in_ms)
    if eff_duration <= 0:
        return []

    bin_centers, active_fraction, pop_rate = population_activity(
        trimmed, n_neurons, eff_duration, bin_ms=activity_bin_ms
    )
    n_bins = len(active_fraction)

    # Stage 1: bracket contiguous supra-onset regions into candidate events.
    above = active_fraction >= onset_active_frac
    raw = []
    i = 0
    while i < n_bins:
        if above[i]:
            j = i
            while j < n_bins and above[j]:
                j += 1
            raw.append([i * activity_bin_ms, min(j * activity_bin_ms, eff_duration)])
            i = j
        else:
            i += 1

    # Merge events separated by short gaps.
    merged = []
    for start, end in raw:
        if merged and start - merged[-1][1] < merge_gap_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Stage 2: measure participation over each padded window.
    spike_arrays = [trimmed[i] for i in range(n_neurons)]
    bursts = []
    for start, end in merged:
        if end - start < min_event_ms:
            continue
        w0 = max(0.0, start - pad_ms)
        w1 = min(eff_duration, end + pad_ms)
        participants = 0
        peak_time = 0.5 * (w0 + w1)
        for s in spike_arrays:
            if s.size and np.any((s >= w0) & (s < w1)):
                participants += 1
        participation = participants / max(1, n_neurons)
        if participation >= participation_threshold:
            # Peak time = center of mass of population activity in the window.
            in_win = [(s[(s >= w0) & (s < w1)]) for s in spike_arrays]
            all_in = np.concatenate([a for a in in_win if a.size]) if any(a.size for a in in_win) else np.array([w0])
            peak_time = float(np.median(all_in))
            bursts.append(
                {
                    "start_ms": float(w0 + burn_in_ms),
                    "end_ms": float(w1 + burn_in_ms),
                    "duration_ms": float(w1 - w0),
                    "peak_time_ms": float(peak_time + burn_in_ms),
                    "n_participants": int(participants),
                    "participation": float(participation),
                }
            )
    return bursts


def burst_statistics(bursts, duration_ms, burn_in_ms=1000.0):
    """Summarize a detected burst sequence.

    Args:
        bursts: List of burst dicts from :func:`detect_network_bursts`.
        duration_ms: Full recording duration in milliseconds.
        burn_in_ms: Startup transient (ms) that was excluded from detection;
            used to compute the analysed duration for the burst rate.

    Returns:
        A dict with ``n_bursts``, ``burst_rate_hz`` (bursts per second over the
        analysed window), ``mean_ibi_ms`` (inter-burst interval, peak-to-peak),
        ``std_ibi_ms``, ``mean_duration_ms``, ``mean_participation``, and
        ``merged`` (``True`` when only one long burst was found -- a sign the
        network has collapsed into continuous firing).
    """
    analysed_ms = max(1e-9, duration_ms - burn_in_ms)
    n = len(bursts)
    if n == 0:
        return {
            "n_bursts": 0,
            "burst_rate_hz": 0.0,
            "mean_ibi_ms": float("nan"),
            "std_ibi_ms": float("nan"),
            "mean_duration_ms": float("nan"),
            "mean_participation": float("nan"),
            "merged": False,
        }
    peaks = np.array([b["peak_time_ms"] for b in bursts])
    ibis = np.diff(np.sort(peaks)) if n > 1 else np.array([])
    durations = np.array([b["duration_ms"] for b in bursts])
    participations = np.array([b["participation"] for b in bursts])
    # A single burst spanning most of the analysed window == continuous firing.
    merged = n == 1 and durations[0] > 0.5 * analysed_ms
    return {
        "n_bursts": int(n),
        "burst_rate_hz": float(n / (analysed_ms / 1000.0)),
        "mean_ibi_ms": float(np.mean(ibis)) if ibis.size else float("nan"),
        "std_ibi_ms": float(np.std(ibis)) if ibis.size else float("nan"),
        "mean_duration_ms": float(np.mean(durations)),
        "mean_participation": float(np.mean(participations)),
        "merged": bool(merged),
    }


def firing_rate_summary(spike_data, duration_ms, burn_in_ms=1000.0):
    """Compute basic per-neuron firing-rate statistics (post burn-in).

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        duration_ms: Recording duration in milliseconds.
        burn_in_ms: Startup transient (ms) excluded before rates are computed.

    Returns:
        A dict with ``mean_rate_hz``, ``median_rate_hz``, ``max_rate_hz``, and
        ``active_fraction`` (fraction of neurons that fired at least once).
    """
    analysed_s = max(1e-9, (duration_ms - burn_in_ms) / 1000.0)
    rates = []
    for spikes in spike_data.values():
        s = np.asarray(spikes, dtype=float)
        s = s[s >= burn_in_ms]
        rates.append(len(s) / analysed_s)
    rates = np.array(rates) if rates else np.array([0.0])
    return {
        "mean_rate_hz": float(np.mean(rates)),
        "median_rate_hz": float(np.median(rates)),
        "max_rate_hz": float(np.max(rates)),
        "active_fraction": float(np.mean(rates > 0)),
    }
