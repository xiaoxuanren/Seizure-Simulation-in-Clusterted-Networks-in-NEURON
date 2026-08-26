"""Participation-based network-burst detection and burst statistics.

This is the biophysical analogue of the LIF project's analysis helpers, but the
burst definition is stricter and matches the project requirement:

    A **network burst** is an event window during which more than
    ``participation_threshold`` of neurons fire (measured post burn-in). The
    project default is **0.35** -- the "loose burst" definition: these events are
    genuinely low-participation, so a >80% detector misses them entirely. Pass
    ``participation_threshold=0.8`` for the strict, near-whole-network events.
    This is NOT the same as "the peak active fraction in a 10 ms bin exceeds the
    threshold": a burst can spread its participants
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
    participation_threshold=0.35,
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
        participation_threshold: Fraction of *distinct* neurons that must be
            strictly exceeded within an event window for it to count as a network
            burst (default 0.35 ⇒ a loose burst needs > 35% participation).
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
        # Spec: a network burst requires strictly MORE than the threshold
        # fraction of neurons (> 35% by default), not >=.
        if participation > participation_threshold:
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


def detect_network_bursts_all(
    spike_data,
    n_neurons,
    duration_ms,
    burn_in_ms=1000.0,
    activity_bin_ms=5.0,
    onset_active_frac=0.01,
    merge_gap_ms=100.0,
    min_event_ms=8.0,
    pad_ms=10.0,
    min_participation=0.10,
    significance_k=5.0,
    threshold_mode="mad",
    full_participation=0.35,
):
    """Detect ALL network bursts -- full AND partial participation.

    NOTE: two defaults deliberately differ from the legacy detector, validated
    against hand-annotated rasters (sweep_c40_seed20 rec 003, 2026-08-26):
    ``onset_active_frac=0.01`` (0.05 never brackets weak partial events --
    they don't reach 5% of neurons in any single 5 ms bin) and
    ``merge_gap_ms=100`` (50 ms splits one event into precursor+main
    fragments; 100 ms matches the literature's within-burst timescale).

    Same Stage-1 bracketing as :func:`detect_network_bursts` (kept verbatim so
    the two detectors bracket identical candidates), but the fixed
    participation gate is replaced by a STATISTICAL acceptance rule against the
    recording's own background, following standard population-rate burst
    detection practice (see Cotterill et al. 2016 J Neurophysiol for the
    method-comparison landscape; the adaptive-threshold option follows the
    logISI bimodality principle of Pasquale et al. 2010):

      * ``threshold_mode="mad"`` (default): a candidate is a burst when its
        peak per-bin active fraction exceeds
        ``median + significance_k * 1.4826 * MAD`` of the background
        (bins outside all candidate windows -- robust to burst contamination).
      * ``threshold_mode="valley"``: the threshold is the valley between the
        two modes of the log active-fraction distribution (self-tuning per
        recording; falls back to "mad" when the distribution is unimodal).

    Additionally ``participation >= min_participation`` rejects micro-events.
    Every accepted burst is classified ``"full"`` (participation >
    ``full_participation`` -- the legacy definition) or ``"partial"``.

    Returns:
        ``(bursts, meta)``: bursts as in :func:`detect_network_bursts` plus
        ``peak_active_fraction``, ``n_spikes`` and ``burst_class`` per burst;
        ``meta`` records the background statistics and threshold used.
    """
    trimmed = {}
    for nid, spikes in spike_data.items():
        s = np.asarray(spikes, dtype=float)
        s = s[s >= burn_in_ms] - burn_in_ms
        trimmed[nid] = s
    eff_duration = max(0.0, duration_ms - burn_in_ms)
    if eff_duration <= 0:
        return [], {}

    bin_centers, active_fraction, _ = population_activity(
        trimmed, n_neurons, eff_duration, bin_ms=activity_bin_ms
    )
    n_bins = len(active_fraction)

    # --- Stage 1: bracket candidates (verbatim from detect_network_bursts) ---
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
    merged = []
    for start, end in raw:
        if merged and start - merged[-1][1] < merge_gap_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    merged = [[s, e] for s, e in merged if e - s >= min_event_ms]

    # --- background statistics from bins OUTSIDE all candidate windows -------
    outside = np.ones(n_bins, dtype=bool)
    for start, end in merged:
        b0 = max(0, int((start - pad_ms) / activity_bin_ms))
        b1 = min(n_bins, int(np.ceil((end + pad_ms) / activity_bin_ms)))
        outside[b0:b1] = False
    bg = active_fraction[outside] if outside.any() else active_fraction
    bg_median = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_median)))
    # 1/n floor keeps the gate meaningful when most background bins are empty
    mad_level = bg_median + significance_k * max(1.4826 * bg_mad, 1.0 / n_neurons)

    threshold_used, mode_used = mad_level, "mad"
    if threshold_mode == "valley":
        floor = 1.0 / n_neurons
        logf = np.log10(active_fraction + floor)
        hist, edges = np.histogram(logf, bins=60)
        kernel = np.ones(5) / 5.0
        smooth = np.convolve(hist, kernel, mode="same")
        peaks = [k for k in range(1, len(smooth) - 1)
                 if smooth[k] >= smooth[k - 1] and smooth[k] >= smooth[k + 1]
                 and smooth[k] > 0]
        if len(peaks) >= 2:
            top2 = sorted(sorted(peaks, key=lambda k: -smooth[k])[:2])
            valley_bin = top2[0] + int(np.argmin(smooth[top2[0]:top2[1] + 1]))
            valley = 10 ** (0.5 * (edges[valley_bin] + edges[valley_bin + 1])) - floor
            if valley > bg_median:
                threshold_used, mode_used = float(valley), "valley"

    # --- Stage 2: statistical acceptance + classification --------------------
    spike_arrays = [trimmed[i] for i in range(n_neurons)]
    bursts = []
    for start, end in merged:
        w0 = max(0.0, start - pad_ms)
        w1 = min(eff_duration, end + pad_ms)
        b0 = max(0, int(w0 / activity_bin_ms))
        b1 = min(n_bins, max(b0 + 1, int(np.ceil(w1 / activity_bin_ms))))
        peak_frac = float(active_fraction[b0:b1].max())
        in_win = [s[(s >= w0) & (s < w1)] for s in spike_arrays]
        participants = sum(1 for a in in_win if a.size)
        participation = participants / max(1, n_neurons)
        if peak_frac < threshold_used or participation < min_participation:
            continue
        all_in = (np.concatenate([a for a in in_win if a.size])
                  if participants else np.array([w0]))
        bursts.append(
            {
                "start_ms": float(w0 + burn_in_ms),
                "end_ms": float(w1 + burn_in_ms),
                "duration_ms": float(w1 - w0),
                "peak_time_ms": float(np.median(all_in) + burn_in_ms),
                "n_participants": int(participants),
                "participation": float(participation),
                "peak_active_fraction": peak_frac,
                "n_spikes": int(sum(a.size for a in in_win)),
                "burst_class": "full" if participation > full_participation else "partial",
            }
        )
    meta = {
        "threshold_mode": mode_used,
        "threshold_used": float(threshold_used),
        "bg_median": bg_median,
        "bg_mad": bg_mad,
        "significance_k": float(significance_k),
        "min_participation": float(min_participation),
        "full_participation": float(full_participation),
        "n_candidates": len(merged),
    }
    return bursts, meta


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
