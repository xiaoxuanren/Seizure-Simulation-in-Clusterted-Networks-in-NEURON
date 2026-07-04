"""Plotting helpers for rasters, population activity, degree, and topology.

Mirrors the LIF project's ``plotting.py`` surface but for the biophysical
network. Every function takes already-computed simulation/topology data and
returns a Matplotlib ``Figure`` so callers (notebook, scripts) control saving
and display. Colour convention: excitatory = blue, inhibitory = red.
"""

import numpy as np

import matplotlib.pyplot as plt

from .analysis import detect_network_bursts, population_activity

_EXC_COLOR = "#1f5fd0"
_INH_COLOR = "#d02f2f"


def _degree_counts(connections, n_neurons):
    """Return per-neuron out-degree and in-degree from a connection table.

    Args:
        connections: Connection table rows ``[pre, post, weight, type]``.
        n_neurons: Total number of neurons.

    Returns:
        A tuple ``(out_degree, in_degree)`` of ``(N,)`` integer arrays.
    """
    out_degree = np.zeros(n_neurons, dtype=int)
    in_degree = np.zeros(n_neurons, dtype=int)
    for row in connections:
        out_degree[int(row[0])] += 1
        in_degree[int(row[1])] += 1
    return out_degree, in_degree


def plot_raster(
    spike_data,
    n_neurons,
    duration_ms,
    is_inhibitory=None,
    cluster_assignments=None,
    burn_in_ms=1000.0,
    detect_bursts=True,
    title="Network raster",
):
    """Plot a cluster-sorted spike raster with a population-activity panel.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        is_inhibitory: Optional ``(N,)`` boolean array; excitatory spikes are
            drawn blue and inhibitory spikes red.
        cluster_assignments: Optional ``(N,)`` cluster index used to sort rows so
            clusters are contiguous on the y-axis.
        burn_in_ms: Startup transient marked with a vertical dashed line.
        detect_bursts: Whether to shade detected network-burst windows.
        title: Figure title.

    Returns:
        The Matplotlib ``Figure``.
    """
    if is_inhibitory is None:
        is_inhibitory = np.zeros(n_neurons, dtype=bool)
    order = np.arange(n_neurons)
    if cluster_assignments is not None:
        order = np.argsort(cluster_assignments, kind="stable")
    row_of = {nid: row for row, nid in enumerate(order)}

    fig, (ax_r, ax_p) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for nid, spikes in spike_data.items():
        s = np.asarray(spikes, dtype=float)
        if s.size == 0:
            continue
        y = np.full(s.shape, row_of[nid])
        color = _INH_COLOR if is_inhibitory[nid] else _EXC_COLOR
        ax_r.scatter(s, y, s=1.5, c=color, marker="|", linewidths=0.5)
    ax_r.set_ylabel("neuron (cluster-sorted)")
    ax_r.set_title(title)
    ax_r.set_ylim(-1, n_neurons)

    bin_centers, active_fraction, _ = population_activity(spike_data, n_neurons, duration_ms, bin_ms=10.0)
    ax_p.fill_between(bin_centers, active_fraction, color="#444", alpha=0.8)
    ax_p.axhline(0.8, color="k", ls=":", lw=1, label="80% participation")
    ax_p.set_ylabel("active fraction\n(10 ms bins)")
    ax_p.set_xlabel("time (ms)")
    ax_p.set_ylim(0, 1)

    for ax in (ax_r, ax_p):
        ax.axvline(burn_in_ms, color="green", ls="--", lw=1, alpha=0.7)
    ax_p.text(burn_in_ms, 0.9, " burn-in", color="green", fontsize=8, va="top")

    if detect_bursts:
        bursts = detect_network_bursts(spike_data, n_neurons, duration_ms, burn_in_ms=burn_in_ms)
        for b in bursts:
            ax_r.axvspan(b["start_ms"], b["end_ms"], color="orange", alpha=0.12)
        ax_p.legend(loc="upper right", fontsize=8)
        ax_r.set_title(f"{title}  ({len(bursts)} network bursts)")

    fig.tight_layout()
    return fig


def plot_population_activity(spike_data, n_neurons, duration_ms, bin_ms=10.0, burn_in_ms=1000.0):
    """Plot the population active-fraction and mean firing-rate time series.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        bin_ms: Bin width for the population signals.
        burn_in_ms: Startup transient marked with a vertical dashed line.

    Returns:
        The Matplotlib ``Figure``.
    """
    bin_centers, active_fraction, pop_rate = population_activity(spike_data, n_neurons, duration_ms, bin_ms)
    fig, ax1 = plt.subplots(figsize=(11, 3.5))
    ax1.plot(bin_centers, active_fraction, color="#444", lw=1)
    ax1.axhline(0.8, color="k", ls=":", lw=1)
    ax1.set_ylabel("active fraction", color="#444")
    ax1.set_xlabel("time (ms)")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, pop_rate, color="#1f5fd0", lw=0.8, alpha=0.6)
    ax2.set_ylabel("mean rate (Hz)", color="#1f5fd0")
    ax1.axvline(burn_in_ms, color="green", ls="--", lw=1, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_degree_distribution(connections, n_neurons, connection_propensity=None, log_axis=True, title=None):
    """Plot the out-degree distribution, contrasting bimodal vs log-normal shape.

    Args:
        connections: Connection table rows ``[pre, post, weight, type]``.
        n_neurons: Total number of neurons.
        connection_propensity: Optional per-neuron log-normal propensity (from
            the log-normal builder); overlaid to show the heavy-tailed driver.
        log_axis: Whether to use a logarithmic y-axis (degree counts span orders
            of magnitude for heavy-tailed graphs).
        title: Optional figure title.

    Returns:
        The Matplotlib ``Figure``.
    """
    out_degree, in_degree = _degree_counts(connections, n_neurons)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    bins = np.arange(0, out_degree.max() + 2)
    ax1.hist(out_degree, bins=bins, color="#1f5fd0", alpha=0.8, label="out-degree")
    ax1.hist(in_degree, bins=bins, color="#d0902f", alpha=0.5, label="in-degree")
    if log_axis:
        ax1.set_yscale("log")
    ax1.set_xlabel("degree")
    ax1.set_ylabel("neuron count (log)" if log_axis else "neuron count")
    ax1.set_title("Degree distribution")
    ax1.legend(fontsize=8)

    # Rank-degree (log-log) reveals a heavy tail as a straight-ish line, and a
    # bimodal discrete-hub graph as a distinct high-degree shelf.
    sorted_out = np.sort(out_degree)[::-1]
    ax2.loglog(np.arange(1, len(sorted_out) + 1), np.maximum(sorted_out, 0.5), "o", ms=3, color="#1f5fd0")
    ax2.set_xlabel("rank")
    ax2.set_ylabel("out-degree")
    ax2.set_title("Rank-degree (log-log)")
    if connection_propensity is not None:
        axp = ax2.twinx()
        axp.hist(np.log10(np.asarray(connection_propensity) + 1e-6), bins=20,
                 color="gray", alpha=0.2, orientation="horizontal")
        axp.set_yticks([])

    fig.suptitle(title or "Degree distribution (log-normal is heavy-tailed; discrete-hub is bimodal)")
    fig.tight_layout()
    return fig


def plot_topology_map(neuron_positions, connections, cluster_assignments, is_inhibitory=None,
                      max_edges=1500, title="Topology (clusters + connectivity)"):
    """Plot the spatial layout with cluster colours and a sample of edges.

    Args:
        neuron_positions: ``(N, 2)`` neuron coordinates.
        connections: Connection table rows ``[pre, post, weight, type]``.
        cluster_assignments: ``(N,)`` cluster index per neuron.
        is_inhibitory: Optional ``(N,)`` boolean array; inhibitory cells are
            outlined in red.
        max_edges: Maximum number of edges drawn (subsampled for legibility).
        title: Figure title.

    Returns:
        The Matplotlib ``Figure``.
    """
    pos = np.asarray(neuron_positions, dtype=float)
    n_neurons = len(pos)
    if is_inhibitory is None:
        is_inhibitory = np.zeros(n_neurons, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    n_edges = len(connections)
    step = max(1, n_edges // max_edges)
    for row in connections[::step]:
        pre, post = int(row[0]), int(row[1])
        color = _INH_COLOR if str(row[3]) == "inh" else _EXC_COLOR
        ax.plot([pos[pre, 0], pos[post, 0]], [pos[pre, 1], pos[post, 1]],
                color=color, lw=0.15, alpha=0.25, zorder=1)

    ax.scatter(pos[:, 0], pos[:, 1], c=cluster_assignments, cmap="tab20", s=28, zorder=3, edgecolors="none")
    inh = np.asarray(is_inhibitory, dtype=bool)
    if inh.any():
        ax.scatter(pos[inh, 0], pos[inh, 1], facecolors="none", edgecolors=_INH_COLOR,
                   s=60, linewidths=1.2, zorder=4, label="inhibitory")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}  (showing {len(connections[::step])}/{n_edges} edges)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_state_comparison(normal, four_ap, n_neurons, duration_ms, burn_in_ms=1000.0):
    """Plot normal vs 4-AP rasters side by side for a burst-frequency contrast.

    Args:
        normal: ``(spike_data, label)`` tuple for the normal state.
        four_ap: ``(spike_data, label)`` tuple for the 4-AP state.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        burn_in_ms: Startup transient marked on each panel.

    Returns:
        The Matplotlib ``Figure``.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, (spike_data, label) in zip(axes, (normal, four_ap)):
        bursts = detect_network_bursts(spike_data, n_neurons, duration_ms, burn_in_ms=burn_in_ms)
        for nid, spikes in spike_data.items():
            s = np.asarray(spikes, dtype=float)
            if s.size:
                ax.scatter(s, np.full(s.shape, nid), s=1.2, c="#1f5fd0", marker="|", linewidths=0.4)
        for b in bursts:
            ax.axvspan(b["start_ms"], b["end_ms"], color="orange", alpha=0.15)
        ax.axvline(burn_in_ms, color="green", ls="--", lw=1, alpha=0.7)
        ax.set_ylabel("neuron")
        ax.set_title(f"{label}  ({len(bursts)} network bursts)")
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    return fig


def plot_burst_frequency_curve(param_values, burst_rates, param_label="gbar_kA (S/cm2)",
                               merged_flags=None, title="Burst frequency vs parameter"):
    """Plot a burst-frequency-vs-parameter dose-response curve.

    Args:
        param_values: Swept parameter values (x-axis), e.g. A-current densities.
        burst_rates: Detected network-burst rate (Hz) at each value.
        param_label: X-axis label.
        merged_flags: Optional per-point booleans marking where bursts have
            merged into continuous firing (drawn as hollow markers).
        title: Figure title.

    Returns:
        The Matplotlib ``Figure``.
    """
    param_values = np.asarray(param_values, dtype=float)
    burst_rates = np.asarray(burst_rates, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(param_values, burst_rates, "-", color="#333", lw=1, zorder=1)
    if merged_flags is None:
        merged_flags = np.zeros(len(param_values), dtype=bool)
    merged_flags = np.asarray(merged_flags, dtype=bool)
    ax.scatter(param_values[~merged_flags], burst_rates[~merged_flags], color="#1f5fd0",
               s=45, zorder=2, label="discrete bursts")
    if merged_flags.any():
        ax.scatter(param_values[merged_flags], burst_rates[merged_flags], facecolors="none",
                   edgecolors="#d02f2f", s=55, zorder=2, label="merged / continuous")
    ax.set_xlabel(param_label)
    ax.set_ylabel("network-burst rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
