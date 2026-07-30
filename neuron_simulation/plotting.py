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
    randomize_rows=False,
    row_seed=0,
    dot_size=20.0,
    show_burst_count=True,
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
        dot_size: Marker area (matplotlib ``scatter`` ``s``) for each spike dot.
            Lower values give finer dots, which read better on long recordings.
        show_burst_count: If ``True`` (and ``detect_bursts``), append the detected
            network-burst count to the title. Burst shading is unaffected.

    Returns:
        The Matplotlib ``Figure``.
    """
    if is_inhibitory is None:
        is_inhibitory = np.zeros(n_neurons, dtype=bool)
    order = np.arange(n_neurons)
    if cluster_assignments is not None:
        order = np.argsort(cluster_assignments, kind="stable")
    if randomize_rows:
        # Shuffle which row each neuron occupies. If burst synchrony is genuine
        # (network-wide), the vertical-stripe pattern survives; if it were an
        # artifact of grouping clusters by index, it would smear out.
        order = np.random.default_rng(row_seed).permutation(n_neurons)
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
        ax_r.scatter(s, y, s=dot_size, c=color, marker=".", linewidths=0.0)
    ax_r.set_ylabel("neuron (randomized)" if randomize_rows else "neuron (cluster-sorted)")
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
        if show_burst_count:
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
                ax.scatter(s, np.full(s.shape, nid), s=4.0, c="#1f5fd0", marker=".", linewidths=0.0)
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


def plot_raster_with_ko(
    spike_data,
    n_neurons,
    duration_ms,
    ko_data,
    is_inhibitory=None,
    cluster_assignments=None,
    burn_in_ms=0.0,
    title="Raster + [K+]o",
    randomize_rows=False,
    row_seed=0,
):
    """Plot a cluster-sorted raster with mean extracellular [K+]o(t) underneath.

    This is the seizure read-out: the bottom panel shows mean [K+]o across
    neurons (with the min-max band). See the note above ``ax_k`` below for the
    measured [K+]o range and why it does not reach the ictal band.

    Args:
        spike_data: Mapping from neuron id to spike times in milliseconds.
        n_neurons: Total number of neurons.
        duration_ms: Recording duration in milliseconds.
        ko_data: The ``ko_data`` dict returned by
            :func:`neuron_simulation.simulation.run_simulation` (or
            ``run_single_state``), with ``times``/``mean_ko``/``min_ko``/``max_ko``.
        is_inhibitory: Optional ``(N,)`` boolean array (inhibitory spikes red).
        cluster_assignments: Optional ``(N,)`` cluster index for row sorting.
        burn_in_ms: Startup transient marked with a vertical dashed line.
        title: Figure title.

    Returns:
        The Matplotlib ``Figure``.
    """
    if is_inhibitory is None:
        is_inhibitory = np.zeros(n_neurons, dtype=bool)
    order = np.arange(n_neurons)
    if cluster_assignments is not None:
        order = np.argsort(cluster_assignments, kind="stable")
    if randomize_rows:
        # Shuffle which row each neuron occupies. If burst synchrony is genuine
        # (network-wide), the vertical-stripe pattern survives; if it were an
        # artifact of grouping clusters by index, it would smear out.
        order = np.random.default_rng(row_seed).permutation(n_neurons)
    row_of = {nid: row for row, nid in enumerate(order)}

    fig, (ax_r, ax_k) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for nid, spikes in spike_data.items():
        s = np.asarray(spikes, dtype=float)
        if s.size == 0:
            continue
        y = np.full(s.shape, row_of[nid])
        color = _INH_COLOR if is_inhibitory[nid] else _EXC_COLOR
        ax_r.scatter(s, y, s=20.0, c=color, marker=".", linewidths=0.0)
    ax_r.set_ylabel("neuron (randomized)" if randomize_rows else "neuron (cluster-sorted)")
    ax_r.set_title(title)
    ax_r.set_ylim(-1, n_neurons)

    times = np.asarray(ko_data["times"], dtype=float)
    # [K+]o axis: this model does NOT reach ictal extracellular potassium.
    # The kdyn accumulation loop has gain ~ tau_k * epsilon, and tau_k is held at
    # 200 ms, which makes it nearly inert by design. Measured range across the
    # single-knob sweep is 3.99-4.26 mM (rest 4.0 mM); with the knob fully off it
    # reaches only ~4.98 mM.
    #
    # This is deliberate. Two seizure mechanisms appear in the literature --
    # adaptation loss and K+ clearance failure -- and this model isolates the first
    # by pinning the second. sAHP is declared NONSPECIFIC_CURRENT with a private
    # ek = -90 mV and no USEION, so its current never enters ik and cannot drive
    # kdyn. Even counting it, the counterfactual is +0.135 mM (3.99 -> ~4.12) --
    # still far from ictal, because tau_k is the binding constraint, not the
    # declaration.
    #
    # For the K+-clearance route, use states.kclearance_seizure_state() (tau_k = 12000).
    ax_k.fill_between(times, ko_data["min_ko"], ko_data["max_ko"], color="#c0392b", alpha=0.2,
                      label="min-max across neurons")
    ax_k.plot(times, ko_data["mean_ko"], color="#c0392b", lw=1.2, label="mean [K+]o")
    ax_k.axhline(4.0, color="k", ls=":", lw=1, label="rest (4 mM)")
    ax_k.set_ylabel("[K+]o (mM)")
    ax_k.set_xlabel("time (ms)")
    ax_k.legend(fontsize=7, loc="upper right")

    if burn_in_ms > 0:
        for ax in (ax_r, ax_k):
            ax.axvline(burn_in_ms, color="green", ls="--", lw=1, alpha=0.7)
    fig.tight_layout()
    return fig


def cluster_connection_stats(connections, cluster_assignments, n_neurons, hub_percentile=90):
    """Compute inter/intra-cluster connection rates, density, and hub stats.

    Rates are over *ordered* (directed) pairs. The within-cluster rate is the
    number of within-cluster edges divided by the number of possible ordered
    within-cluster pairs; likewise for between-cluster.

    Args:
        connections: Connection table rows ``[pre, post, weight, type]``.
        cluster_assignments: ``(N,)`` cluster index per neuron.
        n_neurons: Total number of neurons.
        hub_percentile: Out-degree percentile at/above which a neuron is a hub
            (default 90 => top 10%).

    Returns:
        A dict with ``n_neurons``, ``n_edges``, ``n_clusters``, cluster-size
        summary, ``density``, ``within_rate``, ``between_rate``, ``ratio``,
        ``frac_within``/``frac_between`` (edge composition), out-degree summary,
        and hub summary (``n_hubs``, ``hub_out_thresh``, ``hub_edge_share``,
        ``hub_frac``).
    """
    ca = np.asarray(cluster_assignments).astype(int)
    pre = np.array([int(r[0]) for r in connections], dtype=int)
    post = np.array([int(r[1]) for r in connections], dtype=int)
    same = ca[pre] == ca[post]
    sizes = np.bincount(ca, minlength=(ca.max() + 1 if len(ca) else 1))
    sizes = sizes[sizes > 0]
    poss_within = int(np.sum(sizes * (sizes - 1)))
    poss_between = n_neurons * (n_neurons - 1) - poss_within
    n_edges = len(connections)
    within_rate = same.sum() / max(1, poss_within)
    between_rate = (~same).sum() / max(1, poss_between)
    out_degree, _ = _degree_counts(connections, n_neurons)
    thresh = np.percentile(out_degree, hub_percentile)
    hub = out_degree >= max(thresh, 1)
    return {
        "n_neurons": int(n_neurons),
        "n_edges": int(n_edges),
        "n_clusters": int(len(sizes)),
        "cluster_size_min": int(sizes.min()) if len(sizes) else 0,
        "cluster_size_median": int(np.median(sizes)) if len(sizes) else 0,
        "cluster_size_max": int(sizes.max()) if len(sizes) else 0,
        "density": n_edges / max(1, n_neurons * (n_neurons - 1)),
        "within_rate": float(within_rate),
        "between_rate": float(between_rate),
        "ratio": float(within_rate / between_rate) if between_rate > 0 else float("inf"),
        "frac_within": float(same.sum() / max(1, n_edges)),
        "frac_between": float((~same).sum() / max(1, n_edges)),
        "outdeg_mean": float(out_degree.mean()),
        "outdeg_median": int(np.median(out_degree)),
        "outdeg_max": int(out_degree.max()),
        "n_hubs": int(hub.sum()),
        "hub_out_thresh": int(max(thresh, 1)),
        "hub_edge_share": float(out_degree[hub].sum() / max(1, out_degree.sum())),
        "hub_frac": float(hub.sum() / max(1, n_neurons)),
        "hub_percentile": float(hub_percentile),
    }


def format_topology_stats(stats):
    """Return a printable multi-line summary of ``cluster_connection_stats``.

    Args:
        stats: The dict returned by :func:`cluster_connection_stats`.

    Returns:
        A formatted string block.
    """
    return (
        "==== TOPOLOGY STATS ====\n"
        f"N = {stats['n_neurons']} neurons | {stats['n_edges']} directed synapses | "
        f"{stats['n_clusters']} clusters "
        f"(sizes {stats['cluster_size_min']}/{stats['cluster_size_median']}/{stats['cluster_size_max']})\n"
        f"OVERALL density        = {stats['density'] * 100:.2f}%\n"
        f"WITHIN-cluster rate    = {stats['within_rate'] * 100:.2f}%\n"
        f"BETWEEN-cluster rate   = {stats['between_rate'] * 100:.3f}%\n"
        f"WITHIN : BETWEEN ratio = {stats['ratio']:.1f} x\n"
        f"edge composition       = {stats['frac_within'] * 100:.0f}% within, "
        f"{stats['frac_between'] * 100:.0f}% between\n"
        f"out-degree mean/median/max = {stats['outdeg_mean']:.1f}/"
        f"{stats['outdeg_median']}/{stats['outdeg_max']}\n"
        f"hubs (top {100 - stats['hub_percentile']:g}% out-deg, >= {stats['hub_out_thresh']}) = {stats['n_hubs']} cells "
        f"carry {stats['hub_edge_share'] * 100:.0f}% of all outgoing edges"
    )


def plot_topology_overview(neuron_positions, connections, cluster_assignments,
                           is_inhibitory=None, hub_percentile=90, title=None):
    """Four-panel graph-topology figure: layout, hub fan-out, matrix, degree.

    Panels: (a) spatial layout with cluster colours, inter-cluster edges, hub
    stars and inhibitory outlines; (b) fan-out of the three highest-out-degree
    hubs; (c) connection matrix sorted by cluster (block-diagonal = within);
    (d) out-degree distribution (hub vs non-hub).

    Args:
        neuron_positions: ``(N, 2)`` neuron coordinates.
        connections: Connection table rows ``[pre, post, weight, type]``.
        cluster_assignments: ``(N,)`` cluster index per neuron.
        is_inhibitory: Optional ``(N,)`` boolean array.
        hub_percentile: Out-degree percentile defining hubs (default 90).
        title: Optional suptitle; a stats summary is used if omitted.

    Returns:
        The Matplotlib ``Figure``.
    """
    pos = np.asarray(neuron_positions, dtype=float)
    N = len(pos)
    ca = np.asarray(cluster_assignments).astype(int)
    inh = (np.zeros(N, bool) if is_inhibitory is None
           else np.asarray(is_inhibitory, bool))
    pre = np.array([int(r[0]) for r in connections], dtype=int)
    post = np.array([int(r[1]) for r in connections], dtype=int)
    same = ca[pre] == ca[post]
    out_degree, _ = _degree_counts(connections, N)
    hub = out_degree >= max(np.percentile(out_degree, hub_percentile), 1)
    stats = cluster_connection_stats(connections, ca, N, hub_percentile)

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    cmap = plt.cm.tab20(np.linspace(0, 1, max(2, stats["n_clusters"])))
    # (a) spatial layout
    a = ax[0, 0]
    for i, j in zip(pre[~same], post[~same]):
        a.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="#cccccc", lw=0.2, zorder=1)
    a.scatter(pos[~hub, 0], pos[~hub, 1], s=16, c=ca[~hub], cmap="tab20", zorder=2)
    if inh.any():
        a.scatter(pos[inh, 0], pos[inh, 1], s=26, facecolors="none", edgecolors="#111",
                  lw=0.8, zorder=3, label="inhibitory")
    a.scatter(pos[hub, 0], pos[hub, 1], s=90, marker="*", color=_INH_COLOR,
              edgecolors="k", lw=0.4, zorder=4, label="hub")
    a.set_title(f"(a) Spatial layout — {stats['n_clusters']} clusters, grey = inter-cluster edges", fontsize=11)
    a.legend(fontsize=8, loc="upper right"); a.set_xticks([]); a.set_yticks([])
    # (b) hub fan-out
    b = ax[0, 1]
    b.scatter(pos[:, 0], pos[:, 1], s=8, color="#dddddd", zorder=1)
    cols = [_INH_COLOR, "#1f77b4", "#2ca02c"]
    for k, hb in enumerate(np.argsort(out_degree)[-3:]):
        for t in post[pre == hb]:
            b.plot([pos[hb, 0], pos[t, 0]], [pos[hb, 1], pos[t, 1]],
                   color=cols[k], lw=0.35, alpha=0.7, zorder=2)
        b.scatter(*pos[hb], s=110, marker="*", color=cols[k], edgecolors="k", lw=0.5,
                  zorder=3, label=f"gid {hb} (out={out_degree[hb]})")
    b.set_title("(b) Hub fan-out — 3 highest-out-degree cells", fontsize=11)
    b.legend(fontsize=8, loc="upper right"); b.set_xticks([]); b.set_yticks([])
    # (c) connection matrix sorted by cluster
    c = ax[1, 0]
    order = np.argsort(ca, kind="stable")
    rank = np.empty(N, int); rank[order] = np.arange(N)
    M = np.zeros((N, N)); M[rank[pre], rank[post]] = 1
    c.imshow(M, cmap="Greys", interpolation="nearest", aspect="equal")
    boundary = 0
    for s in np.bincount(ca)[np.bincount(ca) > 0]:
        boundary += s
        c.axhline(boundary - 0.5, color="#c0392b", lw=0.4)
        c.axvline(boundary - 0.5, color="#c0392b", lw=0.4)
    c.set_title("(c) Connection matrix (sorted by cluster)\nblock-diagonal = within, off-diagonal = between", fontsize=11)
    c.set_xlabel("post neuron"); c.set_ylabel("pre neuron")
    # (d) out-degree distribution
    d = ax[1, 1]
    bins = np.arange(0, out_degree.max() + 2)
    d.hist(out_degree[~hub], bins=bins, color="#4a7ab5", alpha=0.85, label=f"non-hub (n={(~hub).sum()})")
    d.hist(out_degree[hub], bins=bins, color=_INH_COLOR, alpha=0.85, label=f"hub (n={hub.sum()})")
    d.axvline(out_degree.mean(), color="#333", ls="--", lw=1, label=f"mean {out_degree.mean():.0f}")
    d.set_title(f"(d) Out-degree distribution\n{stats['hub_edge_share'] * 100:.0f}% of edges from top {stats['hub_frac'] * 100:.0f}% of neurons", fontsize=11)
    d.set_xlabel("out-degree (# postsynaptic targets)"); d.set_ylabel("# neurons"); d.legend(fontsize=8)

    if title is None:
        title = (f"Clustered topology  |  N={N}, density {stats['density'] * 100:.2f}%, "
                 f"within/between {stats['within_rate'] * 100:.1f}%/{stats['between_rate'] * 100:.2f}% "
                 f"(ratio {stats['ratio']:.0f}x)")
    fig.suptitle(title, fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def topology_report(topology, hub_percentile=90, print_stats=True):
    """Convenience: stats + overview figure directly from a topology dict.

    Unpacks a topology dict (as returned by
    :mod:`neuron_simulation.topology`) and returns both its connection stats
    and the four-panel overview figure. Intended so every topology build can
    emit the standard raster-companion topology view in one call.

    Args:
        topology: Topology dict with ``neuron_positions``, ``connections``,
            ``cluster_assignments``, ``neuron_is_inhibitory``, ``n_neurons``.
        hub_percentile: Out-degree percentile defining hubs.
        print_stats: If True, print the formatted stats block.

    Returns:
        A tuple ``(stats, fig)``.
    """
    conn = topology["connections"]
    ca = topology["cluster_assignments"]
    N = topology["n_neurons"]
    inh = topology.get("neuron_is_inhibitory")
    stats = cluster_connection_stats(conn, ca, N, hub_percentile)
    if print_stats:
        print(format_topology_stats(stats))
    fig = plot_topology_overview(topology["neuron_positions"], conn, ca,
                                 is_inhibitory=inh, hub_percentile=hub_percentile)
    return stats, fig
