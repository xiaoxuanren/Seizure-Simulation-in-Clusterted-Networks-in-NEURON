"""Network topology builders (pure NumPy, no NEURON dependency).

This is the biophysical-network analogue of the LIF project's ``network.py``.
It produces the *wiring graph* -- neuron positions, cluster metadata, and a
ground-truth connection table -- which :mod:`neuron_simulation.network_builder`
later instantiates as NEURON cells and synapses. Keeping topology generation
free of NEURON means it is fast, deterministic under a seed, and unit-testable.

Two builders are provided, both using a clustered spatial layout with
distance-dependent connectivity (the same backbone as the LIF
``create_clustered_network``):

* :func:`build_topology_lognormal` -- **preferred / biologically defensible.**
  Adds a *continuous* per-neuron log-normal connection propensity so the degree
  distribution is heavy-tailed with no bimodal gap, at a realistic sparse
  density (~3-4%).

* :func:`build_topology` -- clustered layout plus a *discrete* hub class (a few
  densely, long-range-connected neurons). Kept as an option, but it yields a
  BIMODAL, unrealistically concentrated degree distribution and a higher
  density; use it only when you want a densely-coupled, always-fully-recruiting
  network.

Connection-table convention (identical to the LIF project so the inference
pipeline consumes it unmodified): each row is
``[pre_id:int, post_id:int, weight:float, type:str('exc'|'inh')]`` where
``weight`` is signed (negative for inhibitory) and ``type`` encodes the sign
redundantly. Weights are peak synaptic conductances in microsiemens (uS).
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Weight parameters
# --------------------------------------------------------------------------- #
class NeuronWeightParameters:
    """Default synaptic-weight ranges for the biophysical network (in uS).

    Mirrors the LIF project's ``NetworkWeightParameters`` but expresses weights
    as NEURON peak conductances (microsiemens) rather than LIF current-equivalent
    units. Excitatory weights become ``DepSyn`` conductance increments and
    inhibitory weights become ``ExpSyn`` conductance increments; the network
    builder uses the magnitude and reads the sign only for bookkeeping.

    Args:
        None.

    Returns:
        An initialized ``NeuronWeightParameters`` with default within/between
        cluster ranges and log-normal sampling settings.
    """

    def __init__(self):
        # Peak conductances in microsiemens (uS). 1e-3 uS = 1 nS.
        # NOTE: these ranges are ABOVE the single-event rheobase of the HH cell
        # (~0.00085 uS ~ 6 mV), so a single recurrent spike is suprathreshold.
        # This is unavoidable for network bursting in this sharp-threshold HH
        # point-neuron: a genuinely subthreshold recurrent weight set (all
        # weights < ~0.00085 uS) does NOT sustain bursts (verified -- the network
        # goes silent at low noise and asynchronously tonic at high noise). The
        # 2x-trimmed variant that was tried is therefore not used. See the README
        # "Recurrent coupling and the sharp HH threshold" note.
        self.within_exc_range = (0.0010, 0.0022)
        self.between_exc_range = (0.0008, 0.0016)
        self.within_inh_range = (0.0025, 0.0055)
        self.between_inh_range = (0.0020, 0.0040)
        self.use_lognormal = True
        self.lognormal_sigma = 0.5


def calculate_lognormal_params(mean_val, sigma):
    """Convert a linear-space mean/spread into log-normal ``(mu, sigma_log)``.

    Args:
        mean_val: Desired arithmetic mean of the sampled magnitudes.
        sigma: Desired linear-space spread of the sampled magnitudes.

    Returns:
        A ``(mu, sigma_log)`` tuple suitable for ``np.random.lognormal``.
    """
    mu = np.log(mean_val / np.sqrt(1 + (sigma / mean_val) ** 2))
    sigma_log = np.sqrt(np.log(1 + (sigma / mean_val) ** 2))
    return mu, sigma_log


def generate_lognormal_weight(mean_val, sigma, rng):
    """Sample one positive synaptic magnitude from the configured log-normal law.

    Args:
        mean_val: Desired mean synaptic magnitude in linear space.
        sigma: Desired spread of synaptic magnitudes in linear space.
        rng: A ``numpy.random.Generator`` used for sampling.

    Returns:
        A positive floating-point synaptic magnitude (uS).
    """
    mu, sigma_log = calculate_lognormal_params(mean_val, sigma)
    return float(rng.lognormal(mu, sigma_log))


def get_connection_weight(pre_cluster, post_cluster, is_inhibitory, weight_params, rng):
    """Sample a signed synaptic weight (uS) for one candidate connection.

    Args:
        pre_cluster: Cluster index of the presynaptic neuron.
        post_cluster: Cluster index of the postsynaptic neuron.
        is_inhibitory: Whether the presynaptic neuron is inhibitory.
        weight_params: :class:`NeuronWeightParameters` describing the ranges.
        rng: A ``numpy.random.Generator`` used for sampling.

    Returns:
        A signed synaptic weight in uS: negative for inhibitory presynaptic
        neurons, positive for excitatory ones.
    """
    same = pre_cluster == post_cluster
    if is_inhibitory:
        weight_range = weight_params.within_inh_range if same else weight_params.between_inh_range
    else:
        weight_range = weight_params.within_exc_range if same else weight_params.between_exc_range

    if weight_params.use_lognormal:
        mean_val = float(np.mean(np.abs(weight_range)))
        weight = generate_lognormal_weight(mean_val, weight_params.lognormal_sigma * mean_val, rng)
        weight = float(np.clip(weight, np.min(np.abs(weight_range)), np.max(np.abs(weight_range))))
    else:
        weight = float(rng.uniform(*weight_range))

    return -weight if is_inhibitory else weight


# --------------------------------------------------------------------------- #
# Spatial layout & connection probability
# --------------------------------------------------------------------------- #
def generate_non_overlapping_cluster_positions(num_clusters, cluster_radius, space_size, rng):
    """Place cluster centers so their scatter radii do not overlap.

    Args:
        num_clusters: Number of cluster centers to attempt to place.
        cluster_radius: Radius used later to scatter neurons around a center.
        space_size: Side length of the square embedding space.
        rng: A ``numpy.random.Generator`` used for sampling positions.

    Returns:
        A float array of shape ``(n_placed, 2)`` of cluster-center coordinates.
        Fewer than ``num_clusters`` may be returned if the space is too small.
    """
    positions = []
    min_distance = cluster_radius * 2.0
    max_attempts = 1000

    for _ in range(num_clusters):
        placed = False
        for _attempt in range(max_attempts):
            pos = rng.uniform(0, space_size, 2)
            if not positions:
                positions.append(pos)
                placed = True
                break
            distances = np.sqrt(np.sum((np.array(positions) - pos) ** 2, axis=1))
            if np.all(distances >= min_distance):
                positions.append(pos)
                placed = True
                break
        if not placed:
            print(
                f"Warning: placed only {len(positions)} of {num_clusters} clusters "
                f"in space_size={space_size}"
            )
            break

    return np.array(positions)


def calculate_connection_probability(
    distance,
    same_cluster,
    within_cluster_prob=0.5,
    between_cluster_prob=0.1,
    max_distance=8.0,
    decay_sigma=3.0,
):
    """Compute the base connection probability for one neuron pair.

    Uses a cluster-specific base probability attenuated by a Gaussian distance
    kernel and a hard spatial cutoff, matching the LIF distance rule.

    Args:
        distance: Euclidean distance between the two neurons.
        same_cluster: Whether the pair belongs to the same cluster.
        within_cluster_prob: Base probability for same-cluster pairs.
        between_cluster_prob: Base probability for different-cluster pairs.
        max_distance: Hard cutoff beyond which the probability is zero.
        decay_sigma: Standard deviation of the Gaussian distance kernel.

    Returns:
        A probability in ``[0, 1]`` before any per-neuron propensity scaling.
    """
    if distance > max_distance:
        return 0.0
    base_prob = within_cluster_prob if same_cluster else between_cluster_prob
    distance_factor = np.exp(-(distance ** 2) / (2.0 * decay_sigma ** 2))
    return base_prob * distance_factor


def designate_hub_neurons(cluster_neuron_groups, hub_fraction, rng):
    """Choose per-cluster hub neurons for long-range hub projections.

    Args:
        cluster_neuron_groups: Per-cluster lists of neuron ids.
        hub_fraction: Fraction of each cluster promoted to hubs.
        rng: A ``numpy.random.Generator`` used for the hub draw.

    Returns:
        A tuple ``(hub_neuron_ids, hub_cluster_map)`` with the global hub ids and
        a mapping from hub id to its home cluster index.
    """
    hub_neuron_ids = []
    hub_cluster_map = {}
    for cluster_idx, neuron_ids in enumerate(cluster_neuron_groups):
        n_hubs = max(1, int(len(neuron_ids) * hub_fraction))
        hub_local = rng.choice(len(neuron_ids), n_hubs, replace=False)
        for local_idx in hub_local:
            global_id = neuron_ids[int(local_idx)]
            hub_neuron_ids.append(global_id)
            hub_cluster_map[global_id] = cluster_idx
    return hub_neuron_ids, hub_cluster_map


# --------------------------------------------------------------------------- #
# Shared spatial scaffolding
# --------------------------------------------------------------------------- #
def _build_spatial_layout(
    num_clusters,
    neurons_per_cluster_range,
    inhibitory_probability,
    cluster_radius,
    space_size,
    rng,
):
    """Sample cluster centers, cluster sizes, neuron positions, and E/I labels.

    Args:
        num_clusters: Requested number of clusters.
        neurons_per_cluster_range: Inclusive ``(low, high)`` cluster-size range.
        inhibitory_probability: Probability that a neuron is inhibitory.
        cluster_radius: Scatter radius around each cluster center.
        space_size: Side length of the square embedding space.
        rng: A ``numpy.random.Generator``.

    Returns:
        A tuple ``(neuron_positions, is_inhibitory, cluster_assignments,
        cluster_neuron_groups, cluster_centers, cluster_sizes)``.
    """
    cluster_centers = generate_non_overlapping_cluster_positions(
        num_clusters, cluster_radius, space_size, rng
    )
    n_clusters = len(cluster_centers)
    cluster_sizes = rng.integers(
        neurons_per_cluster_range[0], neurons_per_cluster_range[1] + 1, size=n_clusters
    )

    neuron_positions = []
    is_inhibitory = []
    cluster_assignments = []
    cluster_neuron_groups = [[] for _ in range(n_clusters)]

    neuron_id = 0
    for cluster_id in range(n_clusters):
        center = cluster_centers[cluster_id]
        for _ in range(int(cluster_sizes[cluster_id])):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, cluster_radius)
            pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
            neuron_positions.append(pos)
            is_inhibitory.append(bool(rng.random() < inhibitory_probability))
            cluster_assignments.append(cluster_id)
            cluster_neuron_groups[cluster_id].append(neuron_id)
            neuron_id += 1

    return (
        np.array(neuron_positions, dtype=float),
        np.array(is_inhibitory, dtype=bool),
        np.array(cluster_assignments, dtype=int),
        cluster_neuron_groups,
        cluster_centers,
        cluster_sizes,
    )


def _finalize_topology(
    connections, neuron_positions, is_inhibitory, cluster_assignments,
    cluster_neuron_groups, cluster_centers, cluster_sizes, weight_params, extra_info,
):
    """Assemble the standard topology dict returned by both builders.

    Args:
        connections: List of ``[pre, post, weight, type]`` rows.
        neuron_positions: ``(N, 2)`` neuron coordinates.
        is_inhibitory: ``(N,)`` boolean E/I labels.
        cluster_assignments: ``(N,)`` cluster index per neuron.
        cluster_neuron_groups: Per-cluster neuron-id lists.
        cluster_centers: ``(n_clusters, 2)`` cluster-center coordinates.
        cluster_sizes: ``(n_clusters,)`` neurons per cluster.
        weight_params: The :class:`NeuronWeightParameters` used for sampling.
        extra_info: Builder-specific extra ``cluster_info`` fields (hub or
            log-normal metadata).

    Returns:
        A topology dict with ``connections`` (object array), ``neuron_positions``,
        ``neuron_is_inhibitory``, ``cluster_info``, ``weight_params``, and
        ``n_neurons``.
    """
    n_neurons = len(neuron_positions)
    connections = np.array(connections, dtype=object)
    density = len(connections) / (n_neurons * (n_neurons - 1)) if n_neurons > 1 else 0.0

    cluster_info = {
        "cluster_centers": cluster_centers,
        "cluster_sizes": cluster_sizes,
        "cluster_assignments": cluster_assignments,
        "cluster_neuron_groups": [list(g) for g in cluster_neuron_groups],
        "neuron_is_inhibitory": is_inhibitory.astype(int),
        "density": float(density),
    }
    cluster_info.update(extra_info)

    print(
        f"Topology '{cluster_info.get('topology_kind', 'unknown')}': "
        f"{n_neurons} neurons, {len(connections)} synapses, density={density * 100:.2f}%"
    )
    return {
        "connections": connections,
        "neuron_positions": neuron_positions,
        "neuron_is_inhibitory": is_inhibitory,
        "cluster_assignments": cluster_assignments,
        "cluster_info": cluster_info,
        "weight_params": weight_params,
        "n_neurons": n_neurons,
    }


# --------------------------------------------------------------------------- #
# Builder 1 (PREFERRED): clustered + distance + continuous log-normal propensity
# --------------------------------------------------------------------------- #
def build_topology_lognormal(
    num_clusters=20,
    neurons_per_cluster_range=(10, 16),
    inhibitory_probability=0.2,
    cluster_radius=1.0,
    space_size=15.0,
    within_cluster_prob=0.55,
    between_cluster_prob=0.15,
    max_connection_distance=8.0,
    decay_sigma=3.0,
    ln_sigma=1.0,
    target_density=0.035,
    weight_params=None,
    seed=0,
):
    """Build the preferred clustered + log-normal-propensity topology.

    Every neuron is assigned a continuous log-normal *connection propensity*
    (mean 1, log-space std ``ln_sigma``). The probability of a directed edge is
    the distance-dependent base probability scaled by the *presynaptic* neuron's
    propensity, then globally rescaled so the expected density matches
    ``target_density``. This produces a continuous, heavy-tailed out-degree
    distribution (no bimodal gap) at a biologically realistic sparse density.

    Args:
        num_clusters: Requested number of spatial clusters.
        neurons_per_cluster_range: Inclusive ``(low, high)`` cluster-size range.
        inhibitory_probability: Probability that a neuron is inhibitory.
        cluster_radius: Scatter radius around each cluster center.
        space_size: Side length of the square embedding space.
        within_cluster_prob: Base probability for same-cluster pairs.
        between_cluster_prob: Base probability for inter-cluster pairs.
        max_connection_distance: Hard spatial cutoff for connections.
        decay_sigma: Gaussian distance-kernel width.
        ln_sigma: Log-space standard deviation of the per-neuron propensity;
            larger values give a heavier tail (more high-degree hubs).
        target_density: Target connection density (fraction of ordered pairs);
            base probabilities are rescaled to hit it. Set ``None`` to disable.
        weight_params: Optional :class:`NeuronWeightParameters`; a default is
            created when omitted.
        seed: Seed for the deterministic random draw.

    Returns:
        A topology dict (see :func:`_finalize_topology`). Its ``cluster_info``
        additionally contains ``connection_propensity`` (``(N,)`` float),
        ``ln_sigma``, ``topology_kind='lognormal'``, and the realized
        ``density``.
    """
    if weight_params is None:
        weight_params = NeuronWeightParameters()
    rng = np.random.default_rng(seed)

    (
        neuron_positions,
        is_inhibitory,
        cluster_assignments,
        cluster_neuron_groups,
        cluster_centers,
        cluster_sizes,
    ) = _build_spatial_layout(
        num_clusters, neurons_per_cluster_range, inhibitory_probability,
        cluster_radius, space_size, rng,
    )
    n_neurons = len(neuron_positions)

    # Per-neuron propensity, normalized to mean 1 so density stays controllable.
    propensity = rng.lognormal(mean=0.0, sigma=ln_sigma, size=n_neurons)
    propensity /= np.mean(propensity)

    # Pairwise base probabilities (presynaptic propensity scales out-degree).
    diff = neuron_positions[:, None, :] - neuron_positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    same = cluster_assignments[:, None] == cluster_assignments[None, :]
    base = np.where(same, within_cluster_prob, between_cluster_prob)
    base = base * np.exp(-(dist ** 2) / (2.0 * decay_sigma ** 2))
    base[dist > max_connection_distance] = 0.0
    np.fill_diagonal(base, 0.0)
    prob = base * propensity[:, None]

    # Rescale so the expected density matches the target.
    if target_density is not None:
        expected = prob.sum() / (n_neurons * (n_neurons - 1))
        if expected > 0:
            prob *= target_density / expected
    prob = np.clip(prob, 0.0, 1.0)

    draws = rng.random((n_neurons, n_neurons))
    edges = np.argwhere((draws < prob) & (~np.eye(n_neurons, dtype=bool)))

    connections = []
    for pre_id, post_id in edges:
        pre_id, post_id = int(pre_id), int(post_id)
        inh = bool(is_inhibitory[pre_id])
        weight = get_connection_weight(
            cluster_assignments[pre_id], cluster_assignments[post_id], inh, weight_params, rng
        )
        connections.append([pre_id, post_id, weight, "inh" if inh else "exc"])

    extra = {
        "topology_kind": "lognormal",
        "connection_propensity": propensity.astype(float),
        "ln_sigma": float(ln_sigma),
    }
    return _finalize_topology(
        connections, neuron_positions, is_inhibitory, cluster_assignments,
        cluster_neuron_groups, cluster_centers, cluster_sizes, weight_params, extra,
    )


# --------------------------------------------------------------------------- #
# Builder 2 (OPTION): clustered + discrete hub class
# --------------------------------------------------------------------------- #
def build_topology(
    num_clusters=20,
    neurons_per_cluster_range=(10, 16),
    inhibitory_probability=0.2,
    cluster_radius=1.0,
    space_size=15.0,
    within_cluster_prob=0.55,
    between_cluster_prob=0.15,
    max_connection_distance=8.0,
    decay_sigma=3.0,
    hub_fraction=0.1,
    hub_between_prob=0.4,
    hub_weight_scale=1.5,
    hub_reciprocal_factor=2.0,
    weight_params=None,
    seed=0,
):
    """Build a clustered topology with a discrete hub class.

    Mirrors the LIF ``create_clustered_network`` connectivity: distance-dependent
    base edges plus extra long-range projections from a small set of designated
    hub neurons. DOCUMENTED CAVEAT: the discrete hub class produces a *bimodal*,
    unrealistically concentrated degree distribution and a higher density than
    :func:`build_topology_lognormal`. Prefer the log-normal builder unless you
    specifically want a densely-coupled, always-fully-recruiting network.

    Args:
        num_clusters: Requested number of spatial clusters.
        neurons_per_cluster_range: Inclusive ``(low, high)`` cluster-size range.
        inhibitory_probability: Probability that a neuron is inhibitory.
        cluster_radius: Scatter radius around each cluster center.
        space_size: Side length of the square embedding space.
        within_cluster_prob: Base probability for same-cluster pairs.
        between_cluster_prob: Base probability for inter-cluster pairs.
        max_connection_distance: Hard spatial cutoff for connections.
        decay_sigma: Gaussian distance-kernel width.
        hub_fraction: Fraction of each cluster promoted to hubs.
        hub_between_prob: Inter-cluster connection probability for hub projections.
        hub_weight_scale: Multiplier applied to hub-originating weights.
        hub_reciprocal_factor: Extra probability boost for hub-to-hub projections.
        weight_params: Optional :class:`NeuronWeightParameters`.
        seed: Seed for the deterministic random draw.

    Returns:
        A topology dict (see :func:`_finalize_topology`). Its ``cluster_info``
        additionally contains ``hub_neuron_ids``, ``hub_fraction``,
        ``hub_between_prob``, ``hub_weight_scale``, ``hub_reciprocal_factor``,
        ``n_hub_connections``, and ``topology_kind='discrete_hub'``.
    """
    if weight_params is None:
        weight_params = NeuronWeightParameters()
    rng = np.random.default_rng(seed)

    (
        neuron_positions,
        is_inhibitory,
        cluster_assignments,
        cluster_neuron_groups,
        cluster_centers,
        cluster_sizes,
    ) = _build_spatial_layout(
        num_clusters, neurons_per_cluster_range, inhibitory_probability,
        cluster_radius, space_size, rng,
    )
    n_neurons = len(neuron_positions)

    # Base distance-dependent connections.
    connections = []
    existing_pairs = set()
    for pre_id in range(n_neurons):
        for post_id in range(n_neurons):
            if pre_id == post_id:
                continue
            distance = float(np.sqrt(np.sum((neuron_positions[pre_id] - neuron_positions[post_id]) ** 2)))
            prob = calculate_connection_probability(
                distance,
                cluster_assignments[pre_id] == cluster_assignments[post_id],
                within_cluster_prob,
                between_cluster_prob,
                max_connection_distance,
                decay_sigma,
            )
            if rng.random() < prob:
                inh = bool(is_inhibitory[pre_id])
                weight = get_connection_weight(
                    cluster_assignments[pre_id], cluster_assignments[post_id], inh, weight_params, rng
                )
                connections.append([pre_id, post_id, weight, "inh" if inh else "exc"])
                existing_pairs.add((pre_id, post_id))

    n_base = len(connections)

    # Discrete hub projections.
    hub_neuron_ids, hub_cluster_map = designate_hub_neurons(cluster_neuron_groups, hub_fraction, rng)
    hub_set = set(hub_neuron_ids)
    n_hub_connections = 0
    for hub_id in hub_neuron_ids:
        hub_cluster = cluster_assignments[hub_id]
        inh = bool(is_inhibitory[hub_id])
        for target_cluster_idx, target_ids in enumerate(cluster_neuron_groups):
            if target_cluster_idx == hub_cluster:
                continue
            for target_id in target_ids:
                if (hub_id, target_id) in existing_pairs:
                    continue
                distance = float(np.sqrt(np.sum((neuron_positions[hub_id] - neuron_positions[target_id]) ** 2)))
                if distance > max_connection_distance:
                    continue
                prob = hub_between_prob
                if target_id in hub_set:
                    prob = min(1.0, prob * hub_reciprocal_factor)
                if rng.random() < prob:
                    weight = get_connection_weight(
                        hub_cluster, target_cluster_idx, inh, weight_params, rng
                    ) * hub_weight_scale
                    connections.append([hub_id, target_id, weight, "inh" if inh else "exc"])
                    existing_pairs.add((hub_id, target_id))
                    n_hub_connections += 1

    print(f"  base connections: {n_base}, hub connections: {n_hub_connections}")

    extra = {
        "topology_kind": "discrete_hub",
        "hub_neuron_ids": np.array(hub_neuron_ids, dtype=int),
        "hub_fraction": float(hub_fraction),
        "hub_between_prob": float(hub_between_prob),
        "hub_weight_scale": float(hub_weight_scale),
        "hub_reciprocal_factor": float(hub_reciprocal_factor),
        "n_hub_connections": int(n_hub_connections),
    }
    return _finalize_topology(
        connections, neuron_positions, is_inhibitory, cluster_assignments,
        cluster_neuron_groups, cluster_centers, cluster_sizes, weight_params, extra,
    )
