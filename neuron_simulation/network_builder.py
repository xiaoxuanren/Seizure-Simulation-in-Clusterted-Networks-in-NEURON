"""Assemble a runnable NEURON network from a topology dict.

Given the pure-NumPy topology produced by :mod:`neuron_simulation.topology`,
this module instantiates the biophysical network:

* one :class:`neuron_simulation.neurons.Cell` per neuron (HH + A-current),
* one *point-process synapse per directed edge* so short-term depression and
  the ground-truth wiring stay strictly per-connection,
* a ``NetCon`` from each presynaptic soma to its target synapse, and
* an independent Poisson background generator per cell (the sole drive).

Dale's law is enforced by the topology: a presynaptic neuron's E/I label fixes
the sign of *every* synapse it makes -- excitatory pre neurons build ``DepSyn``
(reversal 0 mV, optional short-term depression), inhibitory pre neurons build
``ExpSyn`` (reversal -75 mV). The signed weight in the connection table and the
synapse's reversal potential agree, so the saved ground truth matches the wired
graph exactly.
"""

from neuron import h

from .neurons import build_cell, load_mechanisms
from .noise import add_poisson_noise


class Network:
    """Container for an instantiated NEURON network and its provenance.

    Args:
        cells: List of :class:`neuron_simulation.neurons.Cell` objects, indexed
            by neuron id.
        synapses: List of point-process synapse objects (one per edge).
        netcons: List of recurrent ``NetCon`` objects (one per edge).
        noise: List of :class:`neuron_simulation.noise.PoissonNoise` generators.
        topology: The topology dict the network was built from.
        config: The resolved build configuration (for metadata/reproducibility).

    Returns:
        An initialized ``Network``.
    """

    def __init__(self, cells, synapses, netcons, noise, topology, config):
        self.cells = cells
        self.synapses = synapses
        self.netcons = netcons
        self.noise = noise
        self.topology = topology
        self.config = config

    @property
    def n_neurons(self):
        """Number of neurons in the network."""
        return len(self.cells)

    @property
    def n_synapses(self):
        """Number of recurrent synapses (directed edges)."""
        return len(self.synapses)

    def apply_state(self, gbar_kA_exc, gbar_kA_inh=None):
        """Set the A-current density on every cell (used to switch states).

        Args:
            gbar_kA_exc: A-current density (S/cm2) for excitatory cells.
            gbar_kA_inh: A-current density for inhibitory cells; defaults to
                ``gbar_kA_exc`` when omitted.

        Returns:
            None. Each cell's ``kA.gbar`` is updated in place.
        """
        if gbar_kA_inh is None:
            gbar_kA_inh = gbar_kA_exc
        for cell in self.cells:
            cell.set_gbar_kA(gbar_kA_inh if cell.is_inhibitory else gbar_kA_exc)


def build_network(
    topology,
    celsius=6.3,
    gbar_kA_exc=0.006,
    gbar_kA_inh=0.004,
    depression=True,
    depression_d=0.5,
    tau_d=800.0,
    exc_tau=3.0,
    inh_tau=6.0,
    e_inh=-75.0,
    syn_delay=1.5,
    spike_threshold=0.0,
    exc_weight_scale=4.0,
    inh_weight_scale=1.5,
    noise_rate=2.0,
    noise_weight=0.0008,
    noise_tau=3.0,
    noise_seed=1000,
    tau_k=200.0,
    kA_globals=None,
):
    """Instantiate a NEURON network from a topology dict.

    Args:
        topology: Topology dict from :mod:`neuron_simulation.topology`.
        celsius: Global simulation temperature (degC). 6.3 keeps NEURON's squid
            ``hh`` kinetics; ~34 gives a faster mammalian-like variant.
        gbar_kA_exc: A-current density (S/cm2) for excitatory cells (the 4-AP
            knob for the excitatory population).
        gbar_kA_inh: A-current density (S/cm2) for inhibitory cells.
        depression: Whether excitatory synapses use short-term depression. When
            ``False`` the depression fraction is forced to 0 (static synapses).
        depression_d: Per-spike depression fraction for excitatory ``DepSyn``.
        tau_d: Resource-recovery time constant (ms) for excitatory ``DepSyn``.
        exc_tau: Excitatory conductance decay time constant (ms).
        inh_tau: Inhibitory conductance decay time constant (ms).
        e_inh: Inhibitory reversal potential (mV).
        syn_delay: Recurrent synaptic delay (ms).
        spike_threshold: Voltage threshold (mV) for spike detection and NetCon
            event triggering.
        exc_weight_scale: Multiplier applied to every excitatory NetCon weight
            (a global gain on recurrent excitation; the LIF analogue of
            ``scale_excitatory_weights``).
        inh_weight_scale: Multiplier applied to every inhibitory NetCon weight.
        noise_rate: Per-neuron Poisson background rate (Hz).
        noise_weight: Background synaptic weight (uS).
        noise_tau: Background excitatory conductance decay (ms).
        noise_seed: Base seed for the per-neuron Poisson streams.
        tau_k: Extracellular-K+ clearance time constant (ms) written to every
            cell's ``kdyn`` mechanism -- the SEIZURE knob. Small (~200 ms) = strong
            glial buffering -> [K+]o stays ~4 mM -> discrete bursts; large
            (~2500 ms) = impaired clearance -> [K+]o accumulates -> seizure.
            Ignored if ``kdyn`` is not inserted.
        kA_globals: Optional mapping of ``kA`` GLOBAL parameter names (without
            the ``_kA`` suffix, e.g. ``{"vhalfm": -50}``) to values, applied
            before building cells. Lets callers tune A-current kinetics without
            recompiling the mechanism.

    Returns:
        An initialized :class:`Network`.
    """
    load_mechanisms()
    h.celsius = float(celsius)

    if kA_globals:
        for name, value in kA_globals.items():
            setattr(h, "%s_kA" % name, float(value))

    is_inh = topology["neuron_is_inhibitory"]
    cluster_assignments = topology["cluster_assignments"]
    connections = topology["connections"]
    n_neurons = topology["n_neurons"]

    # --- cells ---
    cells = []
    for gid in range(n_neurons):
        inhibitory = bool(is_inh[gid])
        cell = build_cell(
            gid,
            is_inhibitory=inhibitory,
            gbar_kA=gbar_kA_inh if inhibitory else gbar_kA_exc,
            cluster_id=int(cluster_assignments[gid]),
            spike_threshold=spike_threshold,
        )
        # Extracellular-K+ clearance rate (the seizure knob). Guarded so this is
        # a no-op if the kdyn mechanism is not inserted (e.g. before the upgrade).
        try:
            cell.soma(0.5).kdyn.tau_k = float(tau_k)
        except AttributeError:
            pass
        cells.append(cell)

    # --- recurrent synapses (one point process + NetCon per edge) ---
    effective_d = float(depression_d) if depression else 0.0
    synapses = []
    netcons = []
    for row in connections:
        pre_id, post_id, weight, conn_type = int(row[0]), int(row[1]), float(row[2]), str(row[3])
        post = cells[post_id]
        if conn_type == "exc":
            syn = h.DepSyn(post.soma(0.5))
            syn.tau = exc_tau
            syn.e = 0.0
            syn.d = effective_d
            syn.tau_d = tau_d
        else:
            syn = h.ExpSyn(post.soma(0.5))
            syn.tau = inh_tau
            syn.e = e_inh

        scale = exc_weight_scale if conn_type == "exc" else inh_weight_scale
        nc = h.NetCon(cells[pre_id].soma(0.5)._ref_v, syn, sec=cells[pre_id].soma)
        nc.threshold = spike_threshold
        nc.delay = syn_delay
        nc.weight[0] = abs(weight) * scale  # NetCon weight is a positive conductance

        post.recurrent_synapses.append(syn)
        post.netcons.append(nc)
        synapses.append(syn)
        netcons.append(nc)

    # --- background noise (sole drive) ---
    noise = add_poisson_noise(
        cells, rate_hz=noise_rate, weight=noise_weight, tau=noise_tau, base_seed=noise_seed
    )

    config = {
        "celsius": float(celsius),
        "gbar_kA_exc": float(gbar_kA_exc),
        "gbar_kA_inh": float(gbar_kA_inh),
        "depression": bool(depression),
        "depression_d": effective_d,
        "tau_d": float(tau_d),
        "exc_tau": float(exc_tau),
        "inh_tau": float(inh_tau),
        "e_inh": float(e_inh),
        "syn_delay": float(syn_delay),
        "spike_threshold": float(spike_threshold),
        "exc_weight_scale": float(exc_weight_scale),
        "inh_weight_scale": float(inh_weight_scale),
        "noise_rate": float(noise_rate),
        "noise_weight": float(noise_weight),
        "noise_tau": float(noise_tau),
        "noise_seed": int(noise_seed),
        "tau_k": float(tau_k),
        "kA_globals": dict(kA_globals) if kA_globals else {},
    }
    print(
        f"Built network: {n_neurons} cells, {len(synapses)} synapses, "
        f"celsius={celsius}, gbar_kA(exc)={gbar_kA_exc}, depression_d={effective_d}"
    )
    return Network(cells, synapses, netcons, noise, topology, config)
