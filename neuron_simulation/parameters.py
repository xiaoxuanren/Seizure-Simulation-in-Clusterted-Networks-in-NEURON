"""Single source of truth for every simulation parameter.

Why this exists
---------------
The operating point used to live in four places (``build_network`` defaults,
two notebooks, and a study script) and no two agreed. This module holds the
canonical value, units, meaning, and *effect of increasing* for every parameter,
and three consumers read from it so they cannot drift apart:

* :func:`defaults` -> the library's default arguments,
* :func:`document` -> the ``parameters`` block written into every saved session,
* :func:`markdown_table` -> the notebook's parameter table (rendered live).

The defaults here ARE the flagship operating point: the 926-cell seed-1 network
whose 50 x 60 s session measures 0.279 Hz mean firing rate.

Groups
------
``topology``  wiring/ground truth (what the inference must recover)
``weights``   synaptic weight ranges (part of the topology output)
``build``     biophysics of the instantiated network
``sim``       integration and recording

``confidence``
--------------
``"documented"`` -- the effect is stated in the project's own docs/tuning table
or follows directly from the mechanism.
``"verify"``     -- the direction is inferred and should be confirmed empirically
before being quoted in a paper.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    """One documented simulation parameter."""

    name: str
    default: Any
    units: str
    group: str
    description: str
    effect_of_increasing: str
    confidence: str = "documented"


def _p(*args, **kwargs):
    return Param(*args, **kwargs)


# --------------------------------------------------------------------------- #
# TOPOLOGY -- the wiring. This IS the ground truth inference is scored against.
# --------------------------------------------------------------------------- #
_TOPOLOGY = [
    _p("num_clusters", 50, "count", "topology",
       "Number of spatial clusters REQUESTED. The packer drops clusters that will not fit in "
       "space_size, so the realised count can be lower (50 requested -> 44 placed at the "
       "flagship settings; N=926).",
       "Larger network and more modular wiring, but only until the space saturates -- past that "
       "the extra clusters are silently dropped. Raise space_size together with it."),
    _p("neurons_per_cluster_range", (4, 40), "count", "topology",
       "(min, max) neurons drawn per cluster; N emerges from this and the seed.",
       "Raises N and widens cluster-size heterogeneity (hub-like large clusters)."),
    _p("inhibitory_probability", 0.2, "fraction", "topology",
       "Probability a neuron is inhibitory (sets the E/I ratio and Dale labels).",
       "More inhibition -> lower firing, harder burst ignition; also more true inh edges to recover.",
       "verify"),
    _p("cluster_radius", 1.0, "space units", "topology",
       "Scatter radius of neurons around their cluster centre.",
       "Looser clusters -> more within/between distance overlap, weaker spatial structure.",
       "verify"),
    _p("space_size", 15.0, "space units", "topology",
       "Side length of the square the clusters are embedded in.",
       "Spreads the network out -> longer distance-dependent delays, sparser connections."),
    _p("seed", 1, "int", "topology",
       "Topology RNG seed. Fixes N, positions, E/I labels, and the exact wiring.",
       "Different seed = a different network and different ground truth (not a magnitude knob)."),
    _p("decay_sigma", 3.0, "space units", "topology",
       "Gaussian width of the distance kernel: P(connect) *= exp(-d^2 / 2*sigma^2).",
       "Longer-range connectivity (less local); lower = tighter, more local wiring."),
    _p("max_connection_distance", 6.0, "space units", "topology",
       "Hard cutoff: no edges beyond this distance.",
       "Permits longer-range edges; lower = harder enforced locality."),
    _p("cell_type_specific", True, "bool", "topology",
       "Use separate E->E / E->I / I->E / I->I probabilities instead of one global pair.",
       "When True the p_xx_* parameters below take over from within/between_cluster_prob."),
    _p("p_ee_within", 0.2, "probability", "topology",
       "E->E connection probability within a cluster.",
       "More recurrent excitation -> easier burst ignition, higher firing, denser true graph."),
    _p("p_ee_between", 0.1, "probability", "topology",
       "E->E connection probability across clusters.",
       "More long-range excitation -> bursts propagate network-wide, higher participation."),
    _p("p_ei_within", 0.20, "probability", "topology",
       "E->I connection probability within a cluster (excitation onto interneurons).",
       "Recruits inhibition faster -> shorter, better-terminated bursts.", "verify"),
    _p("p_ei_between", 0.08, "probability", "topology",
       "E->I connection probability across clusters.",
       "Spreads feedback inhibition between clusters -> more global burst control.", "verify"),
    _p("p_ie_within", 0.40, "probability", "topology",
       "I->E connection probability within a cluster (the main inhibitory brake).",
       "Stronger brake -> lower firing, sparser bursts; also more true inh edges."),
    _p("p_ii_within", 0.50, "probability", "topology",
       "I->I connection probability within a cluster.",
       "More disinhibition among interneurons -> can paradoxically raise E firing.", "verify"),
    _p("within_cluster_prob", 0.25, "probability", "topology",
       "Global within-cluster connection probability (ONLY used when cell_type_specific=False).",
       "Denser intra-cluster wiring -> stronger clustering, more local synchrony."),
    _p("between_cluster_prob", 0.06, "probability", "topology",
       "Global across-cluster probability (ONLY used when cell_type_specific=False).",
       "Denser inter-cluster wiring -> more global synchrony, less modularity."),
    _p("ln_sigma", 0.5, "log units", "topology",
       "Spread of the log-normal connection-propensity distribution across neurons.",
       "Heavier-tailed degree distribution -> a few strong hubs dominate the dynamics."),
    _p("target_density", None, "fraction or None", "topology",
       "If set, forces overall connection density; None lets it emerge from the probabilities.",
       "Pinning density overrides the probability knobs (they then only shape relative structure)."),
]

# --------------------------------------------------------------------------- #
# WEIGHTS -- NeuronWeightParameters; part of the saved topology.
# --------------------------------------------------------------------------- #
_WEIGHTS = [
    _p("within_exc_range", (0.0010, 0.0022), "uS", "weights",
       "Excitatory peak-conductance range, same cluster.",
       "Stronger local excitation -> easier ignition, higher firing; too high = runaway."),
    _p("between_exc_range", (0.0008, 0.0016), "uS", "weights",
       "Excitatory peak-conductance range, across clusters.",
       "Stronger long-range excitation -> bursts recruit the whole network."),
    _p("within_inh_range", (0.0025, 0.0055), "uS", "weights",
       "Inhibitory peak-conductance range, same cluster.",
       "Stronger local inhibition -> lower firing, crisper burst termination."),
    _p("between_inh_range", (0.0020, 0.0040), "uS", "weights",
       "Inhibitory range across clusters (unused: inhibition is within-cluster only).",
       "No effect in the current topology (inhibition does not project between clusters)."),
    _p("use_lognormal", True, "bool", "weights",
       "Draw weights log-normally within each range instead of uniformly.",
       "Produces realistic heavy-tailed synaptic strengths (a few very strong synapses)."),
    _p("lognormal_sigma", 0.5, "log units", "weights",
       "Spread of the within-range weight log-normal.",
       "Wider weight spread -> stronger synapses are easier for inference to recover, weak ones harder."),
]

# --------------------------------------------------------------------------- #
# BUILD -- biophysics of the instantiated network.
# --------------------------------------------------------------------------- #
_BUILD = [
    # --- the seizure knob ---
    _p("sahp_ainc_slow", 0.01, "uS", "build",
       "THE SEIZURE KNOB. Ca2+-dependent slow-AHP (KCa-type) per-spike conductance increment. "
       "NOT the M-current -- that is sahp_ainc_fast. Normal = 0.01; any LOWER value is a seizure state.",
       "More slow adaptation -> lower firing rate, sparser bursts, longer inter-burst interval. "
       "Lowering it models an acquired-epilepsy sAHP deficit, KCa3.1-like "
       "(0.01 -> 0.29 Hz; 0.004 -> 0.63 Hz). See states.py for the channel-identity argument."),
    _p("sahp_tau_slow", 6500.0, "ms", "build",
       "Decay time constant of the slow AHP; sets the multi-second inter-burst interval. "
       "NOTE: 6500 ms sits above both the measured I_sAHP range (1-5 s; ~2.9 s typical) and "
       "sAHP.mod's own documented 1000-4000 ms -- a tuning choice, not a measured value.",
       "Longer-lasting adaptation -> longer inter-burst intervals, slower burst rate."),
    _p("sahp_ainc_fast", 0.005, "uS", "build",
       "Fast spike-frequency-adaptation increment -- the M-current / Kv7 (KCNQ2/3)-like "
       "component, and so the one a KCNQ channelopathy would move. Held FIXED across "
       "normal/seizure by design.",
       "Fewer spikes per burst (thinner bursts) without changing the inter-burst interval."),
    _p("sahp_tau_fast", 300.0, "ms", "build",
       "Decay time constant of the fast SFA component.",
       "Adaptation carries further between bursts -> suppresses inter-burst background firing."),
    _p("sahp_ek", -90.0, "mV", "build",
       "Reversal potential of both adaptation conductances.",
       "Less negative -> weaker hyperpolarizing drive, so adaptation bites less."),
    _p("adapt", True, "bool", "build",
       "Whether cells carry the two-timescale sAHP at all.",
       "False removes the seizure knob entirely; the network loses its burst pacer."),
    # --- excitatory synapses ---
    _p("synapse_model", "ampa_nmda", "str", "build",
       "Excitatory synapse model: 'ampa_nmda' (AMPA+NMDA) or 'depsyn' (single fast exponential).",
       "'ampa_nmda' adds slow NMDA reverberation and lengthens bursts to the culture range."),
    _p("exc_tau", 5.0, "ms", "build",
       "AMPA (fast excitatory) conductance decay time constant.",
       "Longer excitatory pulses -> more temporal summation, easier ignition."),
    _p("tau_nmda", 350.0, "ms", "build",
       "NMDA (slow excitatory) decay time constant; carries burst reverberation.",
       "Longer bursts via sustained reverberation; too long = bursts merge into continuous firing."),
    _p("nmda_ratio", 3.0, "ratio", "build",
       "NMDA/AMPA peak-conductance ratio.",
       "Longer, more self-sustaining bursts (stronger reverberation)."),
    _p("exc_weight_scale", 2.0, "multiplier", "build",
       "Global gain on every excitatory NetCon weight.",
       "More excitable network -> higher firing, denser bursts; too high = runaway/continuous firing."),
    _p("depression", True, "bool", "build",
       "Whether excitatory synapses use short-term depression (the burst terminator).",
       "False forces depression_d=0 (static synapses); at the default gain the network runs away."),
    _p("depression_d", 0.2, "fraction", "build",
       "Per-spike depletion fraction of the synaptic resource (Tsodyks-Markram).",
       "Stronger depression -> bursts self-terminate sooner, lower sustained firing."),
    _p("tau_d", 500.0, "ms", "build",
       "Recovery time constant of the depleted synaptic resource.",
       "Slower recovery -> more accumulated depression, longer enforced quiet after a burst."),
    # --- inhibitory synapses ---
    _p("inh_weight_scale", 2.5, "multiplier", "build",
       "Global gain on every inhibitory NetCon weight.",
       "Stronger brake -> lower firing, sparser bursts; lowering it is the classic disinhibition route."),
    _p("inh_tau", 6.0, "ms", "build",
       "Inhibitory (GABA-A) conductance decay time constant.",
       "Longer-lasting inhibition -> stronger burst suppression.", "verify"),
    _p("e_inh", -75.0, "mV", "build",
       "Inhibitory reversal potential.",
       "Depolarizing it weakens inhibition; above rest it becomes excitatory (a seizure mechanism)."),
    # --- connectivity timing ---
    _p("syn_delay", 1.5, "ms", "build",
       "Baseline recurrent synaptic delay.",
       "Slower signal propagation -> looser spike timing; widens the monosynaptic inference window."),
    _p("delay_per_distance", 2.0, "ms / space unit", "build",
       "Extra conduction delay per unit distance between pre and post.",
       "Longer bursts via spatial propagation; also spreads the true lag across edges, "
       "which matters for the GLM's lag readout."),
    _p("spike_threshold", 0.0, "mV", "build",
       "Voltage threshold for spike detection and NetCon triggering.",
       "Higher -> fewer detected spikes; must stay consistent between simulation and analysis."),
    # --- background drive ---
    _p("noise_rate", 5.0, "Hz", "build",
       "Per-neuron independent Poisson background rate (the network's SOLE drive).",
       "More background -> higher inter-burst firing; below a floor the network is silent."),
    _p("noise_weight", 0.004, "uS", "build",
       "Peak conductance of one background event.",
       "Stronger ignition seed -> more frequent bursts, more background firing."),
    _p("noise_tau", 3.0, "ms", "build",
       "Background excitatory conductance decay.",
       "Longer background pulses -> more summation from the same event rate."),
    _p("noise_seed", 1000, "int", "build",
       "Base seed for per-neuron Poisson streams; combined with gid and recording index.",
       "Different seed = a different noise realization (not a magnitude knob)."),
    # --- intrinsic membrane ---
    _p("celsius", 6.3, "degC", "build",
       "Simulation temperature. 6.3 keeps NEURON's squid hh kinetics; ~34 is mammalian-like.",
       "Faster channel kinetics -> narrower spikes, higher achievable firing rates."),
    _p("gbar_kA_exc", 0.006, "S/cm2", "build",
       "A-type (Kv4) K+ current density in excitatory cells.",
       "More transient outward current -> delayed spike onset, crisper bursts."),
    _p("gbar_kA_inh", 0.004, "S/cm2", "build",
       "A-current density in inhibitory cells (weaker, so they recruit slightly earlier).",
       "Delays interneuron recruitment -> weaker/later feedback inhibition."),
    _p("tau_k", 200.0, "ms", "build",
       "Extracellular K+ clearance time constant (kdyn). HELD FIXED at 200 in the single-knob model.",
       "Impaired clearance -> [K+]o accumulates -> E_K depolarizes -> the separate high-[K+]o "
       "ictal route (NOT this project's seizure knob)."),
    _p("kA_globals", None, "dict or None", "build",
       "Optional overrides for kA kinetics GLOBALs (e.g. {'vhalfm': -50}).",
       "Retunes A-current kinetics without recompiling the mechanism."),
]

# --------------------------------------------------------------------------- #
# SIM -- integration and recording.
# --------------------------------------------------------------------------- #
_SIM = [
    _p("dt", 0.05, "ms", "sim",
       "Fixed integration time step.",
       "Larger = faster but less accurate; the HH cells become unstable if raised too far."),
    _p("recording_duration", 60000.0, "ms", "sim",
       "Length of the KEPT recording, after the discarded transient.",
       "More data per recording -> better-conditioned inference (more spikes per edge)."),
    _p("n_recordings", 50, "count", "sim",
       "Number of independent recordings (trials) per session; each is a fresh noise reseed.",
       "More trials -> more total data and better inference; the wiring is identical across them."),
    _p("discard_transient_ms", 1000.0, "ms", "sim",
       "Startup transient dropped from the front of each run; the clock is then re-zeroed.",
       "Discards more startup artefact at the cost of usable data."),
    _p("v_init", -65.0, "mV", "sim",
       "Initial membrane voltage passed to finitialize.",
       "Mostly affects the discarded transient, not the steady state."),
    _p("participation_threshold", 0.35, "fraction", "sim",
       "Fraction of neurons that must fire in an event window to count as a NETWORK BURST. "
       "0.35 is the project's 'loose burst' definition: these events are low-participation, "
       "so a >80% detector misses them entirely.",
       "Stricter -> fewer events qualify (at 0.8 most loose bursts vanish). Labels the saved "
       "burst_windows only; it does not change the simulated spike data."),
    _p("target_freq", 10, "Hz", "sim",
       "Resampling frequency for the saved raster (resampled_spikes).",
       "Finer saved raster; does not affect spike_times, which inference actually uses."),
    _p("record_voltage", False, "bool", "sim",
       "Record downsampled somatic voltage per neuron.",
       "Enables voltage-augmented inference at a large file-size cost (~222 MB/recording)."),
    _p("voltage_dt", 1.0, "ms", "sim",
       "Voltage sampling interval (must be >= dt).",
       "Coarser voltage traces, smaller files."),
    _p("record_ko", False, "bool", "sim",
       "Record the kdyn extracellular [K+]o per neuron.",
       "Enables the [K+]o traces used in the seizure plots; small cost."),
    _p("ko_dt", 5.0, "ms", "sim",
       "[K+]o sampling interval.",
       "Coarser [K+]o traces, smaller files."),
    _p("noise_seed_base", 1000, "int", "sim",
       "Session-level base seed; each recording keys Random123(base, gid, recording_index).",
       "Different seed = different noise across the whole session (not a magnitude knob)."),
]

PARAMETERS = {p.name: p for p in (_TOPOLOGY + _WEIGHTS + _BUILD + _SIM)}
GROUPS = ("topology", "weights", "build", "sim")


# --------------------------------------------------------------------------- #
# Consumers
# --------------------------------------------------------------------------- #
def defaults(group=None):
    """Return ``{name: default}`` for one group (or all groups when ``None``)."""
    return {p.name: p.default for p in PARAMETERS.values()
            if group is None or p.group == group}


def describe(name):
    """Return the :class:`Param` record for ``name``."""
    return PARAMETERS[name]


def document(resolved, include_missing=False):
    """Annotate a ``{name: value}`` mapping with units/description/effect.

    This is what gets written into ``session_metadata.json`` so a saved session
    explains its own parameters.

    Args:
        resolved: The actual values used for a run.
        include_missing: Also emit documented parameters absent from ``resolved``
            (at their registry default).

    Returns:
        ``{name: {value, units, group, description, effect_of_increasing,
        is_default}}``. Unknown keys are recorded with a note rather than dropped.
    """
    out = {}
    for name, value in resolved.items():
        p = PARAMETERS.get(name)
        if p is None:
            out[name] = {"value": value, "units": "?", "group": "unregistered",
                         "description": "Not in the parameter registry.",
                         "effect_of_increasing": "unknown", "is_default": None}
            continue
        out[name] = {"value": value, "units": p.units, "group": p.group,
                     "description": p.description,
                     "effect_of_increasing": p.effect_of_increasing,
                     "confidence": p.confidence,
                     "is_default": value == p.default}
    if include_missing:
        for name, p in PARAMETERS.items():
            out.setdefault(name, {"value": p.default, "units": p.units, "group": p.group,
                                  "description": p.description,
                                  "effect_of_increasing": p.effect_of_increasing,
                                  "confidence": p.confidence, "is_default": True})
    return out


def markdown_table(group=None, include_effect=True):
    """Render the registry as a markdown table (for the notebook)."""
    groups = (group,) if group else GROUPS
    lines = []
    for g in groups:
        rows = [p for p in PARAMETERS.values() if p.group == g]
        if not rows:
            continue
        lines.append(f"\n**{g.upper()}**\n")
        head = "| parameter | default | units | what it is |"
        sep = "|---|---|---|---|"
        if include_effect:
            head += " effect of increasing |"
            sep += "---|"
        lines += [head, sep]
        for p in rows:
            row = f"| `{p.name}` | `{p.default}` | {p.units} | {p.description} |"
            if include_effect:
                flag = " *(verify)*" if p.confidence == "verify" else ""
                row += f" {p.effect_of_increasing}{flag} |"
            lines.append(row)
    return "\n".join(lines)


def verify_signature(func, group):
    """Return ``{name: (signature_default, registry_default)}`` for drifted params.

    Lets a test fail loudly if a function signature drifts from the registry.
    """
    import inspect
    drift = {}
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name not in PARAMETERS or PARAMETERS[name].group != group:
            continue
        if param.default is inspect.Parameter.empty:
            continue
        if param.default != PARAMETERS[name].default:
            drift[name] = (param.default, PARAMETERS[name].default)
    return drift


def needs_verification():
    """Names whose effect direction is inferred and should be confirmed."""
    return sorted(n for n, p in PARAMETERS.items() if p.confidence == "verify")


def verify_source_defaults(path, func_name, group, ignore=()):
    """Drift-check a function's defaults by PARSING its source (no import).

    Unlike :func:`verify_signature` this needs neither NEURON nor any runtime
    dependency, so it works in CI. Returns ``{name: (source_default,
    registry_default)}`` for parameters in ``group`` whose signature default no
    longer matches the registry.

    Args:
        path: Path to the Python file containing the function.
        func_name: Name of the function whose signature to check.
        group: Registry group to check against (``topology``/``build``/``sim``).
        ignore: Parameter names to skip.

    Returns:
        A dict of drifted parameters; empty when the signature is in sync.
    """
    import ast

    tree = ast.parse(open(path, encoding="utf-8").read())
    func = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if func is None:
        raise LookupError(f"{func_name} not found in {path}")

    args = func.args.args + func.args.kwonlyargs
    defs = list(func.args.defaults) + [d for d in func.args.kw_defaults if d is not None]
    pairs = dict(zip([a.arg for a in args[len(args) - len(defs):]], defs))

    drift = {}
    for name, node in pairs.items():
        if name in ignore or name not in PARAMETERS:
            continue
        p = PARAMETERS[name]
        if p.group != group:
            continue
        try:
            value = ast.literal_eval(node)
        except (ValueError, SyntaxError):
            continue  # non-literal default (e.g. a name); nothing to compare
        if value != p.default:
            drift[name] = (value, p.default)
    return drift


#: The functions whose signatures must stay in sync with the registry.
SYNCED_SIGNATURES = (
    ("neuron_simulation/network_builder.py", "build_network", "build"),
    ("neuron_simulation/topology.py", "build_topology_lognormal", "topology"),
    ("neuron_simulation/simulation.py", "run_simulation", "sim"),
    ("neuron_simulation/workflows.py", "generate_dataset", "sim"),
    ("neuron_simulation/parallel_dataset.py", "generate_dataset_parallel", "sim"),
)


# --------------------------------------------------------------------------- #
# Runtime deviation reporting (informational -- never blocks a run)
# --------------------------------------------------------------------------- #
def deviations(resolved, group=None):
    """Return ``{name: (value, registry_default)}`` for values that differ.

    Purely informational: this reports what a run is doing differently from the
    canonical operating point. It never blocks or alters a run -- exploring
    non-default settings is expected and normal.

    Args:
        resolved: The actual ``{name: value}`` used for a run.
        group: Optionally restrict to one registry group.

    Returns:
        A dict of deviating parameters (empty when everything is canonical).
        Unregistered keys are ignored.
    """
    out = {}
    for name, value in resolved.items():
        p = PARAMETERS.get(name)
        if p is None or (group is not None and p.group != group):
            continue
        if value != p.default:
            out[name] = (value, p.default)
    return out


def format_deviations(resolved, group=None, prefix="  ", max_items=None):
    """One-line human-readable summary of :func:`deviations`, or ``""``.

    Args:
        resolved: The actual ``{name: value}`` used for a run.
        group: Optionally restrict to one registry group.
        prefix: String prepended to the line.
        max_items: Truncate after this many entries (``None`` = show all).

    Returns:
        A string like ``"  non-default: sahp_ainc_slow=0.004 (default 0.01)"``,
        or an empty string when nothing deviates.
    """
    dev = deviations(resolved, group=group)
    if not dev:
        return ""
    items = sorted(dev.items())
    shown = items if max_items is None else items[:max_items]
    parts = [f"{k}={v!r} (default {d!r})" for k, (v, d) in shown]
    if max_items is not None and len(items) > max_items:
        parts.append(f"+{len(items) - max_items} more")
    return f"{prefix}non-default: " + ", ".join(parts)
