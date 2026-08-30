"""Pharmacological / pathophysiological states.

The seizure mechanism in this project is a **two-parameter knob**, both
parameters on the Ca2+-dependent slow AHP. One fixed network, two parameters,
two phenotypes:

* ``normal_state``  -> ``sahp_ainc_slow = 0.01``, ``sahp_tau_slow = 6500 ms``
  (strong, long-lasting slow adaptation; quiet, sparse loose bursts; [K+]o
  stays ~4 mM). PINNED: these are what the shipped flagship session used.
* ``seizure_state(ainc, tau)`` -> **lower** ``sahp_ainc_slow`` and/or
  **shorter** ``sahp_tau_slow``. The shipped seizure datasets use
  ``0.004 uS + 3000 ms``.

THE TWO PARAMETERS DO DIFFERENT THINGS (measured, 20 networks, 2026-08):

    sahp_ainc_slow  DEPTH of adaptation -> RECRUITMENT.  0.010 -> 0.004 takes
                    burst participation 0.51 -> 0.95 and firing rate 5.8x up,
                    and erases the topology-dependent participation spread
                    (normal 0.27-0.91 across networks -> seizure 0.90-0.99).
    sahp_tau_slow   DURATION of adaptation -> RECOVERY CLOCK, i.e. burst
                    FREQUENCY, scaling roughly as 6500/tau. 6500 -> 3000 ms
                    takes ~11 -> ~24 bursts/min with participation unchanged.

Measured normal -> seizure across all 20 networks (paired noise streams):
firing rate 0.30 -> 1.72 Hz (5.8x), bursts 9.4 -> 25.5 per 60 s recording
(2.7x), participation 0.51 -> 0.95, burst duration 149 -> 442 ms (3.0x).

NOT 4-AP: blocking the A-current (4-AP's only target in this model) was
measured across 0-9 mM equivalent on 20 networks and left every phenotype
axis flat -- see analysis/fourap_dose_response.py and the ladder figures.
NOT the K+-clearance ictal route either: raising tau_k SUPPRESSES bursting
(participation 0.95 -> 0.16 as [K+]o reaches 12-16 mM), it does not lengthen
events -- see analysis/tauk_ictal_ladder.py. The phenotype produced here
matches the 4-AP CULTURE/MEA signature (higher rate, more frequent and more
synchronous bursts) but not the 4-AP SLICE ictal regime (31-103 s
discharges); this model's events top out below 1 s.

CHANNEL IDENTITY -- read this before citing the knob in a talk or paper.
``sahp_ainc_slow`` has ``tau_slow = 6500 ms``, which by both kinetics and
pharmacology is the **Ca2+-dependent slow AHP** (a KCa conductance), NOT the
M-current. The literature splits the AHP by timescale: the mAHP lasts 50-100 ms
and carries the Kv7/KCNQ (M-current) component, whose gating runs on tens of
milliseconds; the Ca2+-dependent sAHP lasts 1-5 s (measured I_sAHP decay
tau ~2.9 s) and survives apamin, XE-991 (Kv7 block) and Cs+. In this model that
split maps onto the two components exactly as ``sAHP.mod`` documents them:
``sahp_ainc_fast`` (tau 300 ms) is the M-current/Kv7-like term, and
``sahp_ainc_slow`` is the Ca2+-dependent sAHP.

So lowering the knob models an **acquired-epilepsy sAHP deficit** -- the
KCa3.1-like reduction seen in post-status-epilepticus hippocampus, where
suppression of the sAHP is the main cause of augmented spike output
(Tamir et al. 2017; Tiwari et al. 2019 J Neurosci 39:9914; CRF/CRF1R route,
J Neurosci 42:5843, 2022; KCa3.1 in L5 neocortex, Roshchin et al. 2020
Sci Rep 10:14484). It does NOT model KCNQ2/3 loss-of-function: that is a genetic
neonatal syndrome acting mainly through the mAHP, i.e. through
``sahp_ainc_fast``, which this project holds FIXED. (Earlier revisions of this
docstring said KCNQ2/3; that attribution was wrong. The one narrow link between
KCNQ and the sAHP -- Tzingounis & Nicoll 2008 PNAS 105:19974 -- is specific to
dentate granule cells and is not the mainline story.)

Either way this is the **adaptation-deficit** seizure -- the mild-[K+]o bursting
phenotype -- NOT the K+-clearance ictal route.

Everything else is held FIXED across the two states, by design:

* ``tau_k = 200 ms``          (K+ clearance; the kdyn route is NOT the knob here)
* ``sahp_ainc_fast = 0.005``  (fast SFA; unchanged between states)
* ``gbar_kA_exc/inh``         (A-current; unchanged between states)

Measured reference points (926-cell seed-1 network, 60 s; notebook cells 2/6/8):

    sahp_ainc_slow   firing rate   loose bursts   participation
    0.010 (normal)      0.29 Hz     8 (0.13 Hz)       0.93
    0.004               0.63 Hz    10 (0.17 Hz)       1.00

Lower knob -> more firing, monotonically. The shipped 50-minute flagship session
(``notebooks/NEURON data parallel/normal/20260721_163430``, 50 x 60 s) measures
0.279 Hz across all 50 recordings, confirming it is the NORMAL state at
``sahp_ainc_slow = 0.01``.

Two alternative knobs are retained but are NOT the project's seizure model:

* :func:`kclearance_seizure_state` -- the impaired-K+-clearance (``tau_k``)
  route, for the high-[K+]o ictal phenotype.
* :func:`gbar_block_state` -- the reduced-A-current ("4-AP") knob, which on the
  realistic log-normal topology does not faithfully reproduce seizure.

Each "state" is just a dict of keyword overrides for
:func:`neuron_simulation.network_builder.build_network`. State values are applied
as **defaults**: an explicitly-passed ``build_kwargs`` entry always wins.
"""

import numpy as np

#: Excitatory / inhibitory A-current density (identical in both states).
NORMAL_GBAR_KA_EXC = 0.006
NORMAL_GBAR_KA_INH = 0.004

#: THE SINGLE KNOB -- Ca2+-dependent slow-AHP per-spike increment (uS).
#: NOT the M-current: that is `sahp_ainc_fast` (tau 300 ms), held fixed.
#: See the module docstring for the channel-identity argument.
#: Normal is PINNED (it is what the shipped 50-min flagship was generated with).
NORMAL_SAHP_SLOW = 0.01
#: Only a DEFAULT for :func:`seizure_state`; any value below normal is a valid
#: seizure. Kept as a named constant so bare ``seizure_state()`` is reproducible.
DEFAULT_SEIZURE_SAHP_SLOW = 0.004
#: Back-compat alias.
SEIZURE_SAHP_SLOW = DEFAULT_SEIZURE_SAHP_SLOW

#: THE SECOND HALF OF THE KNOB -- slow-AHP decay time constant (ms), i.e. how
#: long each spike's brake lasts. It sets the RECOVERY CLOCK between population
#: events: measured burst rate scales as ~ NORMAL_SAHP_TAU_SLOW / tau, so
#: 6500 -> 3000 ms roughly doubles the burst rate (11 -> 24 /min) while leaving
#: participation untouched. `sahp_ainc_slow` sets recruitment DEPTH; this sets
#: burst FREQUENCY. Together they are the project's two-parameter seizure knob.
NORMAL_SAHP_TAU_SLOW = 6500.0
#: Only a DEFAULT for :func:`seizure_state`; any value below normal shortens
#: the recovery clock. The shipped seizure datasets used 3000 ms.
DEFAULT_SEIZURE_SAHP_TAU_SLOW = 3000.0

#: Held FIXED across states (the tuned operating base; see the notebooks).
BASE_SAHP_FAST = 0.005
BASE_TAU_K = 200.0

#: Back-compat alias: older code/docs referred to NORMAL_TAU_K.
NORMAL_TAU_K = BASE_TAU_K

#: Only used by the non-default K+-clearance route below.
KCLEARANCE_SEIZURE_TAU_K = 12000.0

#: Build-network keys a state dict may set. Shared by workflows.run_single_state,
#: workflows.generate_dataset and parallel_dataset.generate_dataset_parallel so
#: every entry point merges the same keys with the same precedence.
STATE_BUILD_KEYS = (
    "gbar_kA_exc",
    "gbar_kA_inh",
    "tau_k",
    "sahp_ainc_slow",
    "sahp_ainc_fast",
)


def normal_state():
    """Return the build overrides for the normal (healthy) state.

    Strong slow adaptation (``sahp_ainc_slow = 0.01``) keeps the network quiet:
    sparse loose bursts (~0.13 Hz) on a low-rate asynchronous baseline, [K+]o
    ~4 mM.

    Returns:
        A dict with the fixed A-current and ``tau_k``, the fixed fast-SFA
        increment, the normal ``sahp_ainc_slow``, and a ``state_name``.
    """
    return {
        "state_name": "normal",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": BASE_TAU_K,
        "sahp_ainc_fast": BASE_SAHP_FAST,
        "sahp_ainc_slow": NORMAL_SAHP_SLOW,
        "sahp_tau_slow": NORMAL_SAHP_TAU_SLOW,
    }


def _state(sahp_ainc_slow, state_name, sahp_tau_slow=NORMAL_SAHP_TAU_SLOW):
    """Build a state dict differing from normal ONLY in the two sAHP knobs."""
    return {
        "state_name": state_name,
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": BASE_TAU_K,
        "sahp_ainc_fast": BASE_SAHP_FAST,
        "sahp_ainc_slow": float(sahp_ainc_slow),
        "sahp_tau_slow": float(sahp_tau_slow),
    }


def seizure_state(sahp_ainc_slow=DEFAULT_SEIZURE_SAHP_SLOW,
                  sahp_tau_slow=DEFAULT_SEIZURE_SAHP_TAU_SLOW):
    """Return build overrides for a seizure state: the SAME network, TWO sAHP changes.

    THE SEIZURE KNOB IS TWO PARAMETERS, both on the Ca2+-dependent slow AHP:

      * ``sahp_ainc_slow`` (uS) -- the per-spike increment: how much brake each
        spike adds. Sets RECRUITMENT DEPTH. Lowering 0.010 -> 0.004 takes burst
        participation from ~0.5 to ~0.95 and firing rate ~5.8x up, and it
        ERASES the topology-dependent spread in participation (measured across
        20 networks: normal 0.27-0.91, seizure 0.90-0.99).
      * ``sahp_tau_slow`` (ms) -- the decay constant: how long that brake
        lasts. Sets the RECOVERY CLOCK, i.e. burst FREQUENCY, roughly as
        6500/tau. Shortening 6500 -> 3000 ms takes ~11 -> ~24 bursts/min with
        participation unchanged.

    The two are separable and were measured separately (2026-08 previews):
    ``ainc`` alone gives full-recruitment bursts at the original ~11/min
    rhythm; ``tau`` alone is the frequency dial. The shipped seizure datasets
    use BOTH (0.004 uS + 3000 ms).

    Biologically this is an acquired sAHP deficit -- reduced Ca2+-dependent K+
    (KCa) conductance as reported in post-status-epilepticus hippocampus (see
    the module docstring; NOT a Kv7/KCNQ manipulation, and NOT a 4-AP model:
    A-current block was measured to leave this network's phenotype unchanged
    across 0-9 mM equivalent, 20 networks). Everything else (``tau_k``,
    ``sahp_ainc_fast``, ``gbar_kA_*``, topology, drive, noise streams) is
    identical to :func:`normal_state`, so any difference in the resulting
    activity is attributable to these two parameters.

    Args:
        sahp_ainc_slow: The slow-AHP per-spike increment (uS). Must be in
            ``[0, NORMAL_SAHP_SLOW)`` -- strictly weaker adaptation than
            normal. ``0`` removes the slow AHP entirely (most severe).
            Defaults to :data:`DEFAULT_SEIZURE_SAHP_SLOW` (0.004).
            For a relative depth, pass e.g. ``NORMAL_SAHP_SLOW * 0.4``.
        sahp_tau_slow: The slow-AHP decay constant (ms). Defaults to
            :data:`DEFAULT_SEIZURE_SAHP_TAU_SLOW` (3000). Pass
            :data:`NORMAL_SAHP_TAU_SLOW` (6500) to change ONLY the depth and
            keep the normal burst clock.

    Returns:
        A dict identical to :func:`normal_state` except for
        ``sahp_ainc_slow``, ``sahp_tau_slow`` and ``state_name`` (which embeds
        both values, so they land in the saved session metadata).

    Raises:
        ValueError: If ``sahp_ainc_slow`` is negative or not below
            :data:`NORMAL_SAHP_SLOW`.
    """
    value = float(sahp_ainc_slow)
    if value < 0.0:
        raise ValueError("sahp_ainc_slow must be >= 0.")
    if value >= NORMAL_SAHP_SLOW:
        raise ValueError(
            f"seizure_state() takes the sahp_ainc_slow VALUE (uS), not a severity. "
            f"Got {value:g}, which is not below normal ({NORMAL_SAHP_SLOW:g}) and so "
            f"is not a seizure. Legacy callers used seizure_state(1.0) for "
            f"'max severity' -- that now means sahp_ainc_slow=1.0. Use e.g. "
            f"seizure_state({DEFAULT_SEIZURE_SAHP_SLOW:g}) or "
            f"seizure_state(NORMAL_SAHP_SLOW * 0.4)."
        )
    tau = float(sahp_tau_slow)
    if tau <= 0.0:
        raise ValueError("sahp_tau_slow must be > 0 (ms).")
    name = f"seizure_sahp{value:g}"
    if tau != NORMAL_SAHP_TAU_SLOW:
        name += f"_tau{tau:g}"
    return _state(value, name, sahp_tau_slow=tau)


def seizure_dose_response(values=None, n_points=5,
                          min_sahp_slow=DEFAULT_SEIZURE_SAHP_SLOW):
    """Build a ``sahp_ainc_slow`` sweep for a dose-response curve.

    Args:
        values: Explicit sequence of ``sahp_ainc_slow`` values to sweep. When
            given, ``n_points``/``min_sahp_slow`` are ignored.
        n_points: Number of points when ``values`` is not supplied.
        min_sahp_slow: Seizure end of the default sweep (weakest adaptation);
            the sweep runs from here up to :data:`NORMAL_SAHP_SLOW`.

    Returns:
        A list of state dicts ordered from seizure toward normal.
    """
    if values is None:
        values = np.linspace(float(min_sahp_slow), NORMAL_SAHP_SLOW, int(n_points))
    return [_state(v, f"sahp_slow_{float(v):g}") for v in values]


# --------------------------------------------------------------------------- #
# Alternative (NON-default) seizure routes -- kept for comparison only.
# --------------------------------------------------------------------------- #
def kclearance_seizure_state(severity=1.0):
    """Return overrides for the impaired-K+-clearance seizure (NOT the default).

    The other mechanistic route: a larger ``tau_k`` lets firing-driven [K+]o
    accumulate, depolarizing E_K into a genuine high-[K+]o ictal state rather
    than the mild-[K+]o bursting phenotype of the single-knob model. Provided for
    comparison; it moves ``tau_k``, so it is NOT a single-knob state.

    Args:
        severity: ``0`` gives normal clearance; ``1.0`` gives the reference
            impaired value (``tau_k = 12000 ms``).

    Returns:
        A dict with the normal single-knob base and an elevated ``tau_k``.

    Raises:
        ValueError: If ``severity`` is negative.
    """
    if severity < 0.0:
        raise ValueError("severity must be >= 0.")
    tau_k = BASE_TAU_K + float(severity) * (KCLEARANCE_SEIZURE_TAU_K - BASE_TAU_K)
    state = normal_state()
    state["state_name"] = f"kclearance_seizure_sev{severity:g}"
    state["tau_k"] = float(tau_k)
    return state


def gbar_block_state(block_fraction=0.2):
    """Return overrides for a reduced-A-current state (DEPRECATED knob).

    Phenomenological only. On the realistic log-normal topology, changing
    ``gbar_kA`` does NOT faithfully reproduce the seizure phenotype (the dramatic
    reduced-A-current effect was specific to the dense discrete-hub topology).
    Use :func:`seizure_state` for the project's seizure model.

    Args:
        block_fraction: Fraction of the A-current removed, in ``[0, 1)``.

    Returns:
        A dict with reduced ``gbar_kA_exc``/``gbar_kA_inh`` on the normal base.

    Raises:
        ValueError: If ``block_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= block_fraction < 1.0):
        raise ValueError("block_fraction must be in [0, 1).")
    scale = 1.0 - block_fraction
    state = normal_state()
    state["state_name"] = f"gbar_block{int(round(block_fraction * 100))}"
    state["gbar_kA_exc"] = NORMAL_GBAR_KA_EXC * scale
    state["gbar_kA_inh"] = NORMAL_GBAR_KA_INH * scale
    return state


def four_ap_state(block_fraction=0.2):
    """Deprecated alias for :func:`gbar_block_state` (kept for back-compat).

    Args:
        block_fraction: Fraction of the A-current removed, in ``[0, 1)``.

    Returns:
        The dict returned by :func:`gbar_block_state`.
    """
    return gbar_block_state(block_fraction)


def dose_response_gbar(n_points=8, min_fraction=1.0, max_fraction=0.2):
    """Build a descending A-current-density sweep (DEPRECATED / phenomenological).

    Retained for the discrete-hub topology. Prefer :func:`seizure_dose_response`
    (the single ``sahp_ainc_slow`` knob) for the project's seizure model.

    Args:
        n_points: Number of gbar values in the sweep.
        min_fraction: Largest gbar as a fraction of normal (drug-free end).
        max_fraction: Smallest gbar as a fraction of normal (strong-block end).

    Returns:
        A list of state dicts ordered from drug-free toward strong block.
    """
    sweep = []
    for frac in np.linspace(min_fraction, max_fraction, n_points):
        state = normal_state()
        state["state_name"] = f"gbar_frac_{frac:.2f}"
        state["fraction"] = float(frac)
        state["gbar_kA_exc"] = NORMAL_GBAR_KA_EXC * float(frac)
        state["gbar_kA_inh"] = NORMAL_GBAR_KA_INH * float(frac)
        sweep.append(state)
    return sweep
