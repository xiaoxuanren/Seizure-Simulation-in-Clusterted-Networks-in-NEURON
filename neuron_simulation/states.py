"""Pharmacological / pathophysiological states.

The seizure mechanism in this project is a **single knob**: the slow-AHP /
M-current (Kv7/KCNQ) per-spike increment ``sahp_ainc_slow``. One fixed network,
one parameter, two phenotypes:

* ``normal_state``  -> ``sahp_ainc_slow = 0.01`` (strong slow adaptation;
  quiet, sparse loose bursts; [K+]o stays ~4 mM). This value is PINNED: it is the
  one the shipped 50-minute flagship session was generated with.
* ``seizure_state(value)`` -> **any LOWER** ``sahp_ainc_slow`` (weak slow
  adaptation; more firing, denser bursts). Seizure is defined as "less slow
  adaptation than normal", not as one blessed number -- pass whatever value you
  want. The default (0.004) is a convenience, not a commitment.

Lowering the knob models KCNQ2/3 loss-of-function; raising it models a Kv7
opener (retigabine). This is the **adaptation-deficit** seizure -- the
mild-[K+]o bursting phenotype -- NOT the K+-clearance ictal route.

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

#: THE SINGLE KNOB -- slow-AHP / M-current per-spike increment (uS).
#: Normal is PINNED (it is what the shipped 50-min flagship was generated with).
NORMAL_SAHP_SLOW = 0.01
#: Only a DEFAULT for :func:`seizure_state`; any value below normal is a valid
#: seizure. Kept as a named constant so bare ``seizure_state()`` is reproducible.
DEFAULT_SEIZURE_SAHP_SLOW = 0.004
#: Back-compat alias.
SEIZURE_SAHP_SLOW = DEFAULT_SEIZURE_SAHP_SLOW

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
    }


def _state(sahp_ainc_slow, state_name):
    """Build a state dict differing from normal ONLY in ``sahp_ainc_slow``."""
    return {
        "state_name": state_name,
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": BASE_TAU_K,
        "sahp_ainc_fast": BASE_SAHP_FAST,
        "sahp_ainc_slow": float(sahp_ainc_slow),
    }


def seizure_state(sahp_ainc_slow=DEFAULT_SEIZURE_SAHP_SLOW):
    """Return build overrides for a seizure state: the SAME network, lower slow AHP.

    Seizure here means only "less slow adaptation than normal" (KCNQ2/3 / Kv7
    loss-of-function). There is no blessed seizure number -- pass whatever value
    you want to study. Everything else (``tau_k``, ``sahp_ainc_fast``,
    ``gbar_kA_*``) is identical to :func:`normal_state`, so any difference in the
    resulting activity is attributable to this one parameter.

    Args:
        sahp_ainc_slow: The slow-AHP per-spike increment (uS) for this seizure
            state. Must be in ``[0, NORMAL_SAHP_SLOW)`` -- i.e. strictly weaker
            adaptation than normal. ``0`` removes the slow AHP entirely (the most
            severe case). Defaults to :data:`DEFAULT_SEIZURE_SAHP_SLOW`.
            For a relative depth, pass e.g. ``NORMAL_SAHP_SLOW * 0.4``.

    Returns:
        A dict identical to :func:`normal_state` except for ``sahp_ainc_slow``
        and ``state_name`` (which embeds the value, so it lands in the saved
        session metadata).

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
    return _state(value, f"seizure_sahp{value:g}")


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
