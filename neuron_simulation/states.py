"""Pharmacological / pathophysiological states.

Two orthogonal knobs drive this network away from the healthy regime; each is
expressed here as a dict of keyword overrides for
:func:`neuron_simulation.network_builder.build_network`:

* **Dynamic extracellular potassium** (``kdyn.mod``, parameter ``tau_k``) -- the
  mechanistic seizure model. Firing raises [K+]o, which depolarizes the K+
  reversal potential E_K (Nernst) and creates positive feedback toward an ictal
  (seizure) state; glial/diffusive clearance (``tau_k``) is the negative
  feedback that terminates it. Impaired clearance (large ``tau_k``) is the
  epilepsy model.
* **Delayed-rectifier downmodulation** (``ipotassium.g`` == ``gK``, the Ho et
  al. 2025 4-AP knob) -- the pharmacological *transition*. Reducing PY ``gK``
  broadens spikes -> Ca2+ influx -> I_KCa recruitment -> the 4-AP cascade. This
  replaces the old reduced-``gbar_kA`` route (see :func:`four_ap_state` /
  :func:`dose_response_gK`).

* ``normal_state``  -> reference ``gK`` at ``tau_k = 200 ms`` (strong buffering;
  [K+]o stays ~4 mM; discrete network bursts).
* ``seizure_state`` -> reference ``gK`` at large ``tau_k`` (impaired buffering;
  [K+]o accumulates into the ictal range; elevated, seizure-like firing).
* ``four_ap_state`` -> reduced PY ``gK`` at normal ``tau_k`` (the 4-AP cascade).

Reference ``gK`` (PY 15, FS 10 mS/cm2) comes from
:func:`neuron_simulation.neurons_ho.normal_gK`; the reduced-``gK`` transition
from :func:`neuron_simulation.neurons_ho.four_ap_gK`.
"""

import numpy as np

from .neurons_ho import four_ap_gK, normal_gK

#: Extracellular-K+ clearance time constant (ms) for the normal vs seizure state.
NORMAL_TAU_K = 200.0
SEIZURE_TAU_K = 12000.0
#: Slow-sAHP / fast-SFA per-spike increments (uS) carried by the seizure recipe.
#: NOTE: the Ho cell model (``neurons_ho``) has no sAHP mechanism -- Ca2+-activated
#: K+ (``ikCa``) supplies spike-frequency adaptation instead -- so these
#: ``sahp_*`` overrides are currently INERT (accepted and ignored by
#: build_network / neurons_ho). They are retained so the tuned seizure recipe
#: survives the cell-model swap; fold them into ``ikCa`` during re-tuning if wanted.
NORMAL_SAHP_SLOW = 0.0045
SEIZURE_SAHP_SLOW = 0.001
NORMAL_SAHP_FAST = 0.009
SEIZURE_SAHP_FAST = 0.006


def normal_state():
    """Return the build overrides for the normal (healthy) state.

    Reference delayed-rectifier ``gK`` (PY 15, FS 10) with strong glial K+
    buffering (``tau_k = 200 ms``) keeps [K+]o near 4 mM, so the network produces
    discrete synchronized bursts rather than sustained seizure.

    Args:
        None.

    Returns:
        A dict with ``gK_exc``/``gK_inh``, ``tau_k``, and a human-readable
        ``state_name``.
    """
    return {**normal_gK(), "tau_k": NORMAL_TAU_K}


def seizure_state(severity=1.0):
    """Return the build overrides for a seizure (impaired-K+-clearance) state.

    Models epilepsy as impaired glial/diffusive K+ buffering at reference ``gK``:
    a larger ``tau_k`` lets firing-driven [K+]o accumulate, depolarizing E_K and
    driving positive feedback into an ictal state.

    Args:
        severity: Seizure severity. ``0`` reduces to the normal clearance
            (``tau_k = 200 ms``); ``1.0`` gives the reference impaired value;
            larger values impair clearance further.

    Returns:
        A dict with reference ``gK``, an elevated ``tau_k``, the (inert) sahp
        recipe, and a ``state_name`` reflecting the severity.

    Raises:
        ValueError: If ``severity`` is negative.
    """
    if severity < 0.0:
        raise ValueError("severity must be >= 0.")
    frac = min(float(severity), 1.0)
    tau_k = NORMAL_TAU_K + float(severity) * (SEIZURE_TAU_K - NORMAL_TAU_K)
    sahp_ainc_slow = NORMAL_SAHP_SLOW + frac * (SEIZURE_SAHP_SLOW - NORMAL_SAHP_SLOW)
    sahp_ainc_fast = NORMAL_SAHP_FAST + frac * (SEIZURE_SAHP_FAST - NORMAL_SAHP_FAST)
    return {
        **normal_gK(),
        "state_name": f"seizure_sev{severity:g}",
        "tau_k": tau_k,
        "sahp_ainc_slow": sahp_ainc_slow,
        "sahp_ainc_fast": sahp_ainc_fast,
    }


def four_ap_state(py_gK=0.3, fs_gK=10.0):
    """Return build overrides for the 4-AP transition (reduced PY ``gK``).

    The mechanistic replacement for the deprecated ``gbar_kA`` block: reducing
    the PY delayed-rectifier ``gK`` (15 -> ~0.3) broadens spikes and recruits
    I_KCa, reproducing the 4-AP cascade at normal K+ clearance.

    Args:
        py_gK: Excitatory (PY) delayed-rectifier density (mS/cm2); sweep 15 ->
            0.3 for the cascade.
        fs_gK: Inhibitory (FS) delayed-rectifier density (mS/cm2); leave at 10 to
            isolate the PY effect, or lower it (7 protective / 0.03 depol-block).

    Returns:
        A dict with reduced ``gK_exc``/``gK_inh``, normal ``tau_k``, and a
        ``state_name``.
    """
    return {**four_ap_gK(py_gK=py_gK, fs_gK=fs_gK), "tau_k": NORMAL_TAU_K}


def seizure_dose_response(n_points=6, min_tau_k=NORMAL_TAU_K, max_tau_k=SEIZURE_TAU_K):
    """Build an ascending ``tau_k`` sweep for a seizure dose-response curve.

    Sweeps the K+ clearance time constant from healthy (strong buffering) toward
    impaired (seizure-prone) at reference ``gK``. Pair with per-point [K+]o and
    firing measurements to trace the transition into the ictal regime.

    Args:
        n_points: Number of ``tau_k`` values in the sweep.
        min_tau_k: Smallest (healthiest) clearance time constant (ms).
        max_tau_k: Largest (most impaired) clearance time constant (ms).

    Returns:
        A list of state dicts ordered from healthy toward seizure-prone.
    """
    sweep = []
    span = max(max_tau_k - min_tau_k, 1e-9)
    gk = normal_gK()
    for tau_k in np.linspace(min_tau_k, max_tau_k, n_points):
        frac = (float(tau_k) - min_tau_k) / span  # 0 (healthy) -> 1 (most impaired)
        sweep.append(
            {
                "state_name": f"tau_k_{tau_k:.0f}",
                "gK_exc": gk["gK_exc"],
                "gK_inh": gk["gK_inh"],
                "tau_k": float(tau_k),
                "sahp_ainc_slow": NORMAL_SAHP_SLOW + frac * (SEIZURE_SAHP_SLOW - NORMAL_SAHP_SLOW),
                "sahp_ainc_fast": NORMAL_SAHP_FAST + frac * (SEIZURE_SAHP_FAST - NORMAL_SAHP_FAST),
            }
        )
    return sweep


def dose_response_gK(n_points=8, max_py_gK=15.0, min_py_gK=0.3, fs_gK=10.0):
    """Build a descending PY ``gK`` sweep -- the 4-AP dose-response curve.

    Faithful replacement for the deprecated ``dose_response_gbar``: sweeps the
    excitatory delayed-rectifier from the drug-free reference (15) down toward
    strong block (~0.3) at normal K+ clearance, tracing the 4-AP cascade.

    Args:
        n_points: Number of ``gK`` values in the sweep.
        max_py_gK: Largest PY ``gK`` (drug-free end, mS/cm2).
        min_py_gK: Smallest PY ``gK`` (strong-block end, mS/cm2).
        fs_gK: Inhibitory (FS) ``gK`` held constant across the sweep (mS/cm2).

    Returns:
        A list of state dicts ordered from drug-free toward strong block.
    """
    sweep = []
    for py_gK in np.linspace(max_py_gK, min_py_gK, n_points):
        sweep.append({**four_ap_gK(py_gK=float(py_gK), fs_gK=fs_gK), "tau_k": NORMAL_TAU_K})
    return sweep
