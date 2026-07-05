"""Pharmacological / pathophysiological states.

The seizure mechanism in this project is **dynamic extracellular potassium**
(``kdyn.mod``), not a reduced A-current. Firing raises [K+]o, which depolarizes
the K+ reversal potential E_K (Nernst) and creates positive feedback toward an
ictal (seizure) state; glial/diffusive clearance -- parameterized by ``tau_k`` --
is the negative feedback that terminates it. Impaired clearance (large ``tau_k``)
is the epilepsy model.

* ``normal_state``  -> ``tau_k = 200 ms`` (strong buffering; [K+]o stays ~4 mM;
  discrete network bursts).
* ``seizure_state`` -> large ``tau_k`` (e.g. 2500 ms; impaired buffering; [K+]o
  accumulates into the ictal range; elevated, seizure-like firing).

The old reduced-``gbar_kA`` "4-AP" state is kept as :func:`gbar_block_state`
(with a back-compatible :func:`four_ap_state` alias) but is **deprecated to a
phenomenological knob**: on the realistic log-normal topology, changing
``gbar_kA`` does not faithfully reproduce the seizure phenotype (the dramatic
reduced-A-current effect was specific to the dense discrete-hub topology).

Each "state" is just a dict of keyword overrides for
:func:`neuron_simulation.network_builder.build_network`.
"""

import numpy as np

#: Excitatory / inhibitory A-current density in the normal (drug-free) state.
NORMAL_GBAR_KA_EXC = 0.006
NORMAL_GBAR_KA_INH = 0.004
#: Extracellular-K+ clearance time constant (ms) for the normal vs seizure state.
NORMAL_TAU_K = 200.0
SEIZURE_TAU_K = 2500.0


def normal_state():
    """Return the build overrides for the normal (healthy) state.

    Strong glial K+ buffering (``tau_k = 200 ms``) keeps [K+]o near 4 mM, so the
    network produces discrete synchronized bursts rather than sustained seizure.

    Args:
        None.

    Returns:
        A dict with ``gbar_kA_exc``/``gbar_kA_inh``, ``tau_k``, and a
        human-readable ``state_name``.
    """
    return {
        "state_name": "normal",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": NORMAL_TAU_K,
    }


def seizure_state(severity=1.0):
    """Return the build overrides for a seizure (impaired-K+-clearance) state.

    Models epilepsy as impaired glial/diffusive K+ buffering: a larger ``tau_k``
    lets firing-driven [K+]o accumulate, depolarizing E_K and driving positive
    feedback into an ictal state.

    Args:
        severity: Seizure severity. ``0`` reduces to the normal clearance
            (``tau_k = 200 ms``); ``1.0`` gives the reference impaired value
            (``tau_k = 2500 ms``); larger values impair clearance further.

    Returns:
        A dict with the A-current unchanged, an elevated ``tau_k``, and a
        ``state_name`` reflecting the severity.

    Raises:
        ValueError: If ``severity`` is negative.
    """
    if severity < 0.0:
        raise ValueError("severity must be >= 0.")
    tau_k = NORMAL_TAU_K + float(severity) * (SEIZURE_TAU_K - NORMAL_TAU_K)
    return {
        "state_name": f"seizure_sev{severity:g}",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": tau_k,
    }


def gbar_block_state(block_fraction=0.2):
    """Return build overrides for a reduced-A-current state (DEPRECATED knob).

    Phenomenological only. On the realistic log-normal topology, reducing
    ``gbar_kA`` does NOT faithfully reproduce seizure -- it shifts burst frequency
    with a topology-dependent sign/magnitude. Use :func:`seizure_state` for the
    mechanistic (K+-accumulation) seizure model. Retained for the dense
    discrete-hub topology where the reduced-A-current effect was dramatic.

    Args:
        block_fraction: Fraction of the A-current removed, in ``[0, 1)``.

    Returns:
        A dict with reduced ``gbar_kA_exc``/``gbar_kA_inh``, normal ``tau_k``,
        and a ``state_name``.

    Raises:
        ValueError: If ``block_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= block_fraction < 1.0):
        raise ValueError("block_fraction must be in [0, 1).")
    scale = 1.0 - block_fraction
    return {
        "state_name": f"gbar_block{int(round(block_fraction * 100))}",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC * scale,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH * scale,
        "tau_k": NORMAL_TAU_K,
    }


def four_ap_state(block_fraction=0.2):
    """Deprecated alias for :func:`gbar_block_state` (kept for back-compat).

    Args:
        block_fraction: Fraction of the A-current removed, in ``[0, 1)``.

    Returns:
        The dict returned by :func:`gbar_block_state`.
    """
    return gbar_block_state(block_fraction)


def seizure_dose_response(n_points=6, min_tau_k=NORMAL_TAU_K, max_tau_k=SEIZURE_TAU_K):
    """Build an ascending ``tau_k`` sweep for a seizure dose-response curve.

    Sweeps the K+ clearance time constant from healthy (strong buffering) toward
    impaired (seizure-prone). Pair with per-point [K+]o and firing measurements
    to trace the transition into the ictal regime.

    Args:
        n_points: Number of ``tau_k`` values in the sweep.
        min_tau_k: Smallest (healthiest) clearance time constant (ms).
        max_tau_k: Largest (most impaired) clearance time constant (ms).

    Returns:
        A list of state dicts ordered from healthy toward seizure-prone.
    """
    sweep = []
    for tau_k in np.linspace(min_tau_k, max_tau_k, n_points):
        sweep.append(
            {
                "state_name": f"tau_k_{tau_k:.0f}",
                "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
                "gbar_kA_inh": NORMAL_GBAR_KA_INH,
                "tau_k": float(tau_k),
            }
        )
    return sweep


def dose_response_gbar(n_points=8, min_fraction=1.0, max_fraction=0.2):
    """Build a descending A-current-density sweep (DEPRECATED / phenomenological).

    Retained for the discrete-hub topology. Prefer :func:`seizure_dose_response`
    (K+ clearance) for the mechanistic seizure model.

    Args:
        n_points: Number of gbar values in the sweep.
        min_fraction: Largest gbar as a fraction of normal (drug-free end).
        max_fraction: Smallest gbar as a fraction of normal (strong-block end).

    Returns:
        A list of state dicts ordered from drug-free toward strong block.
    """
    sweep = []
    for frac in np.linspace(min_fraction, max_fraction, n_points):
        sweep.append(
            {
                "state_name": f"gbar_frac_{frac:.2f}",
                "fraction": float(frac),
                "gbar_kA_exc": NORMAL_GBAR_KA_EXC * float(frac),
                "gbar_kA_inh": NORMAL_GBAR_KA_INH * float(frac),
                "tau_k": NORMAL_TAU_K,
            }
        )
    return sweep
