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

The reduced-``gbar_kA`` "4-AP" state is kept as :func:`gbar_block_state` (alias
:func:`four_ap_state`) for API compatibility only. It is **inert at the shipped
``kA`` parameters and must not be used as a seizure knob**.

Measured, not inferred: ``kA.mod`` ships ``vhalfm = -27 mV`` / ``km = 16`` and the
conductance is ``m^4``, so the conductance half-activates ~26.6 mV depolarized of
the m-gate, at ``-0.4 mV`` -- where ``h`` (``vhalfh = -60``, ``kh = 6``) is fully
inactivated. Peak steady-state window conductance is 1.9e-6 S/cm2 = 0.005% of the
``hh`` delayed rectifier, and over -65..-50 mV ``g_kA`` is under 2% of the leak
conductance. Blocking it does nothing to the burst phenotype, at any dose, on any
topology. Pinned by ``tests/test_kA_characterization.py``. (The subthreshold bound
is on conductance: the leak *current* crosses zero at ``el_hh = -54.3 mV``, so a
current ratio there is unbounded and would not mean anything.)

The ``gbar_kA`` numbers are nonetheless retained rather than zeroed, so existing
datasets reproduce bit-for-bit. Inertness is a *subthreshold* claim; ``m^4`` is
non-negligible at the spike peak, so zeroing ``gbar_kA`` perturbs spike waveforms
and this chaotic network turns that into a different spike train -- the arms are
bit-identical until t = 2175.5 ms, and by 20 s all 926 trains differ. Measured by
``scripts/check_ka_contribution.py``. (That run's spike counts, 6180 vs 4442, are
one burst of window quantization, not a rate effect, and its window is too short
to compare burst statistics -- 2 inter-burst intervals for one arm, 1 for the
other. It settles bitwise identity only; the phenotype claim above rests on the
conductance arithmetic instead.)

An earlier version of this note attributed the flat dose-response to the log-normal
topology and claimed a "dramatic reduced-A-current effect" on the dense discrete-hub
topology. That claim is unsupported and has been removed: ``git log`` shows
``vhalfm`` has never held another value and ``kA_globals`` has no call sites, so the
current has been inert on every topology this repo has ever run. An inert mechanism
cannot produce a topology-dependent effect; any such effect came from the companion
LIF project's separate A-current implementation.

Setting ``vhalfm = -54`` restores the documented behaviour (conductance V1/2 =
-27.4 mV) and gives a functional current at the single-cell level, but still yields
no network dose-response: the ``sAHP`` per-spike increment is 1.2-2.6x the leak
conductance and decays over 4-6.5 s, dominating the adaptation budget on the
seconds-scale timescale that sets burst rate. A working ``gbar_kA`` knob needs sAHP
re-balanced against I_A, not just a gating fix -- and the gating fix alone shifts the
tuned baseline. Not applied here.

Each "state" is just a dict of keyword overrides for
:func:`neuron_simulation.network_builder.build_network`.
"""

import numpy as np

#: Excitatory / inhibitory A-current density in the normal (drug-free) state.
NORMAL_GBAR_KA_EXC = 0.006
NORMAL_GBAR_KA_INH = 0.004
#: Extracellular-K+ clearance time constant (ms) for the normal vs seizure state.
NORMAL_TAU_K = 200.0
SEIZURE_TAU_K = 12000.0
#: Slow-sAHP per-spike increment (uS): full in the normal state, strongly reduced
#: in seizure. Reduced slow adaptation (Kv7/sAHP dysfunction) is itself an
#: epilepsy mechanism, and the slow AHP is otherwise anticonvulsant enough that
#: impairing K+ clearance alone does not reach the ictal ceiling.
NORMAL_SAHP_SLOW = 0.0045
SEIZURE_SAHP_SLOW = 0.001
#: Fast-SFA per-spike increment (uS): full in the normal state, reduced in
#: seizure. Kv7/M-current loss reduces fast adaptation too; without this the
#: strong fast SFA is anticonvulsant enough to cap [K+]o below the ictal range.
NORMAL_SAHP_FAST = 0.009
SEIZURE_SAHP_FAST = 0.006


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
    frac = min(float(severity), 1.0)
    tau_k = NORMAL_TAU_K + float(severity) * (SEIZURE_TAU_K - NORMAL_TAU_K)
    sahp_ainc_slow = NORMAL_SAHP_SLOW + frac * (SEIZURE_SAHP_SLOW - NORMAL_SAHP_SLOW)
    sahp_ainc_fast = NORMAL_SAHP_FAST + frac * (SEIZURE_SAHP_FAST - NORMAL_SAHP_FAST)
    return {
        "state_name": f"seizure_sev{severity:g}",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
        "tau_k": tau_k,
        "sahp_ainc_slow": sahp_ainc_slow,
        "sahp_ainc_fast": sahp_ainc_fast,
    }


def gbar_block_state(block_fraction=0.2):
    """Return build overrides for a reduced-A-current state (INERT; API-compat only).

    The A-current is **inert at the shipped ``kA`` parameters**, so this state is a
    no-op at any ``block_fraction``, on any topology: it returns a reduced
    ``gbar_kA`` that the mechanism does not act on. See the module docstring for the
    measured numbers. Use :func:`seizure_state` for the mechanistic
    (K+-accumulation) seizure model.

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
    span = max(max_tau_k - min_tau_k, 1e-9)
    for tau_k in np.linspace(min_tau_k, max_tau_k, n_points):
        frac = (float(tau_k) - min_tau_k) / span  # 0 (healthy) -> 1 (most impaired)
        sweep.append(
            {
                "state_name": f"tau_k_{tau_k:.0f}",
                "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
                "gbar_kA_inh": NORMAL_GBAR_KA_INH,
                "tau_k": float(tau_k),
                "sahp_ainc_slow": NORMAL_SAHP_SLOW + frac * (SEIZURE_SAHP_SLOW - NORMAL_SAHP_SLOW),
                "sahp_ainc_fast": NORMAL_SAHP_FAST + frac * (SEIZURE_SAHP_FAST - NORMAL_SAHP_FAST),
            }
        )
    return sweep


def dose_response_gbar(n_points=8, min_fraction=1.0, max_fraction=0.2):
    """Build a descending A-current-density sweep (INERT; API-compat only).

    The A-current is inert at the shipped ``kA`` parameters, so this sweep is flat
    by construction -- every point returns the same network behaviour. See the
    module docstring. Use :func:`seizure_dose_response` (K+ clearance) for the
    mechanistic seizure model.

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
