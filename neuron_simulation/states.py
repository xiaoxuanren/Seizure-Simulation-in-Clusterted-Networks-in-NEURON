"""Pharmacological states: normal vs 4-AP, and dose-response helpers.

The A-current density ``gbar_kA`` is the pharmacological knob. 4-aminopyridine
(4-AP) blocks A-type / Kv4-like potassium channels, so a 4-AP application is
modelled as a *partial reduction* of ``gbar_kA``.

Key validated finding (encoded here and documented in the README):

* Normal state: ``gbar_kA`` ~ 0.006 S/cm2 -- the network produces discrete,
  well-separated synchronized bursts.
* 4-AP is a **partial** block. Reducing ``gbar_kA`` weakens the burst
  brake, so bursts arrive *more frequently*. There is a dose window in which
  burst frequency rises with block strength.
* A **strong** block (``gbar_kA`` near 0) removes the terminator entirely and
  collapses discrete bursts into continuous firing. 4-AP must therefore stay in
  the partial regime (~0.0045-0.005 S/cm2 for the excitatory population).

Each "state" is just a dict of keyword overrides for
:func:`neuron_simulation.network_builder.build_network`.
"""

import numpy as np

#: Excitatory / inhibitory A-current density in the normal (drug-free) state.
NORMAL_GBAR_KA_EXC = 0.006
NORMAL_GBAR_KA_INH = 0.004


def normal_state():
    """Return the build overrides for the normal (drug-free) state.

    Args:
        None.

    Returns:
        A dict with ``gbar_kA_exc``/``gbar_kA_inh`` and a human-readable
        ``state_name``, suitable for ``build_network(**overrides)`` or
        ``network.apply_state(...)``.
    """
    return {
        "state_name": "normal",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH,
    }


def four_ap_state(block_fraction=0.2):
    """Return the build overrides for a partial 4-AP block.

    Args:
        block_fraction: Fraction of the A-current removed, in ``[0, 1)``. The
            default 0.2 (a 20% reduction, ~0.0048 S/cm2 excitatory) sits in the
            partial-block dose window that *increases* burst frequency. Values
            approaching 1.0 leave the partial regime and merge bursts into
            continuous firing -- a warning is printed above 0.5.

    Returns:
        A dict with reduced ``gbar_kA_exc``/``gbar_kA_inh`` and a
        ``state_name`` reflecting the block fraction.

    Raises:
        ValueError: If ``block_fraction`` is not in ``[0, 1)``.
    """
    if not (0.0 <= block_fraction < 1.0):
        raise ValueError("block_fraction must be in [0, 1).")
    if block_fraction > 0.5:
        print(
            f"WARNING: block_fraction={block_fraction:.2f} is a STRONG block; "
            "discrete bursts may merge into continuous firing (leave the partial "
            "4-AP regime)."
        )
    scale = 1.0 - block_fraction
    return {
        "state_name": f"4AP_block{int(round(block_fraction * 100))}",
        "gbar_kA_exc": NORMAL_GBAR_KA_EXC * scale,
        "gbar_kA_inh": NORMAL_GBAR_KA_INH * scale,
    }


def dose_response_gbar(n_points=8, min_fraction=1.0, max_fraction=0.2):
    """Build a descending sweep of A-current densities for a dose-response curve.

    Sweeps the *excitatory* A-current density from a fraction ``min_fraction`` of
    the normal value (drug-free end) down to ``max_fraction`` (strong-block end).
    Pair with a burst-detection run at each point to trace burst frequency vs
    ``gbar_kA`` (see :func:`neuron_simulation.plotting.plot_burst_frequency_curve`).

    Args:
        n_points: Number of gbar values in the sweep.
        min_fraction: Largest gbar as a fraction of normal (drug-free end).
        max_fraction: Smallest gbar as a fraction of normal (strong-block end).

    Returns:
        A list of dicts, each ``{"gbar_kA_exc", "gbar_kA_inh", "fraction",
        "state_name"}``, ordered from drug-free toward strong block.
    """
    fractions = np.linspace(min_fraction, max_fraction, n_points)
    sweep = []
    for frac in fractions:
        sweep.append(
            {
                "state_name": f"gbar_frac_{frac:.2f}",
                "fraction": float(frac),
                "gbar_kA_exc": NORMAL_GBAR_KA_EXC * float(frac),
                "gbar_kA_inh": NORMAL_GBAR_KA_INH * float(frac),
            }
        )
    return sweep
