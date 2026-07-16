"""Pin the measured characterization of the ``kA`` A-current.

``kA.mod`` computes ``g = gbar * m^4 * h``. The ``m^4`` exponent means the
*conductance* half-activates ``-km * ln(2^(1/4) - 1)`` = 1.665*km millivolts
depolarized of the *m-gate*'s own half-activation ``vhalfm``. At the shipped
``km = 16`` that offset is 26.64 mV, so the shipped ``vhalfm = -27 mV`` puts the
conductance V1/2 at -0.4 mV -- where the inactivation gate ``h``
(``vhalfh = -60``, ``kh = 6``) is fully closed. The mechanism is therefore inert:
peak steady-state window conductance is 0.005% of the ``hh`` delayed rectifier,
and over -65..-50 mV ``g_kA`` is under 2% of the leak conductance.

Note the subthreshold bound is on **conductance**, not current: the leak *current*
crosses zero at ``el_hh = -54.3 mV``, so the current ratio there is unbounded and
says nothing about relevance. Conductance is the honest comparison.

These tests are pure numpy and deliberately do **not** import NEURON, so they run
anywhere. That means they check a *mirror* of the mechanism, so the mirror itself
has to be pinned to ``kA.mod`` or the whole exercise is circular. Three things are
therefore parsed out of the ``.mod`` and asserted, not hardcoded: the PARAMETER
defaults, the ``m^4`` exponent in ``BREAKPOINT``, and the ``minf``/``hinf`` forms
in ``PROCEDURE rates``. If anyone retunes any of them, these tests fail loudly
instead of silently continuing to assert a characterization that no longer holds.

The exponent in particular is load-bearing: the whole 1.665*km offset -- and hence
the entire inertness conclusion -- exists *only* because the conductance goes as
``m^4``. At ``m^3`` the offset would be 21.6 mV, not 26.64.

See the "The A-current is inert" section of ``README.md`` for why the candidate
fix (``vhalfm = -54``) is not applied.
"""

import os
import re

import numpy as np
import pytest

MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "neuron_simulation",
    "mechanisms",
    "kA.mod",
)

#: Conductance of NEURON's built-in ``hh`` delayed-rectifier K+ current (S/cm2).
HH_GK = 0.036
#: Conductance of NEURON's built-in ``hh`` leak current (S/cm2).
HH_GLEAK = 3e-4


def _parse_mod_parameters(path=MOD_PATH):
    """Parse the PARAMETER block defaults out of a NEURON ``.mod`` file.

    Reads the values the mechanism actually ships with, so this module tests the
    real mechanism rather than a copy of its numbers that can drift out of date.

    Args:
        path: Path to the ``.mod`` file. Defaults to :data:`MOD_PATH`.

    Returns:
        A dict mapping parameter name to its float default, e.g.
        ``{"gbar": 0.006, "vhalfm": -27.0, ...}``.

    Raises:
        AssertionError: If the file has no ``PARAMETER`` block.
    """
    with open(path, "r") as handle:
        text = handle.read()

    block = re.search(r"^PARAMETER\s*\{(.*?)^\}", text, re.S | re.M)
    assert block, "No PARAMETER block found in %s" % path

    params = {}
    for line in block.group(1).splitlines():
        line = line.split(":", 1)[0]  # strip the trailing NMODL ``:`` comment
        match = re.match(r"\s*(\w+)\s*=\s*(-?[\d.eE+-]+)", line)
        if match:
            params[match.group(1)] = float(match.group(2))
    return params


def _parse_conductance_exponent(path=MOD_PATH):
    """Return the exponent ``n`` in ``BREAKPOINT``'s ``g = gbar * m^n * h``.

    ``kA.mod`` spells the exponent out as repeated factors (``m * m * m * m``)
    because NMODL has no ``^`` operator in BREAKPOINT, so this counts the factors.

    Args:
        path: Path to the ``.mod`` file. Defaults to :data:`MOD_PATH`.

    Returns:
        The integer exponent applied to ``m``.

    Raises:
        AssertionError: If no ``g = gbar * ... * h`` assignment is found.
    """
    with open(path, "r") as handle:
        text = handle.read()

    # Scope to BREAKPOINT: the COMMENT block also contains a prose "g = gbar * m^4
    # * h", and matching that instead would make this guard test the docs, not the
    # code -- which is exactly the circularity this function exists to prevent.
    block = re.search(r"^BREAKPOINT\s*\{(.*?)^\}", text, re.S | re.M)
    assert block, "No BREAKPOINT block found in %s" % path

    match = re.search(r"^\s*g\s*=\s*gbar\s*\*(.+?)$", block.group(1), re.M)
    assert match, "No 'g = gbar * ...' assignment in BREAKPOINT"

    factors = [f.strip() for f in match.group(1).split("*")]
    assert "h" in factors, "BREAKPOINT no longer multiplies by h: %r" % match.group(1)
    return factors.count("m")


def _parse_rate_forms(path=MOD_PATH):
    """Return the ``minf``/``hinf`` right-hand sides from ``PROCEDURE rates``.

    Whitespace is normalized so formatting changes do not trip the comparison,
    while a genuine change of form (e.g. a flipped exponent sign) does.

    Args:
        path: Path to the ``.mod`` file. Defaults to :data:`MOD_PATH`.

    Returns:
        A dict with ``"minf"`` and ``"hinf"`` normalized expression strings.
    """
    with open(path, "r") as handle:
        text = handle.read()

    # The signature is ``rates(v (mV))`` -- nested parens, so a ``[^)]*`` class
    # would stop at the inner ``)`` and never match.
    block = re.search(r"^PROCEDURE\s+rates\s*\(.*\)\s*\{(.*?)^\}", text, re.S | re.M)
    assert block, "No PROCEDURE rates block found in %s" % path

    forms = {}
    for name in ("minf", "hinf"):
        match = re.search(r"^\s*%s\s*=\s*(.+?)$" % name, block.group(1), re.M)
        assert match, "No %s assignment in PROCEDURE rates" % name
        forms[name] = re.sub(r"\s+", "", match.group(1))
    return forms


@pytest.fixture(scope="module")
def mod():
    """Return the ``kA.mod`` PARAMETER defaults as a dict."""
    return _parse_mod_parameters()


# --------------------------------------------------------------------------- #
# Pin the mirror to the mechanism. Without these, the tests below would happily
# keep asserting the shipped characterization after kA.mod's *structure* changed.
# --------------------------------------------------------------------------- #
def test_conductance_is_still_m_to_the_fourth():
    """The mirror's ``** 4`` must still match BREAKPOINT's ``g = gbar*m*m*m*m*h``.

    This is the load-bearing fact: the 26.64 mV offset, and therefore the whole
    inertness result, follows from the exponent being 4. A retune to ``m^3`` would
    move the offset to 21.6 mV and the peak window g to 6.8e-6 -- so if this ever
    fails, every number in this module is stale.
    """
    assert _parse_conductance_exponent() == 4


def test_rate_equations_still_match_the_mirror():
    """``PROCEDURE rates`` must still match the ``minf``/``hinf`` mirrored here.

    Guards the gate *signs* specifically: flipping ``hinf``'s exponent sign would
    make the current activate rather than inactivate with depolarization, and the
    mechanism would stop being inert -- while a mirror-only test kept passing.
    """
    forms = _parse_rate_forms()
    assert forms["minf"] == "1/(1+exp(-(v-vhalfm)/km))"
    assert forms["hinf"] == "1/(1+exp((v-vhalfh)/kh))"


# --------------------------------------------------------------------------- #
# Rate equations -- mirror kA.mod's PROCEDURE rates(v) exactly.
# --------------------------------------------------------------------------- #
def minf(v, vhalfm, km):
    """Steady-state activation. Mirrors ``minf = 1/(1 + exp(-(v - vhalfm)/km))``."""
    return 1.0 / (1.0 + np.exp(-(v - vhalfm) / km))


def hinf(v, vhalfh, kh):
    """Steady-state inactivation. Mirrors ``hinf = 1/(1 + exp((v - vhalfh)/kh))``."""
    return 1.0 / (1.0 + np.exp((v - vhalfh) / kh))


def g_kA(v, gbar, vhalfm, km, vhalfh, kh):
    """Steady-state window conductance. Mirrors ``g = gbar * m^4 * h`` at steady state."""
    return gbar * minf(v, vhalfm, km) ** 4 * hinf(v, vhalfh, kh)


def m4_offset(km):
    """Return the mV offset from the m-gate V1/2 to the m^4 conductance V1/2.

    Solving ``minf(v)^4 = 0.5`` gives ``minf = 2^(-1/4)``, hence
    ``exp(-(v - vhalfm)/km) = 2^(1/4) - 1`` and ``v - vhalfm = -km*ln(2^(1/4) - 1)``,
    i.e. ``1.665*km``. The offset is positive: raising the exponent to 4 means
    more depolarization is needed to reach half the maximal conductance.

    Args:
        km: Activation slope factor (mV).

    Returns:
        The offset in mV.
    """
    return -km * np.log(2.0 ** 0.25 - 1.0)


def conductance_vhalf(vhalfm, km):
    """Return the voltage at which ``m^4`` reaches half its maximum (mV)."""
    return vhalfm + m4_offset(km)


# --------------------------------------------------------------------------- #
# The shipped mechanism
# --------------------------------------------------------------------------- #
def test_shipped_parameters_are_what_this_module_assumes(mod):
    """Guard: fail loudly if kA.mod is retuned out from under this analysis."""
    assert mod["gbar"] == pytest.approx(0.006)
    assert mod["vhalfm"] == pytest.approx(-27.0)
    assert mod["km"] == pytest.approx(16.0)
    assert mod["vhalfh"] == pytest.approx(-60.0)
    assert mod["kh"] == pytest.approx(6.0)


def test_m4_offset_is_26_64_mV(mod):
    """The m^4 exponent shifts the conductance V1/2 26.64 mV depolarized (km = 16)."""
    assert m4_offset(mod["km"]) == pytest.approx(26.64, abs=0.01)
    # The offset is exactly 1.665*km, the form quoted in kA.mod and the README.
    assert m4_offset(mod["km"]) == pytest.approx(1.665 * mod["km"], abs=0.02)


def test_conductance_vhalf_is_near_zero_mV(mod):
    """vhalfm = -27 puts the CONDUCTANCE V1/2 at -0.4 mV, not at -27 mV."""
    v_half = conductance_vhalf(mod["vhalfm"], mod["km"])
    assert v_half == pytest.approx(-0.4, abs=0.1)

    # ...which is ~60 mV depolarized of the inactivation V1/2, so h is shut there.
    assert hinf(v_half, mod["vhalfh"], mod["kh"]) < 1e-4


def test_peak_window_conductance_is_negligible(mod):
    """Peak steady-state window g < 2e-6 S/cm2 = under 0.01% of the hh gK."""
    v = np.linspace(-100.0, 60.0, 160001)
    g = g_kA(v, mod["gbar"], mod["vhalfm"], mod["km"], mod["vhalfh"], mod["kh"])
    peak = g.max()

    assert peak < 2e-6
    assert peak / HH_GK < 1e-4  # under 0.01% of the delayed rectifier
    # The documented value, pinned so the docs cannot drift from the mechanism.
    assert peak == pytest.approx(1.88e-6, rel=0.02)


def test_subthreshold_conductance_is_under_2_percent_of_leak(mod):
    """At -65..-50 mV, with h at its resting value, g_kA is under 2% of leak.

    This bounds the **conductance**, not the current: ``hh``'s leak current crosses
    zero at ``el = -54.3 mV``, so a current ratio is unbounded in this range and
    would be meaningless. Conductance is the meaningful comparison.

    ``h`` is pinned at ``hinf(-65)`` rather than tracked to steady state: h is the
    slow gate, so on the timescale of a subthreshold depolarization it stays near
    its resting value. Pinning it is the *generous* assumption -- letting h relax
    to steady state would only inactivate it further and shrink g_kA.
    """
    h_rest = hinf(-65.0, mod["vhalfh"], mod["kh"])
    assert h_rest == pytest.approx(0.697, abs=0.01)

    v = np.linspace(-65.0, -50.0, 1001)
    g = mod["gbar"] * minf(v, mod["vhalfm"], mod["km"]) ** 4 * h_rest

    assert g.max() < 0.02 * HH_GLEAK


# --------------------------------------------------------------------------- #
# Contrast: the candidate fix, documented but NOT applied (see README).
# --------------------------------------------------------------------------- #
def test_vhalfm_minus_54_would_restore_a_functional_current(mod):
    """Sanity/contrast: vhalfm = -54 puts the conductance V1/2 at -27.4 mV.

    This documents *why* -54 is the candidate fix. It is not applied on this
    branch: it shifts the tuned baseline and would force a re-tune plus a dataset
    regeneration. See the README section "The A-current is inert".
    """
    candidate_vhalfm = -54.0

    v_half = conductance_vhalf(candidate_vhalfm, mod["km"])
    assert v_half == pytest.approx(-27.4, abs=0.1)

    v = np.linspace(-100.0, 60.0, 160001)
    g = g_kA(v, mod["gbar"], candidate_vhalfm, mod["km"], mod["vhalfh"], mod["kh"])
    assert g.max() > 5e-5

    # ...i.e. ~50x the shipped mechanism's peak window conductance.
    shipped = g_kA(v, mod["gbar"], mod["vhalfm"], mod["km"], mod["vhalfh"], mod["kh"]).max()
    assert g.max() / shipped > 20.0


def test_candidate_fix_is_not_applied(mod):
    """The shipped mechanism must still be the inert one; -54 is tracked elsewhere."""
    assert mod["vhalfm"] == pytest.approx(-27.0), (
        "vhalfm changed -- if the gating fix was applied, the tuned baseline moved "
        "and the existing datasets need regenerating. See README."
    )
