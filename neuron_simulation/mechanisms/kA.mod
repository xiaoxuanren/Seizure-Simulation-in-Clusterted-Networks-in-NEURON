TITLE kA.mod  A-type (Kv4-like) transient potassium current

COMMENT
-----------------------------------------------------------------------------
A-type (Kv4-like) transient outward potassium current for single-compartment
Hodgkin-Huxley cells.

Biophysics
    I_kA = gbar * m^4 * h * (v - ek)

    m : fast, voltage-gated ACTIVATION gate (opens on depolarization)
    h : slow, voltage-gated INACTIVATION gate (closes on depolarization and
        recovers slowly at rest)

The inactivation ``h`` has time constant ``htau0`` (default 20 ms; an earlier
300 ms "slow pacer" variant is deprecated). ``htau0`` does not currently affect
network output either -- see the inertness note below.

!!! THIS MECHANISM IS INERT AT ITS SHIPPED PARAMETERS !!!
    g = gbar * m^4 * h, so the CONDUCTANCE half-activates at vhalfm + 1.665*km --
    ~26.6 mV DEPOLARIZED of the m-gate (km = 16). The shipped vhalfm = -27 mV
    therefore puts the conductance V1/2 at -0.4 mV, where h (vhalfh = -60, kh = 6)
    is fully inactivated.
        peak steady-state window g = 1.88e-6 S/cm2 = 0.005% of hh gK
        g_kA at -65..-50 mV        = under 2% of leak (3e-4 S/cm2)
    ``gbar`` is a DEAD PARAMETER: reducing it does nothing to the burst phenotype
    at any dose, on any topology. Do NOT use it as a 4-AP or seizure knob. The
    live seizure route is dynamic [K+]o accumulation (kdyn.mod).

    But do NOT zero gbar either: the above is a SUBTHRESHOLD statement, and m^4
    is non-negligible at the spike peak, so zeroing gbar perturbs spike waveforms
    and this chaotic network amplifies that into a different spike train (all 926
    trains differ by 20 s; burst statistics unchanged). The shipped values are
    retained so existing datasets reproduce bit-for-bit -- measured by
    scripts/check_ka_contribution.py.

    Pinned by tests/test_kA_characterization.py. See the README section "The
    A-current is inert" for why vhalfm = -54 is not applied here.

Temperature
    Base time constants ``mtau0``/``htau0`` are defined at ``temp`` = 6.3 degC
    (the reference temperature of NEURON's built-in ``hh`` squid kinetics). A
    q10 factor scales them to the simulated ``celsius``, so a mammalian variant
    at ``celsius`` = 34 automatically runs with faster kinetics.

References (form): Connor & Stevens (1971); Huguenard & McCormick (1992);
Dayan & Abbott, Theoretical Neuroscience (A-current). Parameters here are a
compact, tunable Kv4-like caricature rather than a fit to a specific cell.
-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
    SUFFIX kA
    USEION k READ ek WRITE ik
    RANGE gbar, g, ik
    RANGE minf, hinf, mtau, htau
    GLOBAL vhalfm, km, vhalfh, kh, mtau0, htau0, q10, temp
}

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S)  = (siemens)
}

PARAMETER {
    gbar   = 0.006 (S/cm2)  : A-current density; DEAD PARAMETER (see COMMENT)
    vhalfm = -27   (mV)     : activation half-activation voltage
    km     = 16    (mV)     : activation slope factor (larger = shallower)
    vhalfh = -60   (mV)     : inactivation half-voltage (slow gate)
    kh     = 6     (mV)     : inactivation slope factor
    mtau0  = 1.0   (ms)     : activation time constant at temp (fast)
    htau0  = 20    (ms)     : inactivation time constant at temp (300 ms
                            : slow-pacer variant is deprecated). No effect on
                            : network output while the mechanism is inert.
    q10    = 3              : temperature sensitivity of the gating kinetics
    temp   = 6.3   (degC)   : reference temperature for mtau0/htau0
}

ASSIGNED {
    v      (mV)
    ek     (mV)
    celsius (degC)
    ik     (mA/cm2)
    g      (S/cm2)
    minf
    hinf
    mtau   (ms)
    htau   (ms)
    tadj
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar * m * m * m * m * h
    ik = g * (v - ek)
}

INITIAL {
    tadj = q10 ^ ((celsius - temp) / 10)
    rates(v)
    m = minf
    h = hinf
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

UNITSOFF
PROCEDURE rates(v (mV)) {
    : fast activation, slow inactivation; taus scaled by temperature (tadj)
    minf = 1 / (1 + exp(-(v - vhalfm) / km))
    hinf = 1 / (1 + exp((v - vhalfh) / kh))
    mtau = mtau0 / tadj
    htau = htau0 / tadj
}
UNITSON
