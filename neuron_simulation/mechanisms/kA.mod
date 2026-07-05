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

The inactivation ``h`` (time constant ``htau0``, default 20 ms) shapes the
fast transient outward current that helps sculpt crisp, discrete bursts. (An
earlier 300 ms "slow pacer" variant is deprecated -- it gave mushier bursts.)

NOTE: on the realistic log-normal topology, reducing ``gbar`` does NOT
faithfully reproduce seizure. The mechanistically correct seizure route is
dynamic [K+]o accumulation (see kdyn.mod); reduced-``gbar`` is retained only as
a phenomenological knob.

The gbar knob (phenomenological / deprecated as a 4-AP model)
    ``gbar`` (S/cm2) is nominally the target of 4-aminopyridine (4-AP).
    Normal ~0.006 S/cm2. A partial reduction of ``gbar`` (e.g. 0.0045-0.005
    S/cm2) shifts burst frequency, but on the realistic topology the sign and
    magnitude of that effect are not a faithful 4-AP model -- see
    ``neuron_simulation/states.py``.

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
    gbar   = 0.006 (S/cm2)  : A-current density; THE 4-AP knob (reduce = block)
    vhalfm = -27   (mV)     : activation half-activation voltage
    km     = 16    (mV)     : activation slope factor (larger = shallower)
    vhalfh = -60   (mV)     : inactivation half-voltage (slow gate)
    kh     = 6     (mV)     : inactivation slope factor
    mtau0  = 1.0   (ms)     : activation time constant at temp (fast)
    htau0  = 20    (ms)     : inactivation time constant at temp (fast -> crisp
                            : discrete bursts; 300 ms slow-pacer variant gave
                            : mushier bursts and is deprecated)
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
