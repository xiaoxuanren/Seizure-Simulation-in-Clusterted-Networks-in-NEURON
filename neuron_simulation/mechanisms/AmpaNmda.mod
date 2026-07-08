TITLE AmpaNmda.mod  Two-component (AMPA + NMDA) depressing excitatory synapse

COMMENT
-----------------------------------------------------------------------------
Conductance-based excitatory synapse with a FAST AMPA component and a SLOW,
voltage-dependent NMDA component, plus Tsodyks-Markram short-term depression
on a shared depletable resource ``R`` (identical bookkeeping to DepSyn.mod).

Why two components
    Realistic network bursts last 200-800 ms because recurrent excitation
    *reverberates* -- and that reverberation is carried by NMDA receptors,
    whose slow decay (~50-150 ms) and voltage-dependent Mg2+ unblock sustain
    depolarization once a burst ignites. A single fast AMPA-only synapse
    produces sharp ~75 ms bursts; adding the NMDA component lengthens bursts to
    the culture range with every time constant at its measured value.

Dynamics
    g_ampa' = -g_ampa / tau_ampa            : fast AMPA conductance decay
    g_nmda' = -g_nmda / tau_nmda            : slow NMDA conductance decay
    R'      = (1 - R) / tau_d               : depression resource recovery

    On each presynaptic spike (NET_RECEIVE):
        g_ampa = g_ampa + weight * R
        g_nmda = g_nmda + weight * nmda_ratio * R
        R      = R * (1 - d)

    Mg2+ block (Jahr & Stevens 1990):
        B(v) = 1 / (1 + exp(-0.062 v) * mg / 3.57)     : ~0.01 at rest, ->0.5+ depolarized
    i = (g_ampa + g_nmda * B(v)) * (v - e)              : e = 0 mV

Parameters (all at literature values)
    tau_ampa   AMPA decay, ~2-5 ms (fast AMPA-R).
    tau_nmda   NMDA decay, ~50-150 ms (NMDA-R; Jahr & Stevens 1990; Lester 1990).
    nmda_ratio NMDA/AMPA peak-conductance ratio, ~0.5-1 at cortical synapses.
    mg         extracellular [Mg2+], 1 mM (physiological).
    d, tau_d   short-term depression (Silver 2002: d~0.5, tau_d~500-800 ms).

Units: g in microsiemens (uS), i in nanoamps (nA); NetCon ``weight`` is the
peak AMPA conductance increment in uS.

References: Jahr & Stevens (1990) J Neurosci; Tsodyks & Markram (1997);
Wang (1999) J Neurosci (NMDA reverberation); Lau & Bi (2005) PNAS.
-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
    POINT_PROCESS AmpaNmda
    RANGE tau_ampa, tau_nmda, e, d, tau_d, nmda_ratio, mg, g_ampa, g_nmda, i, R
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    tau_ampa   = 5      (ms)    : fast AMPA decay
    tau_nmda   = 100    (ms)    : slow NMDA decay
    e          = 0      (mV)    : reversal (excitatory)
    d          = 0.5           : per-spike depression fraction (0 = static)
    tau_d      = 600    (ms)    : depression resource recovery
    nmda_ratio = 0.7          : NMDA/AMPA peak-conductance ratio
    mg         = 1.0    (mM)   : extracellular Mg2+
}

ASSIGNED {
    v (mV)
    i (nA)
    B
}

STATE {
    g_ampa (uS)
    g_nmda (uS)
    R
}

INITIAL {
    g_ampa = 0
    g_nmda = 0
    R = 1
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    B = 1 / (1 + exp(-0.062 * v) * mg / 3.57)
    i = (g_ampa + g_nmda * B) * (v - e)
}

DERIVATIVE state {
    g_ampa' = -g_ampa / tau_ampa
    g_nmda' = -g_nmda / tau_nmda
    R' = (1 - R) / tau_d
}

NET_RECEIVE(weight (uS)) {
    g_ampa = g_ampa + weight * R
    g_nmda = g_nmda + weight * nmda_ratio * R
    R = R * (1 - d)
}
