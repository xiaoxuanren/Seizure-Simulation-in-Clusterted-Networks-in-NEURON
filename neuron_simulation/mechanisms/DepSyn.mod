TITLE DepSyn.mod  Depressing excitatory synapse (short-term depression)

COMMENT
-----------------------------------------------------------------------------
Conductance-based excitatory synapse with Tsodyks-Markram-style short-term
depression driven by a single depletable resource variable ``R``.

Dynamics
    g' = -g / tau                     : conductance decays exponentially
    R' = (1 - R) / tau_d              : resource recovers toward 1

    On each presynaptic spike (NET_RECEIVE):
        g = g + weight * R            : increment scaled by AVAILABLE resource
        R = R * (1 - d)               : deplete the resource by fraction d

    i = g * (v - e)                   : excitatory current (e = 0 mV)

Roles of the parameters
    d      per-spike depression fraction in [0, 1).
           d = 0    -> R stays 1  -> a STATIC exponential synapse.
           d = 0.5  -> each spike halves the available resource (strong STD).
    tau_d  resource-recovery time constant (ms). Larger = slower recovery =
           more accumulated depression during high-frequency drive.

Why keep it
    The A-current (kA.mod) sets the ignition threshold and burst frequency (it
    is the 4-AP knob), while short-term depression is the burst terminator /
    brake against runaway. In the tuned default regime here, depression is
    load-bearing: with it OFF (d = 0) at the default recurrent strength the
    network runs away into continuous firing instead of discrete bursts. (The
    LIF-based reference reported that the network still bursts with static
    synapses at a weaker operating point; see the README's honest-note.)

Units: g in microsiemens (uS), i in nanoamps (nA), so the NetCon ``weight`` is
a peak conductance increment in uS.

Reference (form): Tsodyks & Markram (1997); Varela et al. (1997).
-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
    POINT_PROCESS DepSyn
    RANGE tau, e, d, tau_d, g, i, R
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau   = 3    (ms)   : conductance decay time constant
    e     = 0    (mV)   : reversal potential (excitatory)
    d     = 0.5         : per-spike depression fraction (0 = static synapse)
    tau_d = 800  (ms)   : synaptic resource recovery time constant
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    g (uS)
    R
}

INITIAL {
    g = 0
    R = 1
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g * (v - e)
}

DERIVATIVE state {
    g' = -g / tau
    R' = (1 - R) / tau_d
}

NET_RECEIVE(weight (uS)) {
    g = g + weight * R
    R = R * (1 - d)
}
