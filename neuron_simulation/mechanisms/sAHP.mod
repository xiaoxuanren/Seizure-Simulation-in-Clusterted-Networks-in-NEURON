TITLE sAHP.mod  Two-timescale spike-triggered adaptation (fast SFA + slow AHP)

COMMENT
-----------------------------------------------------------------------------
Two spike-triggered K+ conductances with DISTINCT time constants, sharing one
reversal potential. Each spike deposits a fixed increment into both; they decay
independently.

    FAST component (g_fast, tau_fast ~ 100-300 ms)
        Fast spike-frequency adaptation (M-current / Kv7-like). Builds up within
        a burst and limits the number of spikes each cell fires per burst to a
        few, then decays between bursts so it does NOT suppress the sparse
        asynchronous inter-burst firing.
    SLOW component (g_slow, tau_slow ~ 1-4 s)
        Slow afterhyperpolarization (Ca2+-dependent sAHP). Accumulates across a
        burst and decays over seconds, setting the multi-second INTER-BURST
        INTERVAL. This is the project's SEIZURE KNOB (see states.py). It is the
        KCa-type sAHP, not the M-current -- do not attribute it to Kv7/KCNQ.
        The shipped network runs tau_slow = 6500 ms, above the range quoted here
        and above the measured I_sAHP (1-5 s); that is a tuning choice.

    g_fast' = -g_fast / tau_fast
    g_slow' = -g_slow / tau_slow
    On each spike (NET_RECEIVE):
        g_fast = g_fast + ainc_fast
        g_slow = g_slow + ainc_slow
    i = (g_fast + g_slow) * (v - ek)

Why two timescales
    A single-timescale adaptation cannot do both jobs at once: strong+fast
    limits intra-burst spikes but re-ignites bursts too soon; strong+slow gives
    a long inter-burst interval but suppresses inter-burst firing AND fires many
    spikes per burst. Splitting them (fast -> few spikes/burst, slow -> long
    IBI) decouples spike count, IBI, and inter-burst rate, each at biophysically
    reasonable time constants.

Parameters
    ainc_fast, tau_fast   fast SFA increment (uS) and decay (ms, ~100-300).
    ainc_slow, tau_slow   slow AHP increment (uS) and decay (ms, ~1000-4000).
    ek                    K+ reversal (mV, ~ -90).

Driven by a self-NetCon from the cell's own soma voltage (one per cell).
Units: g in microsiemens (uS), i in nanoamps (nA).

KNOWN SIMPLIFICATIONS (documented, not bugs -- see MODEL_CHARACTERIZATION.md)
    1. NONSPECIFIC_CURRENT with a private ek. Electrically this is a K+
       conductance, but it does not USEION k, so its current never enters ik and
       is invisible to the kdyn extracellular-K+ pool. Its ek is also pinned at
       -90 mV while kdyn writes ek = -77 mV for hh/kA, so two K+ reversals
       coexist in one compartment.
    2. Linear and UNSATURATING. The real I_sAHP is gated by a nonlinear Ca2+
       sensor (hippocalcin; Tzingounis et al. 2007 Neuron 53:487) and saturates.
       Here the steady-state load is g = ainc * rate * tau with no ceiling.
    3. NO temperature scaling. kA.mod carries a q10 and hh is temperature
       dependent, but tau_fast/tau_slow here are fixed while h.celsius = 6.3.
    4. ainc is uS PER SPIKE, not a channel density, so it lumps channel density,
       Ca2+ influx per spike and Ca2+ sensitivity into one number. A
       channelopathy would move only the first.
References: fast SFA -- Brown & Adams (1980) I_M; Pospischil et al. (2008).
slow AHP -- Sah (1996); Gulledge et al. (2013). Adaptation form -- Brette &
Gerstner (2005).
-----------------------------------------------------------------------------
ENDCOMMENT

NEURON {
    POINT_PROCESS sAHP
    RANGE ainc_fast, tau_fast, ainc_slow, tau_slow, ek, g_fast, g_slow, i
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    ainc_fast = 0.003 (uS)
    tau_fast  = 200   (ms)
    ainc_slow = 0.001 (uS)
    tau_slow  = 2000  (ms)
    ek        = -90   (mV)
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    g_fast (uS)
    g_slow (uS)
}

INITIAL {
    g_fast = 0
    g_slow = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = (g_fast + g_slow) * (v - ek)
}

DERIVATIVE state {
    g_fast' = -g_fast / tau_fast
    g_slow' = -g_slow / tau_slow
}

NET_RECEIVE(weight) {
    g_fast = g_fast + ainc_fast
    g_slow = g_slow + ainc_slow
}
