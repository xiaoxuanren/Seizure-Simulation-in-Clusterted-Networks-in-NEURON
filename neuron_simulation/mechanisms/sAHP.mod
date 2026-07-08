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
        INTERVAL.

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
