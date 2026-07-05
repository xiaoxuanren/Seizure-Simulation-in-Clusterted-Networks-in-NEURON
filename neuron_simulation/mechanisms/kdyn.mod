TITLE Dynamic extracellular K+ concentration ([K+]o accumulation, Cressman/Frohlich-style)
: Firing drives K+ efflux (ik>0 outward) -> [K+]o rises -> E_K depolarizes (Nernst)
: -> reduced K+ drive + depolarization -> more firing (SEIZURE positive feedback).
: Glial/diffusive clearance (tau_k) pulls [K+]o back toward rest -> negative feedback
: that terminates the ictal event. Impaired clearance (large tau_k) => seizure-prone.
NEURON {
    SUFFIX kdyn
    USEION k READ ik WRITE ek
    RANGE ko, ko_rest, tau_k, epsilon, ki
}
UNITS { (mV)=(millivolt) (mA)=(milliamp) (mM)=(milli/liter) }
PARAMETER {
    ki      = 72.0  (mM)     : intracellular K+ (held fixed)
    ko_rest = 4.0   (mM)     : baseline extracellular K+
    tau_k   = 200.0 (ms)     : clearance time constant (glia + diffusion)
    epsilon = 0.06           : lumped efflux coupling (ik -> d[K+]o/dt)
    ik_rest = 0.0006 (mA/cm2): resting outward K+ leak subtracted off
}
ASSIGNED { ik (mA/cm2)  ek (mV) }
STATE { ko (mM) }
INITIAL { ko = ko_rest  ek = 26.64*log(ko/ki) }
BREAKPOINT { SOLVE state METHOD cnexp  ek = 26.64*log(ko/ki) }
DERIVATIVE state {
    ko' = epsilon*(ik - ik_rest) - (ko - ko_rest)/tau_k
}
