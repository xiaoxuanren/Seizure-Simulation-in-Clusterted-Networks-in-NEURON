: HH persistent sodium current

NEURON{
	SUFFIX isodiumP
	NONSPECIFIC_CURRENT i
	RANGE g, rev_na
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	g = 0.2 (mS/cm2)
	j_init = 0.05
	rev_na = 70 (mV)
}

ASSIGNED{
	v (mV)
	i (uA/cm2)
}

STATE {jj}
BREAKPOINT{
	SOLVE states METHOD cnexp
	i = g*(v-rev_na)*jj*0.001 : factor to convert to mA/cm2
}
INITIAL {
	jj = j_init
}
DERIVATIVE states{
	jj' = 5.0*(jinf(v)-jj)
}

FUNCTION jinf(Vm (mV)){
	UNITSOFF
		jinf = 1/(1+exp(-(Vm+42.0)/5.0))
		UNITSON
}

