: HH leak currents

NEURON{
	SUFFIX ileak
	NONSPECIFIC_CURRENT i
	RANGE glna, glcl, rev_na, rev_cl
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	glcl = 0.15 (mS/cm2)
	glna = 0.025 (mS/cm2)
	rev_na = 70 (mV)
	rev_cl = -74.0 (mV)
}

ASSIGNED{
	v (mV)
	i (uA/cm2)
}

BREAKPOINT{
	: SOLVE state METHOD cnexp

	i = (glcl*(v-rev_cl) + glna*(v-rev_na))*0.001 :factor to convert to mA/cm2
}

