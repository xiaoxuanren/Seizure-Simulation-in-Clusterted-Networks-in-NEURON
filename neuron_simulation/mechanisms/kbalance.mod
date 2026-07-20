: K balance

NEURON{
	SUFFIX kbalance
	:NONSPECIFIC_CURRENT i
	USEION k READ  ki WRITE ki, ko
	RANGE  ko_init, ki_init
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
	(molar) = (1/liter)
	(mM) = (millimolar)
}
PARAMETER{
	ko_init = 2 (mM)
	ki_init =133 (mM)
}
ASSIGNED{
	ki (mM)
	ko (mM)
}
INITIAL{
	ki = ki_init
	ko = ko_init
}

