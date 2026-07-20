: External current

NEURON{
	SUFFIX iext
	NONSPECIFIC_CURRENT i
	RANGE iext
}

UNITS{	
	(uA) = (microamp)
	(molar) = (1/liter)
}
PARAMETER{
	iext = 0.36 (uA/cm2)
}
ASSIGNED{
	i (uA/cm2)
}

BREAKPOINT{
	i = -iext

}

