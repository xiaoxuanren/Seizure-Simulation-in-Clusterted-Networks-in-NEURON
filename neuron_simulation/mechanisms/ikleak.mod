: HH potassium leak current

NEURON{
	SUFFIX ikleak
	:NONSPECIFIC_CURRENT i
	USEION k READ ek WRITE ik
	RANGE g,i
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	g = 0.035 (mS/cm2)
}

ASSIGNED{
	v (mV)
	ek (mV)
	i (uA/cm2)
	ik (uA/cm2)
}
BREAKPOINT{

	 ik = g*(v-ek)*0.001 :factor to convert to mA/cm2
	:ik = i
	:ik = g*(v-ek)
	:i = 0
	:	printf("Kleak: %g \t %g \t %g \n", t, ik, v)

}

