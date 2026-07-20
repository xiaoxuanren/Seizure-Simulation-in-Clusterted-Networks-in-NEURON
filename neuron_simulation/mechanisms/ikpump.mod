: HH potassium pump current

NEURON{
	SUFFIX ikpump
	NONSPECIFIC_CURRENT i
	USEION k READ ko WRITE ik
	RANGE ikpumpmax, koeq
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER{ 
	: ikpumpmax = 5.0 (uA/cm2)
	ikpumpmax = 7.0 (uA/cm2)
	koeq = 3.0 (mM)	
}

ASSIGNED{
	ko (mM)
	i (uA/cm2)
	ik (uA/cm2)
}
BREAKPOINT{
	ik = -(ikpumpmax/(1+koeq/ko)^2)*0.001 : Fully mimic 2K+(in) -3Na+ (out) exchange
	i = -ik*1.5


	:ik = -(ikpumpmax/(1+koeq/ko)^2)*0.001 : factor to convert to mA/cm2 (Here ik doesn't cause extra current that directly contributes to dV/dt) 
	:i = -ik
	:	printf("Kpump: %g \t %g \t %g \n", t, ik, v)

	

}

