: HH potassium m_current

NEURON{
	SUFFIX ipotassiumM
	: NONSPECIFIC_CURRENT i
	USEION k READ ek WRITE ik
	RANGE g,i
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	g = 0.02 (mS/cm2)
	l_init = 0.1
}

ASSIGNED{
	v (mV)
	ek (mV)
	ik (uA/cm2)
	i (uA/cm2)
}

STATE {l}
BREAKPOINT{
	SOLVE states METHOD cnexp
	ik = g*(v-ek)*l*0.001 : factor to convert to mA/cm2
:	printf("time: %g, current: %g, voltage: %g, ek: %g \n", t, i, v, ek)
	:	printf("potassium: %g \t %g \t %g \n", t, ik, v)
}


INITIAL {
		l=l_init
}
DERIVATIVE states{
	l' = (linf(v)-l)/taul(v)
}

FUNCTION alpha(Vm (mV)) (/ms){
		LOCAL x,fac1,fac2
		x=Vm+30.0
		fac1 = 0.001
		fac2 = 9.0
		UNITSOFF
		if (fabs(x)<1e-6){
			alpha=fac1*fac2
		}
		else{
			alpha = fac1*x/(1-exp(-x/fac2))
		}
	UNITSON
}

FUNCTION beta(Vm (mV)) (/ms){
		LOCAL x,fac1,fac2
		x=Vm+30.0
		fac1 = 0.001
		fac2 = 9.0
		UNITSOFF
		if (fabs(x) < 1e-6){
			beta=fac1*fac2
		}
		else{
			beta = -fac1*x/(1-exp(x/fac2))
		}
		UNITSON
}

FUNCTION linf(Vm (mV)){
	UNITSOFF
		linf=alpha(Vm)/(alpha(Vm)+beta(Vm))
		UNITSON
}

FUNCTION taul(Vm (mV)){
	UNITSOFF
		taul=0.34/(alpha(Vm)+beta(Vm))
		UNITSON
}

