: HH potassium current

NEURON{
	SUFFIX ipotassium
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
	g = 9 (mS/cm2)
	n_init = 0.1
}

ASSIGNED{
	v (mV)
	ek (mV)
	ik (uA/cm2)
	i (uA/cm2)
}

STATE {n}
BREAKPOINT{
	SOLVE states METHOD cnexp
	:SOLVE states METHOD euler

	:SOLVE states METHOD runge
	:SOLVE states METHOD derivimplicit
	ik = g*(v-ek)*n*n*n*n*0.001 : factor to convert to mA/cm2
	 :ik = i
:	printf("time: %g, current: %g, voltage: %g, ek: %g \n", t, i, v, ek)
	:	printf("potassium: %g \t %g \t %g \n", t, ik, v)


	:ik = g*(v-ek)*n*n*n*n
	:i = 0
}
INITIAL {
	n = n_init
}
DERIVATIVE states{
	n' = ((1-n)*alpha(v) - n*beta(v))/0.2
}

FUNCTION alpha(Vm (mV)) (/ms){
	LOCAL x,y
	UNITSOFF
	x = (Vm + 34.0)/(-10)
	if (x > 700){
		: alpha = 0
		x = 700
	:	printf("potassium alpha, time: %g, current: %g, voltage: %g \n", t, i, Vm)

	}
	
	if (fabs(x)>1e-6){
		alpha = 0.1*x/(exp(x)-1)
	}
	else{
		alpha = 0.1/(1+x/2.0+x*x/6.0)
	}

	UNITSON

}

FUNCTION beta(Vm (mV)) (/ms){
	LOCAL x
	UNITSOFF
	x = -(Vm+44.0)/80.0
	if (x>700){
		x=700
:		printf("potassium beta, time: %g, current: %g, voltage: %g \n", t, i, Vm)
	}
	beta = 0.125*exp(x)
	UNITSON
}
