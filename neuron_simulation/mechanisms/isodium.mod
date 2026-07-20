: HH sodium current

NEURON{
	SUFFIX isodium
	NONSPECIFIC_CURRENT i
	RANGE g, rev_na
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	g = 35.0 (mS/cm2)
	h_init = 0.05
	rev_na = 70 (mV)
}

ASSIGNED{
	v (mV)
	i (uA/cm2)
}

STATE {h}
BREAKPOINT{
	SOLVE states METHOD cnexp
	:SOLVE states METHOD euler

	:SOLVE states METHOD runge
	:SOLVE states METHOD derivimplicit

	i = g*(v-rev_na)*(m_infinity(v)^3)*h*0.001 : factor to convert to mA/cm2
}
INITIAL {
	h = h_init
}
DERIVATIVE states{
	h' = ((1-h)*alpha(v) - h*beta(v))/0.2
}

FUNCTION m_infinity(Vm (mV)){
	LOCAL alpha_m, beta_m, x, y, x1

	x1 = -(Vm+60.0)/18.0
	if (x1>700){
		x1 = 700
:		printf("sodium beta_m, time: %g, current: %g, voltage: %g \n", t, i, Vm)

	}
	: beta_m = 4.0*exp(-(Vm+60.0)/18.0)
	beta_m = 4.0*exp(x1)


	x = Vm+35.0
	y = -0.1*x
	if (y > 700){
		:alpha_m = 0
		y = 700
:		printf("sodium alpha_m, time: %g, current: %g, voltage: %g \n", t, i, Vm)


	}
	if (fabs(y)>1e-6){
		alpha_m = y/(exp(y)-1)
	}
	else{
		alpha_m = 1/(1+y/2.0+y*y/6.0)
	}


 	m_infinity = alpha_m/(alpha_m+beta_m)
}

FUNCTION alpha(Vm (mV)) (/ms){
	LOCAL x
	UNITSOFF
	x = -(Vm+58.0)/20.0
	if (x>700){
		x=700
:		printf("sodium alpha, time: %g, current: %g, voltage: %g \n", t, i, Vm)

	}
	alpha = 0.07*exp(x)
	UNITSON
}

FUNCTION beta(Vm (mV)) (/ms){
	LOCAL x, y, x1
	UNITSOFF
	x = (Vm + 28.0)
	x1 = -0.1*x
	if (x1>700){
		x1=700
:		printf("sodium beta, time: %g, current: %g, voltage: %g \n", t, i, Vm)
	}
	y = exp(x1) + 1
	beta = 1.0/y
	UNITSON
}

