: HH calcium-mediated potassium current

NEURON{
	SUFFIX ikCa
	NONSPECIFIC_CURRENT i
	USEION k READ ek WRITE ik
	RANGE g, gCa, rev_ca, tau_ca,i
}

UNITS{
	(mS) = (millisiemens)
	(mV) = (millivolt)
	(uA) = (microamp)
}

PARAMETER{ 
	g = 5.0 (mS/cm2)
	gCa = 0.1 (mS/cm2)
	rev_ca = 120 (mV)
	cai_init = 0.000 (mM)
	tau_ca = 80 (ms)
}

ASSIGNED{
	v (mV)
	ek (mV)
	ik (uA/cm2)
	i (uA/cm2)
}

STATE {xcai}
BREAKPOINT{
	SOLVE states METHOD cnexp

	: ik = g*(v-ek)*(xcai/(1.0+xcai))*0.001 :factor to convert to mA/cm2
	ik = g*(v-ek)*(xcai/(1.0+xcai))*0.001 
	i = current_ca(v)*0.001	
	:i = 0
}

INITIAL {
	xcai= cai_init
}
DERIVATIVE states{
	: xcai' = -0.002*gCa*(v-rev_ca)/(1+exp(-(v+25)/2.5)) - xcai/tau_ca
	: xcai' = -0.002*gCa*(v-rev_ca)/gating_kca(v) - xcai/tau_ca
	xcai' = -0.002*current_ca(v) - xcai/tau_ca

}

FUNCTION current_ca(Vm (mV)){
	LOCAL x
	UNITSOFF
	current_ca = gCa*(Vm-rev_ca)/gating_kca(Vm)
	UNITSON
}

FUNCTION gating_kca(Vm (mV)){
	LOCAL x
	UNITSOFF
	x = -(Vm+25)/2.5
	if (x>700){
		x=700
	: printf("gating_kca, time: %g, current: %g, voltage: %g \n", t, i, Vm)

	}
	gating_kca = 1+exp(x)
	UNITSON
}
