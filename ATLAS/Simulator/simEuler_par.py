import numpy as np
seed = 1
default_rng = np.random.default_rng(seed)

def simEuler_par(RHS_parameter, Sim_parameter,rng=-1):
	if rng==-1:
		rng = default_rng
	diffusion = RHS_parameter["diffusion"]
	drift = RHS_parameter["drift"]
	D = RHS_parameter["D"]
	UpperBound = RHS_parameter["UpperBound"]
	LowerBound = RHS_parameter["LowerBound"]
	X_int = Sim_parameter["X_int"]
	T_max = Sim_parameter["T_max"]
	dt = Sim_parameter["dt"]
	N = Sim_parameter["N"]
	
	t_span = range(0,T_max+dt,dt)
	tN = len(t_span)-1
	
	X = np.zeros(N,D,tN*N+1)
	sqrtdt = dt**0.5
	
	