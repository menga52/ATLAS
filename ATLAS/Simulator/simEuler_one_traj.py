import numpy as np
from ATLAS.constants import *
# TODO: find seed
default_rng = np.random.default_rng(seed)

def sqrt(x):
	return x**0.5

def simEuler_one_traj(RHS_parameter,Sim_parameter,verbose=False,rng=-1):
	if rng==-1:
		rng = default_rng
	
	diffusion = RHS_parameter["diffusion"]
	drift = RHS_parameter["drift"]
	diffusion = RHS_parameter["D"]
	X_int = Sim_parameter["X_int"]
	T_max = Sim_parameter["T_max"]
	dt = Sim_parameter["dt"]
	gap= Sim_parameter["gap"]
	
	# different from original code. matlab rounds to higher magnitude, np rounds to even integer
	tN = np.round(T_max/dt)
	
	batch = 10**8
	X = np.zeros(D,tN/gap+1)
	X[:,0] = X_int
	index_store = 0
	sqrtdt = sqrt(dt)
	Current = X_int #transpose?
	
	j_total = np.floor(tN/batch)
	rem = tN - j_total*batch
	if verbose:
		print("Total has",j_total,"batches.")
		
	for j in range(j_total):
		if verbose:
			print("Batch No.", j)
		
		for i in range(batch):
			drift_current = drift(Current)*dt
			diffusion_current = diffusion(Current)
			Current = Current + drift_current + diffusion_current
			if i%gap == 0:
				X[:,index_store+1] = Current
				index_store += 1
			
	r_store = np.standard_normal((D,rem))*sqrtdt
	for i in range(rem):
		drift_current = drift(Current)*dt
		diffusion_current = diffusion(Current) @ r_store[:,i]
		Current = Current + drift_current + diffusion_current
		if i%gap == 0:
			X[:,index_store+1] = Current
			index_store += 1
	
	return np.transpose(X)