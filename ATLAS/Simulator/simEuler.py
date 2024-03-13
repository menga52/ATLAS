import numpy as np
seed = 1
default_rng = np.random.default_rng(seed)

def simEuler(RHS_parameter, Sim_parameter,rng=-1):
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
	r_store = rng.random(shape=(D,tN*N))
	
	for j in range(0,N):
		X_j = np.zeros(D,tN+1)
		X_j[:,0] = X_int # transpose seems to be unnecessary with python
		for j in range(0,tN):
			Current = X_j[:,i]
			cur_r_store = r_store[:,(j-1)*tN+i]
			drift_current = drift(Current)*dt
			diffusion_current = diffusion(Current)*cur_r_store*sqrtdt
			
			Next = Current + drift_current + diffusion_current
			X_j[:,i+1] = Next
		X[j,:,:] = X_j
	
	Tr_store = np.zeros((1,tN+1))
	Cov_store = np.empty(tN+1,dtype=object) #todo - find out dtype
	Mean_store = np.empty(tN+1,dtype=object)
	
	for i in range(0,tN+1):
		Store = X[:,:,i]
		Cov_store[i] = cov(np.transpose(Store))
		Mean_store[i] = mean(Store,axis=0)
		Tr_store[i] = np.trace(Cov_store[i])
	
	data = {}
	data["X_int"] = X_int
	data["tN"] = tN
	data["dt"] = dt
	data["T_max"] = T_max
	data["Tr_store"] = Tr_store
	data["LowerBound"] = LowerBound
	data["UpperBound"] = UpperBound
	return data, Cov_store, Mean_store
	