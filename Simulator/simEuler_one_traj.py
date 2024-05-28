import numpy as np
from ATLAS import constants
# TODO: find seed
seed = 1
default_rng = np.random.default_rng(seed)

def sqrt(x):
  return x**0.5

def simEuler_one_traj(RHS_parameter,Sim_parameter,verbose=False,rng=-1):
  if rng==-1:
    rng = default_rng
    print("simEuler_one_traj default rng")
  
  diffusion = RHS_parameter["diffusion"]
  drift = RHS_parameter["drift"]
  D = RHS_parameter["D"]
  X_int = Sim_parameter["X_int"]
  T_max = Sim_parameter["T_max"]
  dt = Sim_parameter["dt"]
  gap= Sim_parameter["gap"]
  
  # different from original code. matlab rounds to higher magnitude, np rounds to even integer
  tN = np.round(T_max/dt)
  
  batch = 10**8 #TODO
  # batch = 10
  if verbose and (tN/gap) % 1 != 0:
    print("non-integer tN/gap")
  X = np.zeros((D,int(tN/gap+1)))
  X[:,0] = X_int[0]
  index_store = 0
  sqrtdt = sqrt(dt)
  Current = np.transpose(X_int) #transpose?
  # Current = X_int.reshape()
  
  j_total = int(np.floor(tN/batch))
  rem = int(tN - j_total*batch)
  if verbose:
    print("Total has",j_total,"batches.")
    
  for j in range(j_total):
    if verbose:
      print("Batch No.", j)
    r_store = rng.standard_normal((D,batch))*sqrtdt
    for i in range(batch):
      drift_current = drift(Current)*dt
      diffusion_current = diffusion(Current) @ r_store[:,i].reshape(-1,1)
      Current = Current + drift_current + diffusion_current
      if i%gap == 0:
        X[:,index_store+1] = Current.reshape(-1)
        index_store += 1
      
  r_store = rng.standard_normal((D,rem))*sqrtdt
  for i in range(rem):
    drift_current = drift(Current)*dt
    diffusion_current = diffusion(Current) @ r_store[:,i].reshape(-1,1)
    diffusion_current = diffusion_current.reshape(-1,1)
    Current = Current + drift_current + diffusion_current
    if i%gap == 0:
      X[:,index_store+1] = Current.reshape(-1)
      index_store += 1
  
  return np.transpose(X)