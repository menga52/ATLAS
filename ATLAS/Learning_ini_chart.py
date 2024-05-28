import numpy as np
from time import time
from ATLAS import utilities

from ATLAS.ATLAS.Learning_Slow_Manifold import Learning_Slow_Manifold
seed = 1
default_rng = np.random.default_rng(seed)

"""
parameters - Parameters object (PeanutParameters, etc)
    output:
chart - a list of 
"""
def Learning_ini_chart(parameters,warmup = 10000, verbose=False,rng=-1):
  if rng == -1:
    rng = default_rng
  K_int = parameters.K_int
  chart = np.empty(K_int, dtype=object)
  Sim_parameter = {}
  Sim_parameter["T_max"] = parameters.T_one
  Sim_parameter["dt"] = parameters.dt
  Sim_parameter["gap"] = 1
  Sim_parameter["X_int"] = parameters.X_int

  if verbose:
    print("Initial point:",Sim_parameter["X_int"]," T_max:",Sim_parameter["T_max"])
  start_time = time()
  simulator_one = parameters.simulator_one
  output = simulator_one(Sim_parameter,verbose=verbose,rng=rng)
  stop_time = time()
  time_elapsed = stop_time - start_time
  if verbose:
    print("one long trajectory is simulated. The time spent is",time_elapsed/60,"minutes.")

  # evenly sample N initial points from the trajectory
  output = output[warmup:][:] # [:] from original code
  L = output.shape[0]
  # X_int_sample = output[0:np.round(L/K_int):][:]
  X_int_sample = [output[i] for i in \
      range(0,L,int(utilities.round(L/K_int)))]
      # cast *should* be unnecessary (but isn't)
  X_int_sample = np.asarray(X_int_sample)

  time_start = time()
  if verbose:
    print("Parallelly run chart simulator starting from",K_int,"initial points")
  simulator_par = parameters.simulator_par
  for i in range(K_int):
    Sim_parameter = {}
    Sim_parameter["T_max"] = parameters.T_max
    Sim_parameter["dt"] = parameters.dt
    Sim_parameter["N"] = parameters.N
    Sim_parameter["X_int"] = X_int_sample[i]

    data, Cov_store, Mean_store = simulator_par(Sim_parameter,verbose=verbose,rng=rng)
    D = parameters.D
    d = parameters.d
    modify = parameters.modify
    # print("data",data)
    # print("Cov_store.shape",Cov_store.shape)
    # print("Mean_store.shape",Mean_store.shape)

    
    chart[i] = Learning_Slow_Manifold(data, Cov_store, Mean_store, D,d,modify)
    if verbose:
      print("Landmark",i,"is learned.")
  
  time_stop = time()
  time_elapsed = time_stop-time_start
  if verbose:
    print("Initial learning stage is completed. The time spent is",time_elapsed/60,"minutes")
  return chart