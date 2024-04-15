import numpy as np
from time import time

from ATLAS.ATLAS.Learning_Slow_Manifold import Learning_Slow_Manifold

#TODO: return???
def Learning_ini_chart(model, verbose=False):
  K_int = model.parameters.K_int
  chart = np.empty(K_int, dtype=object)
  Sim_parameter = {}
  Sim_parameter["T_max"] = model.parameters.T_one
  Sim_parameter["dt"] = model.parameters.dt
  Sim_parameter["gap"] = 1
  Sim_parameter["X_int"] = model.parameters.X_int

  if verbose:
    print("Initial point:",Sim_parameter["X_int"]," T_max:",Sim_parameter["T_max"])
  start_time = time()
  simulator_one = model.simulator_one
  output = simulator_one(Sim_parameter)
  stop_time = time()
  time_elapsed = stop_time - start_time
  if verbose:
    print("one long trajectory is simulated. The time spent is",time_elapsed/60,"minutes.")

  # evenly sample N initial points from the trajectory
  output = output[10000-1:][:] # [:] from original code
  L = output.shape[0]
  X_int_sample = output[0:np.round(L/K_int):][:]

  time_start = time()
  print("Parallelly run chart simulator starting from",K_int,"initial points")
  simulator_par = model.simulator_par
  for i in range(K_int):
    if verbose:
      print("Landmark",i,"is learned.")
    Sim_parameter = {}
    Sim_parameter["T_max"] = model.parameter.T_max
    Sim_parameter["dt"] = model.parameter.dt
    Sim_parameter["N"] = model.parameter.N
    Sim_parameter["X_int"] = X_int_sample[i]

    data, Cov_store, Mean_store = simulator_par(Sim_parameter)
    D = model.parameter.D
    d = model.parameter.d
    modify = model.parameter.modify
    chart[i] = Learning_Slow_Manifold(data, Cov_store, Mean_store, D,d,modify)
  
  time_stop = time()
  time_elapsed = time_stop-time_start
  if verbose:
    print("Initial learning stage is completed. The time spent is",time_elapsed/60,"minutes")