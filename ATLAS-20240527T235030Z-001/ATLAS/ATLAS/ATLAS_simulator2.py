import numpy as np
from ATLAS.ATLAS.Learning_Slow_Manifold import Learning_Slow_Manifold
from ATLAS.ATLAS.landmark_add import landmark_add
seed = 1
default_rng = np.random.default_rng(seed)

def ATLAS_simulator2(weighted_dd2, chart_sim_parameter, RHS_parameter, simulator_par, chart, rng=-1):
  if rng == -1:
    rng = default_rng
  X_int                    = chart_sim_parameter["X_int"]
  dt_s                     = chart_sim_parameter["dt_s"]
  D                        = chart_sim_parameter["D"]
  d                        = chart_sim_parameter["d"]
  Nstep                    = chart_sim_parameter["Nstep"]
  nearest                  = chart_sim_parameter["nearest"]
  connectivity             = chart_sim_parameter["connectivity"]
  gap                      = chart_sim_parameter["gap"]
  explore_threshold        = chart_sim_parameter["explore_threshold"]
  
  connectivity_indices = np.empty(connectivity.shape[0],dtype=object)
  # for k in range(connectivity.shape[0]):
  #   temp = np.where(connectivity_indices>0)
  #   found_indices = zip(temp[0],temp[1])
  #   connectivity_indices[k] = found_indices
  for k in range(len(connectivity[0])):
    connectivity_indices[k] = np.where(connectivity[k][:])

  T_max = RHS_parameter["T_max"]
  N = RHS_parameter["N"]
  dt = RHS_parameter["dt"]
  t0 = RHS_parameter["t0"]
  chi_p = RHS_parameter["chi_p"]
  threshold = RHS_parameter["threshold"]
  modify = RHS_parameter["modify"]
  connectivity_threshold = RHS_parameter["connectivity_threshold"]

  X = np.zeros((np.round(Nstep/gap), D))
  neigh = connectivity_indices[nearest]
  X_curr = X_int
  index_store = 1
  r_store = rng.standard_normal((d,Nstep))*np.sqrt(dt_s)
  nearest_store = np.zeros(np.round(Nstep/gap)) # added rounding

  for j in range(Nstep):
    X_curr_proj, b, _, H_hat, T, nearest, neigh = weighted_dd2(X_curr, chart, neigh, nearest, connectivity_indices)
    if len(np.where(T<explore_threshold)[0])==0:
      # ^ if nowhere is T>=explore_threshold
      print("At step",j,"add the landmark",len(chart)+1,"to the ATLAS")
      Sim_parameter = {}
      Sim_parameter["T_max"] = T_max
      Sim_parameter["dt"] = dt
      Sim_parameter["N"] = N
      Sim_parameter["X_int"] = X_curr_proj

      data, Cov_store, Mean_store = simulator_par(Sim_parameter)
      chart_add = Learning_Slow_Manifold(data, Cov_store, Mean_store, D, d, modify)
      chart, connectivity, _ = landmark_add(chart, connectivity, chart_add, t0, chi_p, threshold[0], connectivity_threshold)
      connectivity_indices = np.empty(connectivity.shape[0])
      
      for k in range(len(connectivity[0])):
        connectivity_indices[k] = np.where(connectivity[k][:])
      
      nearest = len(chart) # the added landmark is the last one in the chart
      neigh = connectivity_indices[nearest]
      X_curr_proj = chart_add["X_int"]
      b = chart_add["b"]
      H_hat = chart_add["U"] @ np.diag(chart_add.sigma)

      X_curr = np.transpose(X_curr_proj) + b @ dt_s + H_hat @ r_store[:][j]
      X_curr = np.transpose(X_curr)
      if j % gap == 0:
        X[index_store][:] = X_curr_proj
        nearest_store[index_store] = nearest
        index_store += 1
  return X, nearest_store, chart
      




