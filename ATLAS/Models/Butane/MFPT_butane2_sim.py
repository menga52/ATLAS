import numpy as np
from time import time
from ATLAS.Model.set_well import set_well

seed = 1
default_rng = np.random.default_rng(seed)

def MFPT_butane2_sim(weighted_dd2, well_threshold, output, N_IC, phi_store, chart, chart_sim_parameter,rng=-1,verbose=False):
  if rng==-1:
    rng = default_rng
  d = chart_sim_parameter["d"]
  dt_s = chart_sim_parameter["dt_s"]
  connectivity = chart_sim_parameter["connectivity"]

  connectivity_indices = np.empty(len(connectivity))
  for k in range(len(connectivity)):
    connectivity_indices[k] = [j for j in range(len(connectivity[k])) if connectivity[k][j]!=0]
  
  sqrtdts = np.sqrt(dt_s)
  X_int_store = set_well(output, N_IC, well_threshold)
  FPT_sim = np.empty(N_IC)
  exit_phi = np.empty(N_IC) # -1 means left point, +1 means right point
  if verbose:
    print("Use ATLAS simulator with",N_IC,"initial points")
  
  def loop(i):
    RAND_RESERVOID_MAX_SIZE = 10**8
    RAND_RESERVOIR_SIZE = 10**5
    rand_reservoir = rng.standard_normal((d,RAND_RESERVOIR_SIZE))*sqrtdts
    rand_reservoir_index = 0

    X_curr = X_int_store[i][:]
    phi_curr = np.arctan2(X_curr[5],X_curr[3])
    nearest = np.argmin(np.abs(phi_store-phi_curr))
    neigh = connectivity_indices[nearest]

    step = 0

    while phi_curr<well_threshold[1] and phi_curr>well_threshold[0]:
      X_curr_proj, b, _, H_hat, _, nearest, neigh = weighted_dd2(X_curr, chart, neigh, nearest, connectivity_indices)
      X_curr = np.transpose(X_curr_proj) + b*dt_s + H_hat @ rand_reservoir[:,rand_reservoir_index]
      X_curr = np.transpose(X_curr)
      phi_curr = np.arctan2(X_curr[5],X_curr[3])
      rand_reservoir_index += 1
      step += 1
      if rand_reservoir_index > RAND_RESERVOIR_SIZE:
        if RAND_RESERVOIR_SIZE*2 < RAND_RESERVOID_MAX_SIZE:
          RAND_RESERVOIR_SIZE *= 2
        rand_reservoir = rng.standard_noraml((d,RAND_RESERVOIR_SIZE))*sqrtdts
        rand_reservoir_index = 1
    
    exit_phi[i] = phi_curr
    FPT_sim[i] = step*dt_s
  
  t_start = time()
  #TODO: parallelize
  for i in range(N_IC):
    loop(i)
  
  t_elapsed = time() - t_start
  if verbose:
    print("The time spent is",t_elapsed/3600, "hours")
    # np.std underestimates matlab's std() because it uses sqrt(n) in the denominator instead of sqrt(n-1)
    print("MFPT in the ATLAS simulator is", np.mean(FPT_sim),"+-",np.std(FPT_sim)*1.96/np.sqrt(N_IC))
  
  return FPT_sim, exit_phi, t_elapsed