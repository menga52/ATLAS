import numpy as np
import time
from ATLAS.Models.Peanut.phitheta import phitheta

seed = 1
default_rng = np.random.default_rng(seed)

def MFPT_peanut_sim_zeroline(weighted_dd2, X_int_store, nearest_store, chart,chart_sim_parameter,phi_zero, theta_zero,rng=-1,verbose=False):
  if rng==-1:
    rng = default_rng
  d = chart_sim_parameter["d"]
  dt_s = chart_sim_parameter["dt_s"]
  connectivity = chart_sim_parameter["connectivity"]
        
  connectivity_indices = np.empty(1,len(connectivity),dtype=object)
  for k in range(len(connectivity)):
    connectivity_indices[k] = [i for i in range(len(connectivity[k])) if connectivity[k][i]!=0]
        
  sqrtdts = np.sqrt(dt_s);
  N_IC = X_int_store.shape[0]
  FPT_sim = np.zeros(1, N_IC)
  # if true, rng('default'); rng(1); end
  print('Use ATLAS Simulator with', N_IC, 'initial points')
  tstart= time.time();
 
  
  def loop(i):
    RAND_RESERVOID_MAX_SIZE = 10**8
    RAND_RESERVOIR_SIZE     = 10**5
    rand_reservoir          = rng.standard_normal((d,RAND_RESERVOIR_SIZE)) * sqrtdts
    rand_reservoir_index    = 1
    
    X_curr                  = X_int_store[i,:]
    index_int             = phitheta(X_curr, phi_zero, theta_zero)

    step                    = 0
    nearest                 = nearest_store(i);
    neigh                   = connectivity_indices[nearest]
    while True:
      X_curr_proj , b,_, H_hat,_, nearest, neigh = weighted_dd2( X_curr, chart, neigh, nearest,connectivity_indices)
      X_curr = np.transpose(X_curr_proj) + b @ dt_s + H_hat @ rand_reservoir[:,rand_reservoir_index]
      X_curr = np.transpose(X_curr)
            
      index_curr = phitheta(X_curr, phi_zero, theta_zero); 
      if index_curr != index_int:
        break
      rand_reservoir_index += 1
      step += 1
      if rand_reservoir_index > RAND_RESERVOIR_SIZE:
        if RAND_RESERVOIR_SIZE*2 < RAND_RESERVOID_MAX_SIZE:
          RAND_RESERVOIR_SIZE = RAND_RESERVOIR_SIZE*2
        rand_reservoir = rng.standard_normal(d,RAND_RESERVOIR_SIZE) * sqrtdts
        rand_reservoir_index = 1
    FPT_sim[i] = step*dt_s
  # parfor i = 1:N_IC 
  for i in range(N_IC):
    loop(i)
  t_final = time.time() - tstart
  if verbose:
    print('The time spent is', t_final/3600, 'hours')
    print('MFPT in the original simulator is', np.mean(FPT_sim),'+-',np.std(FPT_sim)*1.96/np.sqrt(N_IC))
  return FPT_sim, t_final