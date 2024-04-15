import numpy as np
from time import time

seed = 1
default_rng = np.random.default_rng(seed)

def MFPT_butane_ori(well_threshold, output, N_IC, RHS_parameter, chart_sim_parameter, rng=-1, verbose=False):
  drift = RHS_parameter.drift #RHS_parameter["drift"]
  diffusion = RHS_parameter.diffusion
  dt = RHS_parameter.dt
  dt_s = chart_sim_parameter.dt_s
  ratio = round(dt_s/dt)
  D = RHS_parameter.D
  sqrtdt = np.sqrt(dt)
  X_int_store = np.set_well(output,  N_IC, well_threshold)
  FPT_ori = np.zeros(N_IC)
  exit_phi = np.zeros(N_IC) # -1 means left point, +1 means right point

  t_start = time()
  
  def loop(i):
    RAND_RESERVOID_MAX_SIZE = 10**8
    RAND_RESERVOIR_SIZE = 5*10**6
    rand_reservoir = rng.standard_normal((D,RAND_RESERVOIR_SIZE))*sqrtdt
    rand_reservoir_index = 0
    X_curr = np.transpose(X_int_store[i][:])
    phi_curr = np.arctan2(X_curr[5], X_curr[3])
    step = 0

    while phi_curr < well_threshold[1] and phi_curr > well_threshold[0] or step%ratio == 0:
      drift_current = drift(X_curr)*dt
      diffusion_current = diffusion(X_curr) @ rand_reservoir[:][rand_reservoir_index]
      X_curr = X_curr + drift_current + diffusion_current
      phi_curr = np.arctan2(X_curr[5], X_curr[3])
      step += 1
      rand_reservoir_index += 1

      if rand_reservoir_index >= RAND_RESERVOIR_SIZE:
        if RAND_RESERVOIR_SIZE*2 < RAND_RESERVOID_MAX_SIZE:
          RAND_RESERVOIR_SIZE *= 2
        rand_reservoir = rng.standard_normal((D,RAND_RESERVOIR_SIZE))*sqrtdt
        rand_reservoir_index = 1
    exit_phi[i] = phi_curr
    FPT_ori[i] = step*dt
  
  # change to parallel
  for i in range(N_IC):
    loop(i)
  

  t_elapsed = time() - t_start
  if verbose:
    print("The time spent is", t_elapsed/3600, "hours")
    # np.std uses n in the denominator rather than n-1, as used by matlab's std()
    # so this version has smaller margin of error
    print("MFPT in the original simulator is", np.mean(FPT_ori),'+-',np.std(FPT_ori)*1.96/np.sqrt(N_IC))
  
  return FPT_ori, exit_phi, t_elapsed


