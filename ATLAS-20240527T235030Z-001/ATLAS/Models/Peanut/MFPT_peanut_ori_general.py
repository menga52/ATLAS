import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

seed = 1
default_rng = np.random.default_rng(seed)
pi = 3.14159

def MFPT_peanut_ori_general(well_threshold, X_int_store,RHS_parameter,chart_angle,rng=-1,verbose=False):
  """
  well_threshold - 
  RHS_parameter - dictionary, comes from PeanutParameters.py in this directory
  chart_angle - 
  rng (optional) - random num gen objection
  verbose - if true, printing for debugging will occur
  """
  if rng==-1:
    rng = default_rng
  drift               = RHS_parameter["drift"]
  diffusion           = RHS_parameter["diffusion"]
  dt                  = RHS_parameter["dt"]
  t0                  = RHS_parameter["t0"]
  ratio               = round(t0/dt)
  D                   = RHS_parameter["D"]
  sqrtdt              = np.sqrt(dt);
  N_IC = X_int_store.shape[0] # assumes that X_int_store is of type np.ndarray
  FPT_ori             = np.zeros((1, N_IC))
  chart_angle         = np.transpose(chart_angle)
        
  # if true, rng('default'); rng(1); end        
  print('Use Original Simulator with', N_IC,'initial points')
     
  tstart = time.time();

  # inner helper function
  def loop(i):
    RAND_RESERVOID_MAX_SIZE = 10**8
    RAND_RESERVOIR_SIZE     = 5*10**6
    rand_reservoir          = rng.standard_normal((D,RAND_RESERVOIR_SIZE)) * sqrtdt
    rand_reservoir_index    = 1
    X_curr                  = np.transpose(X_int_store[i,:])
    step = 0

    while True: #???       
      drift_curr        = drift(X_curr)
      diffusion_curr    = diffusion(X_curr)

      drift_current     = drift_curr * dt
      diffusion_current = diffusion_curr @ rand_reservoir[:,rand_reservoir_index]
      X_curr            = X_curr +  drift_current + diffusion_current
              
      # check whether current point is in the well or not
      if step%ratio == 0:
        theta_X_curr           = np.arctan2(np.sqrt(X_curr[0]**2 + X_curr[1]**2 ), X_curr[2])
        phi_X_curr             = np.arctan2(X_curr(2), X_curr(1)) % (2*pi)
        angle                  = np.asarray([phi_X_curr, theta_X_curr])
        neighbors_obj = NearestNeighbors(n_neighbors=1,algorithm='auto').fit(chart_angle)
        distances, indices = neighbors_obj.kneighbors(angle)
        index_of_nearest = indices[0][0]
        nearest = chart_angle[index_of_nearest]
        if nearest not in well_threshold:
          break
              
             
      rand_reservoir_index += 1
      step += 1
             
      if rand_reservoir_index > RAND_RESERVOIR_SIZE:
        if RAND_RESERVOIR_SIZE*2 < RAND_RESERVOID_MAX_SIZE:
          RAND_RESERVOIR_SIZE = RAND_RESERVOIR_SIZE*2
        rand_reservoir       = rng.standard_normal((D,RAND_RESERVOIR_SIZE))*sqrtdt
        rand_reservoir_index = 1
       
    FPT_ori[i] = step*dt;
  # parfor i = 1:N_IC 
  # TODO: parallelize
  for i in range(N_IC):
    loop(i)

  t_final      = time.time() - tstart;
  if verbose:
    print('The time spent is', t_final/3600, 'hours')
    print('MFPT in the original simulator is', np.mean(FPT_ori),'+-',np.std(FPT_ori)*1.96/np.sqrt(N_IC))
  return FPT_ori, t_final
        
end