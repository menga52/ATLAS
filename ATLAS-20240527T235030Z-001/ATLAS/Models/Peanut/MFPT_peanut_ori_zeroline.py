import numpy as np
import time 
from ATLAS.Models.Peanut.phitheta import phitheta

seed = 1
default_rng = np.random.default_rng(seed)

def MFPT_peanut_ori_zeroline(X_int_store,RHS_parameter,phi_zero, theta_zero,rng=-1,verbose=False):
  """
  X_int_store - 
  RHS_parameter - dictionary object from PeanutParameters
  phi_zero - 
  theta_zero - 
  rng (optional) - random num gen
  verbose - whether to print debugging messages
  """
  if rng==-1:
    rng = default_rng
  drift               = RHS_parameter["drift"]
  diffusion           = RHS_parameter["diffusion"]
  dt                  = RHS_parameter["dt"]
  t0                  = RHS_parameter["t0"]
  ratio               = round(t0/dt)
  D                   = RHS_parameter["D"]
  sqrtdt              = np.sqrt(dt)
  N_IC = X_int_store.shape[0] # assumes that X_int_store is of type np.ndarray
  FPT_ori = np.zeros((1, N_IC))

  # if true, rng('default'); rng(1); end      
  print('Use Original Simulator with', N_IC, 'initial points')
        
  tstart = time.time()

  def loop(i):     
    RAND_RESERVOID_MAX_SIZE = 10**8
    RAND_RESERVOIR_SIZE     = 5*10**6
    rand_reservoir          = rng.standard_normal((D,RAND_RESERVOIR_SIZE))*sqrtdt
    rand_reservoir_index    = 1
    X_curr                  = np.transpose(X_int_store[i,:])
    step     = 0
    ii       = 1
         
    index_int = phitheta(X_curr, phi_zero, theta_zero);
         
    while True:   
      drift_curr = drift(X_curr)
      diffusion_curr = diffusion(X_curr)
      drift_current = drift_curr*dt
      diffusion_current = diffusion_curr * rand_reservoir[:,rand_reservoir_index]
      X_curr = X_curr +  drift_current + diffusion_current
      if step % ratio == 0:
        [index_curr] = phitheta(X_curr, phi_zero, theta_zero);
        if index_curr != index_int:
          break
           
      rand_reservoir_index = rand_reservoir_index+1;
      step = step + 1;
            
      if rand_reservoir_index > RAND_RESERVOIR_SIZE:
        if RAND_RESERVOIR_SIZE*2 < RAND_RESERVOID_MAX_SIZE:
          RAND_RESERVOIR_SIZE = RAND_RESERVOIR_SIZE*2
        rand_reservoir        = rng.standard_normal((D,RAND_RESERVOIR_SIZE))*sqrtdt
        rand_reservoir_index  = 1
      
    FPT_ori[i] = step*dt;
  #parfor i = 1:N_IC 
  for i in range(N_IC):
    loop(i)
  t_final      = time.time()- tstart
  if verbose:
    print('The time spent is', t_final/3600, 'hours')
    print('MFPT in the original simulator is', np.mean(FPT_ori),'+-',np.std(FPT_ori)*1.96/np.sqrt(N_IC))
  return FPT_ori, t_final