import numpy as np

default_rng = np.random.default_rng()

def simEulernonlintrans_one_traj(RHS_parameter,Sim_parameter,rng=-1):
  # This is Euler Method. Usually for Ito Integral
  # X is (#saved states) x dimension
  if rng==-1:
    rng = default_rng
  diffusion = RHS_parameter["diffusion"]
  drift = RHS_parameter["drift"]
  nonlin_trans = RHS_parameter["nonlin_trans"]
  nonlin_trans_inv = RHS_parameter["nonlin_trans_inv"]
  D = RHS_parameter["D"]
  X_int = Sim_parameter["X_int"]
  T_max = Sim_parameter["T_max"]
  dt = Sim_parameter["dt"]
  gap = Sim_parameter["gap"]

  tN = round(T_max/dt)
  batch = 10**8
  X                   = np.zeros((D,tN/gap+1))
  X[:,0]              = np.transpose(X_int)
  index_store         = 1
  sqrtdt              = np.sqrt(dt)
  # Current             = nonlin_trans_inv(X_int');
  Current = np.zeros(X_int.shape)
  for i in range(len(Current)):
    Current[i] = nonlin_trans_inv[X_int[i]]


  j_total    = np.floor(tN/batch);
  rem        = tN - j_total * batch ;
  print('Total has',j_total,'batches.')

  for j in range(j_total):
    print('Batch No.',j)
    r_store     = rng.standard_normal((D, batch)) * sqrtdt
    for i in range(batch):
      drift_current      = drift(Current) * dt
      diffusion_current  = diffusion(Current) * r_store[:, i]
      Current            = Current+drift_current+diffusion_current
      if i%gap == 0:
          #X[:,index_store+1] = nonlin_trans(Current);
          for i in range(len(X)):
            X[i][index_store+1] = nonlin_trans[Current[i]]
          index_store += 1
     

  r_store = rng.standard_normal((D, rem)) * sqrtdt
  for i in range(rem):
    drift_current     = drift(Current) * dt
    diffusion_current = diffusion(Current) * r_store [:,i]
    Current           = Current+drift_current+diffusion_current
    if i%gap == 0:
        #X[:,index_store+1] = nonlin_trans(Current)
        for i in range(len(X)):
          X[i][index_store+1] = nonlin_trans[Current[i]]
        index_store += 1

  X = np.transpose(X)
  return X