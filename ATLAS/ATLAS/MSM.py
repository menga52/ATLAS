import numpy as np

seed = 1
default_rng = np.random.default_rng(seed)

def MSM(chart, connectivity, MSM_parameter, weighted_dd2, d, rng=-1):
  if rng == -1:
    rng = default_rng
  print("Starting MSM part")

  # this function gives the transition matrix M = exp(P*step*t0)
  step     = MSM_parameter["step"]
  dt_s     = MSM_parameter["dt_s"]
  N        = MSM_parameter["N_state;"]
  K        = len(chart)
  TranM    = np.zeros(K,K)

  connectivity_indices = np.empty(len(connectivity[0]))
  for k in range(len(connectivity[0])):
    connectivity_indices[k] = np.where(connectivity[k][:])
  
  def loop_body(i):
    TranM_i       = np.zeros(1,K)
    X_int         = chart[i]["X_int"]
    nearest_store = np.zeros(1,N)

    for j in range(N):
      X = X_int;  
      nearest = i
      neigh = connectivity_indices[nearest]
      for k in range(step):
        X_proj, b,  _, H_hat, _, nearest, neigh = weighted_dd2(X, chart, neigh, nearest, connectivity_indices)
        X = np.transpose(X_proj) + b @ dt_s + H_hat @ rng.standard_normal((d,1)) * np.sqrt(dt_s)
        X = np.transpose(X)
      _,_,_,_,_, nearest,_ = weighted_dd2( X, chart, neigh, nearest,connectivity_indices )         
      nearest_store[j] = nearest
             
    a = np.unique[nearest_store]
    for index in a:
      TranM_i[index] = np.histogram(nearest_store,a[i])/N
      
    TranM[i][:] = TranM_i
  
  print("MSM stage is completed")
  return TranM