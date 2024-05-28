import numpy as np
seed = 1
default_rng = np.random.default_rng(seed)
import jax
import jax.numpy as jnp

def simEuler(RHS_parameter, Sim_parameter,parallel=False,verbose=False,rng=-1):
  if rng==-1:
    rng = default_rng
  diffusion = RHS_parameter["diffusion"]
  drift = RHS_parameter["drift"]
  D = RHS_parameter["D"]
  UpperBound = RHS_parameter["UpperBound"]
  LowerBound = RHS_parameter["LowerBound"]
  X_int = Sim_parameter["X_int"]
  T_max = Sim_parameter["T_max"]
  dt = Sim_parameter["dt"]
  N = Sim_parameter["N"]
  
  t_span = jnp.arange(0,T_max+dt,dt)
  tN = len(t_span)-1
  
  X = np.zeros((N,D,tN+1))
  sqrtdt = dt**0.5
  r_store = rng.standard_normal(size=(D,tN*N))
  def update(j):
    # X_j = np.zeros(shape=(D,tN+1))
    X_j = []
    # X_j[:,0] = X_int
    X_j.append(X_int.reshape(-1))
    for i in range(0,tN):
      Current = X_j[i].reshape(-1,1)
      cur_r_store = rng.standard_normal(D)
      cur_r_store = [1,1,1] # get rid of
      drift_current = drift(Current)*dt
      diffusion_ = diffusion(Current)
      diffusion_current = diffusion(Current) @ cur_r_store*(dt**0.5)
      diffusion_current = diffusion_current.reshape(-1,1)
      Next = Current + drift_current + diffusion_current
      
      X_j.append(Next.reshape(-1))
    # print(j,X_j)
    return jnp.transpose(jnp.array(X_j))
  
  # def update2(j,par=True):
  #   # X_j = np.zeros(shape=(D,tN+1))
  #   X_j = []
  #   # X_j[:,0] = X_int
  #   X_j.append(X_int.reshape(-1))
  #   for i in range(0,tN):
  #     # Current = X_j[:,i].reshape(-1,1) # reshape into column
  #     Current = X_j[i].reshape(-1,1)
  #     cur_r_store = r_store[:,(j-1)*tN+i]
  #     drift_current = drift(Current)*dt
  #     diffusion_current = diffusion(Current) @ cur_r_store*sqrtdt
  #     diffusion_current = diffusion_current.reshape(-1,1)
  #     Next = Current + drift_current + diffusion_current
  #     # X_j[:,i+1] = Next.reshape(-1)
  #     X_j.append(Next.reshape(-1))
  #   # print(X_j)
  #   return np.transpose(np.asarray(X_j))
  
  if not parallel:
    for j in range(N):
      temp = update(j)
      # print("temp",temp)
      X[j] = temp
  else:
    X = jnp.array(jax.vmap(update)(jnp.arange(j)))
  
  Tr_store = np.zeros((1,tN+1))
  Cov_store = np.empty((1,tN+1),dtype=object)
  Mean_store = np.empty((1,tN+1),dtype=object)
  
  for i in range(0,tN+1):
    Store = X[:,:,i]
    # print("Store",Store,Store.shape,type(Store))
    Cov_store[0][i] = np.cov(np.transpose(Store))
    # if i==0:
    #   print("Cov_store[0][1]",Cov_store[0][0])
    Mean_store[0][i] = np.mean(Store,axis=0).reshape(1,-1)
    Tr_store[0][i] = np.trace(Cov_store[0][i])
  
  data = {}
  data["X_int"] = X_int
  data["tN"] = tN
  data["dt"] = dt
  data["T_max"] = T_max
  data["Tr_store"] = Tr_store
  data["LowerBound"] = LowerBound
  data["UpperBound"] = UpperBound
  ret_Cov_store = np.zeros((1,tN+1,D,D))
  ret_Mean_store = np.zeros((1,tN+1,1,D))
  for i in range(tN+1):
    ret_Cov_store[0][i] = Cov_store[0][i]
    for j in range(D):
      ret_Mean_store[0][i][0][j] = Mean_store[0][i][0][j]
  # we want arrays with the right dimensions
  return data, ret_Cov_store, ret_Mean_store #, Store