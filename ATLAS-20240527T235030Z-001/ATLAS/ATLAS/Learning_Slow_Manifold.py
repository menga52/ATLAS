import numpy as np
from jax import numpy as jnp
from scipy.linalg import subspace_angles
from ATLAS.ATLAS.ChartEntry import ChartEntry

def Learning_Slow_Manifold(data, Cov_store, Mean_store, D, d, modify):
  # alias parameters
  dt = data["dt"]
  UpperBound = data["UpperBound"]
  LowerBound = data["LowerBound"]

  s_u = int(np.round(UpperBound/dt))
  s_l = int(np.round(LowerBound/dt))
  s_span = range(s_l,s_u)
  t_span = jnp.array(jnp.arange(s_l*dt,s_u*dt,dt)).reshape(1,-1) # row
  t_span = t_span - jnp.full(t_span.shape,jnp.mean(t_span))

  Cov_span = jnp.array([Cov_store[0][k] for k in s_span])
  Mean_span = jnp.array([Mean_store[0][k] for k in s_span])
  sum_Cov = jnp.zeros((D,D))
  sum_Mean = jnp.zeros((1,D))

  M = len(s_span)

  # for i in range(M):
  #   sum_Cov = sum_Cov + Cov_span[i]
  #   sum_Mean = sum_Mean + Mean_span[i]
  sum_Cov = sum(Cov_span)
  sum_Mean = sum(Mean_span)

  
  temp_Cov = jnp.zeros((D,D))
  temp_Mean = jnp.zeros((1,D)) # row

  def div_by_M(x):
    return x/M
  
  sum_Cov_normalized = jnp.vectorize(div_by_M)(sum_Cov)
  sum_Mean_normalized = jnp.vectorize(div_by_M)(sum_Mean)
  def sub_sum_Cov_normalized(x):
    return x - sum_Cov_normalized
  Cov_span = jnp.vectorize(sub_sum_Cov_normalized,signature="(m,n)->(m,n)")(Cov_span)
  def sub_sum_Mean_normalized(x):
    return x - sum_Mean_normalized
  Mean_span = jnp.vectorize(sub_sum_Mean_normalized,signature="(m,n)->(m,n)")(Mean_span)

  for i in range(M):
    # Cov_span[i] = Cov_span[i] - sum_Cov_normalized
    temp_Cov = temp_Cov + Cov_span[i] * t_span[0][i]

    # Mean_span[i] = Mean_span[i] - sum_Mean_normalized
    temp_Mean = temp_Mean + Mean_span[i] * t_span[0][i]
  
  #b_hat = \sum(m_bar(t)*t_bar)/|t_bar|^2;
  #Lambda_hat =  \sum(Lambda_bar(t)*t_bar)/|t_bar|^2;
  y = np.linalg.norm(t_span,2)**2
  def div_by_normsq(x):
    return x/y
  Lambda_full = jnp.vectorize(div_by_normsq)(temp_Cov)
  b_full = jnp.vectorize(div_by_normsq)(temp_Mean)

  if modify == 0:
    # Use intercept as Landmark of the chart
    x0 = sum_Mean_normalized - b_full*((UpperBound+LowerBound)/2)
  else:
    # Use tau_min time point as Landmark of the chart
    x0 = sum_Mean_normalized - b_full*((UpperBound - LowerBound)/2)
  Gamma_full = sum_Cov/M - Lambda_full*(UpperBound+LowerBound)/2
  # Projection to d dimension

  U,S,_ = np.linalg.svd(Lambda_full)
  S = jnp.diag(S)
  Lambda_hat = U[:,:d]@ S[:d,:d] @ jnp.transpose(U[:,:d])
  sigma = jnp.sqrt(jnp.diag(S[:d][:d]))
  sigma = np.transpose(sigma) #make sure sigma is a row vector
  
  U = U[:,:d]
  
  V,VS,_ = jnp.linalg.svd(Gamma_full)
  diag_VS = VS.reshape(-1,1) #column vector
  VS_len = len(VS)
  index = jnp.where(VS>VS[0]/3)[0] # where returns a tuple
  V_index = []
  # for i in range(len(index)):
    # very likely bug
    # could likely be replaced with [a in X if a>0.2]
    # j = (VS_len+1)*index[i]+1
    # if subspace_angles(U, V[:][j])[0]/np.pi > 0.2: # this needs to be changed (original comment)
    #   V_index = np.hstack()
  # V_index = [V[:][(VS_len+1)*index[i]] for i in range(len(index)) if subspace_angles(U, V[:][(VS_len+1)*index[i]])[0]/np.pi > 0.2]
  for i in range(index.shape[0]): # this needs to be changed (orig)
    # print(V[:,index[i]].shape)
    # print("U.shape",U.shape)
    if subspace_angles(U, V[:,index[i]].reshape(D,-1))/jnp.pi > 0.2:
      # V_index = jnp.append(V_index, V[:,index[i]])
      V_index.append(V[:,index[i]])
  V_index = jnp.array(V_index).reshape(U.shape[0],-1)
    
  # another likely bug
  E = jnp.append(U,V_index,axis=1)
  Q,R = np.linalg.qr(E) # Q and R not unique
  WU = Q@ np.linalg.pinv(R @ np.transpose(R))@ np.transpose(Q) @ U

  #store the data into a chart
  chart = ChartEntry()
  chart.LearningTime = np.NaN    # number
  chart.U = U     # column, D by 1
  chart.V = V_index              # ????
  chart.sigma = sigma            # 1-dim matr, length d
  chart.sigma_fast = diag_VS     # column, D by 1
  chart.Lambda = Lambda_hat      # matrix, D by D
  chart.b = np.transpose(b_full) # column, D by 1
  chart.X_int = x0               # row vector, 1 by D
  chart.WU = WU   # column, D by 1

  return chart
