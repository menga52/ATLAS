import numpy as np
from scipy.linalg import subspace_angles
from ATLAS.ATLAS.ChartEntry import ChartEntry

def Learning_Slow_Manifold(data, Cov_store, Mean_store, D, d, modify):
  # alias parameters
  dt = data.dt
  UpperBound = data.UpperBound
  LowerBound = data.LowerBound

  s_u = np.round(UpperBound/dt)
  s_l = np.round(LowerBound/dt)

  s_span = range(s_l,s_u+1)
  t_span = range(s_l*dt,(s_u+1)*dt,dt)
  t_span = t_span - np.mean(t_span)

  Cov_span = [Cov_store[k] for k in s_span]
  Mean_span = [Mean_store[k] for k in s_span]
  sum_Cov = np.zeros((D,D))
  sum_Mean = np.zeros((D,D))

  M = len(s_span)

  for i in range(M):
    sum_Cov = sum_Cov + Cov_span[i]
    sum_Mean = sum_Mean + Mean_span[i]
  
  temp_Cov = np.zeros((D,D))
  temp_Mean = np.zeros(D)

  for i in range(M):
    Cov_span[i] = Cov_span[i] - sum_Cov/M
    temp_Cov = temp_Cov + Cov_span[i] @ t_span[i]

    Mean_span[i] = Mean_span[i] - sum_Mean/M
    temp_Mean = temp_Mean + Mean_span[i] @ t_span[i]
  
  #b_hat = \sum(m_bar(t)*t_bar)/|t_bar|^2;
  #Lambda_hat =  \sum(Lambda_bar(t)*t_bar)/|t_bar|^2;
  
  Lambda_full = temp_Cov/np.linalg.norm(t_span)**2
  b_full = temp_Mean/np.linalg.norm(t_span)**2

  if modify == 0:
    # Use intercept as Landmark of the chart
    x0 = sum_Mean/M - b_full*(UpperBound+LowerBound)/2
  else:
    # Use tau_min time point as Landmark of the chart
    x0 = sum_Mean/M - b_full*(UpperBound - LowerBound)/2
  Gamma_full = sum_Cov/M - Lambda_full*(UpperBound+LowerBound)/2
  # Projection to d dimension

  U,S,_ = np.linalg.svd(Lambda_full)
  Lambda_hat = U[:,:d]@ S[:d,:d] @ U[:,:d]
  sigma = np.sqrt(np.diag(S[:d][:d]))
  sigma = np.tranpose(sigma) #make sure sigma is a row vector
  U = U[:][:d]  
  
  V,VS,_ = np.linalg.svd(Gamma_full)
  diag_VS = np.diag(VS)
  VS_len = len(VS)
  index = np.where(VS>VS[0]/3)
  V_index = []
  #for i in range(len(index)):
    # very likely bug
    # could likely be replaced with [a in X if a>0.2]
    #j = (VS_len+1)*index[i]+1
    #if np.subspace_angles(U, V[:][j])/np.pi > 0.2: # this needs to be changed (original comment)
      # V_index = 
  V_index = [V[:][(VS_len+1)*index[i]] for i in range(len(index)) if subspace_angles(U, V[:][j])/np.pi > 0.2]
  # another likely bug
  E = np.append(U,V_index,axis=1)
  Q,R = np.linalg.qr(E) # Q and R not unique
  WU = Q@ np.linalg.pinv(R @ np.transpose(R))@ np.transpose(Q) @ U

  #store the data into a chart
  chart = ChartEntry()
  chart.LearningTime = np.NaN
  chart.U = U
  chart.V = V_index
  chart.sigma = sigma
  chart.sigma_fast = diag_VS
  chart.Lambda = Lambda_hat
  chart.b = np.transpose(b_full)
  chart.X_int = x0
  chart.WU = WU

  return chart
