import numpy as np

"""
X0 - 
"""
def weighted_drift_diffusion2(X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D, d, threshold, option, mode):
  index = 1
  while index:
    L_n = len(neigh)
    T = np.zeros(L_n)
    for k in range(L_n):
      cur_chart = chart[neigh[k]]
      Center = cur_chart.X_int
      sigma = cur_chart.sigma
      if option == 1:
        WU = cur_chart.WU # this is fast mode projection
        x_coeff = (X0-Center) @ WU
      elif option == 2:
        U = cur_chart.U # this is orthogonal projection
        X_coeff = (X0-Center) @ U
      t = sum([x_coeff[i]**2/sigma[i]**2/chi_p for i in range(min(len(x_coeff), len(sigma)))]) # min should be unnecessary
      T[k] = min(np.sqrt(t/t0), 10)
    index_of_min_T = 0
    min_value = np.infty
    for i in range(len(T)):
      if T[i]<min_value:
        min_value = T[i]
        index_of_min_T = i
    if neigh[i] == nearest:
      index = 0
    else:
      nearest = neigh[i]
      neigh = connectivity_indices[nearest]
  
  L_n = len(neigh)
  T = np.zeros(L_n)
  Lambda = np.zeros((D,D))
  b = np.zeros(D)
  X = np.zeros((L_n,D))
  weight = 0
  X_proj = X0

  # Project X_curr on the estimated manifold and estimate the effective drift and diffusion
  for k in range(L_n):
    cur_chart = chart[neigh[k]]
    Center = cur_chart.X_int
    if mode == 1 and np.linalg.norm(X0-Center) > threshold[0]:
      T[k] = 10
      X[k][:] = np.zeros(D)
    else:
      U = cur_chart.U
      Ut = np.transpose(U)
      sigma = cur_chart.sigma
      
      if option == 1:
        WU = cur_chart.WU
        x_coeff = (X0-Center) @ WU
      elif option == 2:
        x_coef = (X0-Center) @ U
      
      t = sum([x_coeff[i]**2/sigma[i]**2/chi_p for i in range(len(x_coeff))]) # min should be unnecessary
      T[k] = min (np.sqrt(t/t0), 10)
      expmTk = np.exp(-T[k])
      X[k][:] = (x_coeff @ Ut + Center) @ expmTk

      weight += expmTk
      b += cur_chart.b * expmTk
      Lambda += cur_chart.Lambda * expmTk
      
  if weight != 0:
    X_proj = sum(X,1)
    b /= weight
    Lambda /= weight
  
  UL, SL, _ = np.linalg.svd(Lambda)
  UL = UL[:][:d]
  SL = SL[:d][:d]
  Lambda_hat = UL @ SL @ np.transpose(UL)
  H_hat = UL @ np.sqrt(SL)

  # Likely useful when D will be much larger; the LambdaUd can be computed once for all in each chart, and interpolated
  # [U2,S2,~]               = svds(Lambda,d,'largest','SubspaceDimension',d+2,'Tolerance',1e-3,'LeftStartVector',?LambdaUd?,'MaxIterations',10);%,'Display',true);
  # Lambda_hat2             = U2(:,1:d) * S2(1:d, 1:d) * U2(:, 1:d)';
  # H_hat2                  = U2(:,1:d) * sqrt( S2(1:d, 1:d) );
  # norm(Lambda_hat-Lambda_hat2)/norm(Lambda_hat)
  # min(norm(H_hat-H_hat2),norm(H_hat+H_hat2))

  return X_proj, b, Lambda_hat, H_hat, T, nearest, neigh
    