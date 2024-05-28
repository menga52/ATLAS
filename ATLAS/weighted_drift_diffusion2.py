import numpy as np

"""
X0 - 1 by D row
chart - list of ChartEntry objects
neigh - list of integers (indices for chart list)
nearest - ChartEntry
connectivity_indices - list of lists of indices - if [i][j]==1, i is connected to j
t0 - number, the relaxation time. 0.1/
chi_p - test statistic for 95% confidence interval. 5.991
D - # dimensions in unreduced model
d - # dimensions in reduced model
threshold - This is R_max determined by the diameter of the 
      slow manifold when setting the value of R or the
      maximum possible layer
option - 1=fast mode, 2=orthogonal projection
mode - whether to cap distance at 10 (?)

returns:
X_proj - 1 by D row
b - D by 1 column
Lambda_hat - D by D matrix
H_hat - D by 1 column
T - 1 by L_n row  (L_n is number of neighbors)
nearest - index of nearest neighbor in list
neigh - list of indices of neighbors
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
        X_coeff = (X0-Center) @ U # row vec 1 by d
      t = sum([x_coeff[0][i]**2/sigma[0][i]**2 for i in \
          range(min(len(x_coeff[0]),len(sigma[0])))])/chi_p
      # t = sum([x_coeff[i]**2/sigma[i]**2/chi_p for i in range(min(len(x_coeff), len(sigma)))]) # min should be unnecessary
      T[k] = min(np.sqrt(t/t0), 10)
    index_of_min_T = 0
    min_value = np.infty
    for i in range(len(T)):
      if T[i]<min_value:
        min_value = T[i]
        index_of_min_T = i
    if neigh[index_of_min_T] == nearest:
      index = 0
    else:
      nearest = neigh[index_of_min_T]
      neigh = connectivity_indices[nearest]
  
  L_n = len(neigh)
  T = np.zeros((1,L_n))
  Lambda = np.zeros((D,D))
  b = np.zeros((D,1))
  X = np.zeros((L_n,D)) # changing this to jnp.array would be difficult
  weight = 0
  X_proj = X0

  # Project X_curr on the estimated manifold and estimate the effective drift and diffusion
  for k in range(L_n):
    cur_chart = chart[neigh[k]]
    Center = cur_chart.X_int
    if mode == 1 and np.linalg.norm(X0-Center,2) > threshold[0]:
      T[0][k] = 10
      X[k][:] = np.zeros(D)
    else:
      U = cur_chart.U
      Ut = np.transpose(U)
      sigma = cur_chart.sigma
      if option == 1:
        WU = cur_chart.WU
        x_coeff = (X0-Center) @ WU
      if option == 2:
        x_coeff = (X0-Center) @ U
      t = sum([x_coeff[0][i]**2/sigma[0][i]**2 for i in \
          range(min(len(x_coeff[0]),len(sigma[0])))])/chi_p
      # t = sum([x_coeff[i]**2/sigma[i]**2/chi_p for i in range(len(x_coeff))]) # min should be unnecessary
      T[0][k] = min (np.sqrt(t/t0), 10)
      expmTk = np.exp(-T[0][k])
      X[k,:] = (x_coeff @ Ut + Center) * expmTk

      weight += expmTk
      b += cur_chart.b * expmTk
      Lambda += cur_chart.Lambda * expmTk
  def div_by_weight(x):
    return x/weight
      
  if weight != 0:
    X_proj = np.vectorize(div_by_weight)(sum(X))
    X_proj = X_proj.reshape(1,-1) # should remain 2-dim array
    b /= weight
    Lambda /= weight
  
  UL, SL, _ = np.linalg.svd(Lambda)
  UL = UL[:,:d]
  SL = SL[:d]
  SL = np.diag(SL)
  Lambda_hat = UL @ SL @ np.transpose(UL)
  H_hat = UL @ np.sqrt(SL)

  # Likely useful when D will be much larger; the LambdaUd can be computed once for all in each chart, and interpolated
  # [U2,S2,~]               = svds(Lambda,d,'largest','SubspaceDimension',d+2,'Tolerance',1e-3,'LeftStartVector',?LambdaUd?,'MaxIterations',10);%,'Display',true);
  # Lambda_hat2             = U2(:,1:d) * S2(1:d, 1:d) * U2(:, 1:d)';
  # H_hat2                  = U2(:,1:d) * sqrt( S2(1:d, 1:d) );
  # norm(Lambda_hat-Lambda_hat2)/norm(Lambda_hat)
  # min(norm(H_hat-H_hat2),norm(H_hat+H_hat2))

  return X_proj, b, Lambda_hat, H_hat, T, nearest, neigh
    

    