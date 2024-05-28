import numpy as np

def H_trueX(X,H_true):
  x  = X[0]
  y  = X[1]
  z  = X[2]
  theta = np.arctan2( np.sqrt(x**2+y**2),z)
  phi = np.arctan2(y, x)
  
  H_true_X = H_true(theta, phi)
  return H_true_X