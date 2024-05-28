import numpy as np

def btrue(X,b_ito, b_stra):
  x = X[0]
  y = X[1]
  z = X[2]
  theta = np.arctan2( np.sqrt(x**2+y**2),z)
  phi = np.arctan2(y,x)
  b = b_ito(theta,phi) + b_stra(theta,phi)
  return b