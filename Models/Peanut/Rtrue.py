import numpy as np

def Rtrue(X,R):
  x = X[0]
  y = X[1]
  z = X[2]
  theta = np.arctan2( np.sqrt(x**2+y**2),z)
  phi = np.arctan2(y,x)
  Rx = R(theta, phi)
  return Rx