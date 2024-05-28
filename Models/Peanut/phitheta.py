import numpy as np
pi = 3.14159

def phitheta(X_curr, phi_zero, theta_zero):
  """
  X_curr - 
  phi_zero - assumed to be a vector
  theta_zero - assumed to be a number
  """
  theta_curr = np.arctan2( np.sqrt( X_curr[0]**2 + X_curr[1]**2 ), X_curr[2])
  phi_curr = np.arctan2(X_curr(2), X_curr(1)) % (2*pi)
  
  i=np.argmin(np.abs(phi_curr-phi_zero))
  if theta_curr>theta_zero[i]:
      index = -1
  else:
      index = 1
  return index