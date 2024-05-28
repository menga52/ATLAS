import numpy as np

def get_drift(X,epsilon,a1,a2,c1,c2,c3,c4,c5,c6,verbose=False):
  """
  X - a column vector depicting a point
  epsilon - 
  a1 - 
  a2 - 
  c1 - 
  c2 - 
  c3 - 
  c4 - 
  c5 - 
  c6 - 
  """
  if verbose:
    print("computing get_drift")
  x = X[0][0]
  y = X[1][0]
  z = X[2][0]

  r       = np.sqrt(x**2+y**2+z**2)
  r_sq    = r**2
  rsin    = np.sqrt(x**2 + y**2)
  rsin_sq = rsin**2  #x**2+y**2

  J = [[ x/r, x*z/rsin, -y],\
        [y/r, y*z/rsin,  x],\
        [z/r, -rsin,       0 ]]
              
  b = [[-c1/epsilon*(1-np.sqrt(a1+a2*z**2/r_sq)/r)],\
        [c3 * ( 4*z**3/r**3-3*z/r )/rsin], \
        [c5 * ( y*z/(rsin* r_sq) + x/r_sq)]]
  J = np.asarray(J)
  b = np.asarray(b)
                  
  #Ito         = 1/2* ([ -x; -y; -z]* c4^2 * rsin_sq./(r_sq.^2) + [-x; -y; 0] * c6^2./r_sq);
  d1          = c4**2 * rsin_sq/(r_sq**2)
  d2          = c6**2/r_sq
  sumd1d2     = d1+d2
  Ito         = 1/2 * np.asarray([[-x * sumd1d2],[ -y* sumd1d2],[ -z *d1  ]])

  drift       = J @ b + Ito
  return drift