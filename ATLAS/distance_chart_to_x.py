import numpy as np

def distance_chart_to_x(x, chart, t0, chi_p,threshold,mode):
# This function calculates the Mahalanobis-like distance 
# distance here defines the sqrt of the first time that the chart hits x,
# provided the time is less than 4*t0.  Here the time is in the unit of t0. 


#d = ~ d(x, x_L)/sqrt(t_0),
# sqrt(t0) is sqrt(D*t0) where D=1 m^2/s (L^2/T), that is why sqrt(t0) has unit of length

# This distance doesn't include the effect of the drift term. With the
# diffusion dorminate case, the drift term will be not siginificant. 

# if ~isrow(x)
#     x=x';  % make sure x is a row
# end

  sigma       = chart.sigma
  X_int       = chart.X_int
  WU          = chart.WU

  if  np.linalg.norm(x-X_int,2)>threshold  and mode == 1: # In exploration mode, it is necessary to notify it is too far away.
    # In simuation mode, with the projection, it is okay. 
    # threshold is determined by the thickness of fast mode, 
    d = 10 # x is too far away from the invariant manifold
  else:
    x_coeff     = (x - X_int) @ WU
    total = sum([c**2 for c in x_coeff[0]]) 
    t           = (total / sigma**2)/(chi_p)
    d           = min(np.sqrt(t/t0),10)
  return d