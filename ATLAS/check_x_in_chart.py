import numpy as np

def check_x_in_chart(x, chart, t0, r, chi_p, threshold):
  """
  x - (1,D) row vector/point (2D ndarray)
  chart - ChartEntry object
  t0 - number, the relaxation time. 0.1/
  r - connectivity_threshold 0.3/
  chi_p - test statistic for 95% confidence interval. 5.991
  threshold - This is R_max determined by the diameter of the 
      slow manifold when setting the value of R or the
      maximum possible layer
  """
  # This function checks whether x is in the chart of time r*t0. Usually,
  # 0<r<4, 0<d(x,z_L)<2sqrt(t0)

  # chi_p is 95% statistics, in 2D, it is 5.991
  # if not isrow(x)
  #     x=x';  % make sure x is a row
  # end

  sigma       = chart.sigma
  X_int       = chart.X_int
  WU          = chart.WU

  Center      = X_int
  x_coeff     = (x - Center) @ WU
  total = sum([c**2 for c in x_coeff[0]])
  if  (total / sigma**2) <= (r**2 * t0 * chi_p) \
      and (np.linalg.norm(x-X_int,2)<threshold):   
    # this is  an very ad-hoc threshold. which is determined by the thickness of fast mode
    # this is necessary when building the
    # connectivity of charts. some charts that are far away in
    # Euclidean distance may be very close in
    # diffusion distance. 
    return True
  return False