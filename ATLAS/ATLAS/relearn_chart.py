"""
Relearn the chart with different diffusion coefficient or to further relax
The diffusion coefficient here is smaller and exploration mode is not
needed since the invariant manifold is the same.

Another situation is when T_max is not enough to relax the IC onto the
invariant manifold because it might be too far away or different
components have different relaxation scale
"""

import numpy as np

from ATLAS.ATLAS.Learning_Slow_Manifold import Learning_Slow_Manifold
from ATLAS.ATLAS.landmark import landmark
from ATLAS.ATLAS.check_x_in_chart import check_x_in_chart

# set up new parameter if needed
# note t0, lowerbound and upperbound of training window may be reset

def relearn_chart(chart, relearn_parameter, index_to_learn, verbose=False):
  N_relearn = relearn_parameter["N_relearn"]
  RHS_parameter = relearn_parameter["RHS_parameter"]
  simulator_par = relearn_parameter["simulator_par"]
  iter_max = relearn_parameter["iter"]
  relative_threshold  = relearn_parameter["relative_threshold"]
  modify = RHS_parameter["modify"]
  T_max = RHS_parameter["T_max"]
  dt = RHS_parameter["dt"]
  D = RHS_parameter["D"]
  d = RHS_parameter["d"]
  t0 = RHS_parameter["t0"]
  chi_p = RHS_parameter["chi_p"]
  threshold = RHS_parameter["threshold"]
  connectivity_threshold = RHS_parameter["connectivity_threshold"]
  K = len(chart)

  indicator = np.ones(K)
  i_to_learn = []
  for i in range(K):
    if i in index_to_learn:
      if verbose:
        print("Landmark No.",i)
      relative_error = np.ones(2)
      j = 1 # number of iterations
      while (relative_error[0]>relative_threshold[0] or relative_error[1]>relative_threshold[1]) \
          and indicator[i] and j<=iter_max:
        # if both relative error are less than 2%, or landmark is close
        # to other preious landmarks or reaches maximum, jump out of loop
        if verbose:
          print("iter No.",j)
        Sim_parameter = {}
        Sim_parameter["T_max"] = T_max
        Sim_parameter["dt"] = dt
        Sim_parameter["N"] = N_relearn
        Sim_parameter["X_int"] = chart[i].X_int
        data, Cov_store, Mean_store = simulator_par(Sim_parameter)
        chart_new = Learning_Slow_Manifold(data, Cov_store, Mean_store, D, d, modify)
        relative_error[0] = np.linalg.norm(chart[i].sigma - chart_new.sigma)/np.linalg.norm(chart_new.sigma)
        relative_error[1] = np.linalg.norm(chart[i].b-chart_new.b)/np.linalg.norm(chart_new.b)
        if verbose:
          print("relative error is",relative_error)
        chart[i] = chart_new
        for k in range(i-1):
          if indicator[k] and check_x_in_chart(chart[i].X_int, chart[k], t0, 0.3, chi_p, threshold[0]) \
              and check_x_in_chart(chart[k].X_int, chart[i], t0, 0.3, chi_p, threshold[0]):
            indicator[i] = 0 # this chart has been removed
            if verbose:
              print("landmark", i, "is removed")
              break
        j += 1
      if j>iter_max and indicator[i]:
        i_to_learn.append(i) # these landmarks are not accurate
  chart = [chart[i] for i in range(K) if indicator[i]]
  chart, _, _ = landmark(chart, t0, chi_p, threshold[0], connectivity_threshold)
  return chart, i_to_learn
