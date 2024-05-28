from ATLAS.ATLAS.ATLAS_simulator2 import ATLAS_simulator2
from ATLAS.ATLAS.landmark import landmark


def peanut_simulation_learning(weighted_dd2, chart_sim_parameter, RHS_parameter, simulator_par):
  print('Use ATLAS simulator to simulate a single traj.')
  chart_sim_parameter["Nstep"] = 2*10**7
  chart_sim_parameter["gap"] = 1000
  _, _, chart = ATLAS_simulator2(weighted_dd2, chart_sim_parameter, RHS_parameter, simulator_par, chart)
  t0 = chart_sim_parameter["t0"]
  chi_p = chart_sim_parameter["chi_p"]
  threshold = chart_sim_parameter["threshold"]
  connectivity_threshold = chart_sim_parameter["connectivity_threshold"]
  [chart, connectivity, P] = landmark(chart, t0, chi_p,threshold[0], connectivity_threshold)
  print("save(chart_fileName,...)")
  #save( chart_fileName ,'chart','connectivity','P')