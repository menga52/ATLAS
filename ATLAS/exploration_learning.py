from ATLAS.ATLAS.weighted_drift_diffusion2 import weighted_drift_diffusion2
from ATLAS.ATLAS.random_start import random_start
from ATLAS.ATLAS.ATLAS_simulator2 import ATLAS_simulator2
from ATLAS.ATLAS.landmark import landmark
from ATLAS.ATLAS.relearn_chart import relearn_chart

def exploration_learning(model, chart, bins, verbose=False):
  print("Starting exploration learning stage")
  mode = 1 # set to exploration mode

  t0 = model.parameters.t0
  chi_p = model.parameters.chi_p
  D = model.parameters.D
  d = model.parameters.d
  threshold = model.parameters.threshold
  option = model.parameters.option
  def weighted_dd2(X0, chart, neigh, nearest, connectivity_indices):
    return weighted_drift_diffusion2(X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D, d, threshold, option, mode)
  # use the exploration mode to further learn
  explore_round = 1
  explore_round_max = 100
  K = len(chart)

  random_start(chart, bins, model.parameters.chart_sim_parameter)
  bin_N = 0 # seems bin_N was undefined at this point in matlab code
  chart_sim_parameter = model.parameters.chart_sim_parameter
  RHS_parameter = model.parameters.RHS_parameter
  simulator_par = model.parameters.simulator_par
  relearn_parameter = model.parameters.relearn_parameters
  connectivity_threshold = model.parameters.connectivity_threshold
  while bin_N != 1 and explore_round < explore_round_max:
    print("Round",explore_round,"of exploration")
    _,_,chart = ATLAS_simulator2(weighted_dd2, chart_sim_parameter, RHS_parameter, simulator_par, chart)
    chart, _, _, _, _ = landmark(chart, t0, chi_p, threshold[0], connectivity_threshold)
    if verbose:
      print("No. of landmarks after exploration is ", len(chart))
    if model.parameters.relearn_option == 1:
      K_next = len(chart)
      index_to_learn = range(K,K_next)
      chart = relearn_chart(chart, relearn_parameter, index_to_learn)
      chart, connectivity, P, bin_N, bins = landmark(chart, t0, chi_p, threshold[0], connectivity_threshold)
    
    K = len(chart)
    random_start(chart, bins, model.parameters.chart_sim_parameter)
    explore_round += 1
  
  # save( chart_fileName ,'chart','connectivity','P')    
  print('exploration learning stage is completed')
        





