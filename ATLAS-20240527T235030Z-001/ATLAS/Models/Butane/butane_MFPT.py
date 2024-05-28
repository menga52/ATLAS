import numpy as np
from ATLAS.Models.Butane.MFPT_butane2_ori import MFPT_butane2_ori

seed = 1
default_rng = np.random.default_rng(seed)

"""
RHS_parameter - a dictionary. See ATLAS.Models.Parameters
N_IC - an integer, counting something.
output - unknown, likely a 2D array. Likely computed by butane_ori_simulation
chart_sim_parameter - dictionary, see ATLAS.Models.Parameters
t0 - unknown
rng - random number generator object from numpy
verbose - True=print things. for debugging
"""
def butane_MFPT(RHS_parameter, N_IC, output, chart_sim_parameter, t0,datapath=-1,rng=-1,verbose=False):
  Mean_FPT = np.zeros((6,10))
  relative_error_FPT = np.zeros((3,10))
  Q1 = np.zeros((3,3,10))
  Q2 = np.zeros((3,3,10))
  t_final_ori = np.zeros(3)
  dt_s = t0

  if verbose:
    print("Starting MFPT part")
  pi = np.pi
  well_threshold1 = [-pi/3, pi/3]
  well_threshold2 = [pi/3, pi]
  well_threshold3 = [-pi, pi/3]
  if verbose:
    print("original simulator")
  FPT_ori1, exit_curr1_ori, t_final_ori[0] = MFPT_butane2_ori(well_threshold1, output, N_IC, RHS_parameter, chart_sim_parameter)
  FPT_ori2, exit_curr2_ori, t_final_ori[1] = MFPT_butane2_ori(well_threshold2, output, N_IC, RHS_parameter, chart_sim_parameter)
  FPT_ori3, exit_curr3_ori, t_final_ori[2] = MFPT_butane2_ori(well_threshold3, output, N_IC, RHS_parameter, chart_sim_parameter)

  for k in range(10):
    if verbose:
      print("round #",k)
    chart_fileName = datapath+"chart"+str(k)+".mat"



