from ATLAS.ATLAS.check_x_in_chart import check_x_in_chart
from ATLAS.ATLAS.distance_chart_to_x import distance_chart_to_x
from ATLAS.ATLAS.Learning_Slow_Manifold import Learning_Slow_Manifold
from ATLAS.Tests.Octave import python_to_matlab, python_array_to_cell
from ATLAS.Tests.is_close import is_close
from ATLAS.ATLAS.weighted_drift_diffusion2 import weighted_drift_diffusion2
from ATLAS.ATLAS.ChartEntry import ChartEntry
from ATLAS.ATLAS.random_start import random_start
from ATLAS.ATLAS.landmark import landmark
from ATLAS.DummyRNG import DummyRNG
from oct2py import octave
from jax import numpy as jnp
import numpy as np
from ATLAS import constants

def intToBool(val):
  if val==1: return True
  return False

class ATLASTests:
  """

  """

  def __init__(self):
    octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/ATLAS/")
    
  def get_args(self):
    x1 = jnp.array([[1,1,1.7]])
    x2 = jnp.array([[1,1,1.8]])
    # chart = {"sigma":1,"X_int":jnp.asarray([[1,1,2]]),\
    #         "WU":jnp.identity(3)}
    chart = ChartEntry()
    chart.sigma = 1
    chart.X_int = jnp.asarray([[1,1,2]])
    chart.WU = jnp.identity(3)
    t0 = 0.1
    r = 0.3
    chi_p = 5.991
    threshold = 1
    return x1,x2,chart,t0,r,chi_p,threshold

  def check_x_in_chart_test(self,verbose=False):
    x1,x2,chart,t0,r,chi_p,threshold = self.get_args()
    args_list1 = [x1,chart,t0,r,chi_p,threshold]
    args_str1 = "("+",".join([python_to_matlab(arg) for arg in args_list1])+")"
    args_list2 = [x2,chart,t0,r,chi_p,threshold]
    args_str2 = "("+",".join([python_to_matlab(arg) for arg in args_list2])+")"
    mat1 = intToBool(octave.eval(["check_x_in_chart"+args_str1+";"]))
    mat2 = intToBool(octave.eval(["check_x_in_chart"+args_str2+";"]))
    py1 = check_x_in_chart(*args_list1)
    py2 = check_x_in_chart(*args_list2)
    assert py1 == mat1, "check_x_in_chart"+args_str1+"="+str(mat1)+", not "+str(py1)
    assert py2 == mat2, "check_x_in_chart"+args_str2+"="+str(mat2)+", not "+str(py2)
  
  def distance_chart_to_x_test(self,verbose=False):
    x1,x2,chart,t0,_,chi_p,threshold = self.get_args()
    x3 = jnp.array([[999,999,999]])
    mode = 1
    args_list1 = [x1,chart,t0,chi_p,threshold,mode]
    args_str1 = "("+",".join([python_to_matlab(arg) for arg in args_list1])+")"
    args_list2 = [x2,chart,t0,chi_p,threshold,mode]
    args_str2 = "("+",".join([python_to_matlab(arg) for arg in args_list2])+")"
    mode3 = 2
    args_list3 = [x3,chart,t0,chi_p,threshold,mode3]
    args_str3 = "("+",".join([python_to_matlab(arg) for arg in args_list3])+")"
    mat1 = octave.eval(["distance_chart_to_x"+args_str1+";"])
    mat2 = octave.eval(["distance_chart_to_x"+args_str2+";"])
    mat3 = octave.eval(["distance_chart_to_x"+args_str3+";"])
    py1 = distance_chart_to_x(*args_list1)
    py2 = distance_chart_to_x(*args_list2)
    py3 = distance_chart_to_x(*args_list3)
    tested_function = "distance_chart_to_x"
    assert is_close(py1, mat1), tested_function+args_str1+"="+str(mat1)+", not "+str(py1)
    assert is_close(py2, mat2), tested_function+args_str2+"="+str(mat2)+", not "+str(py2)
    assert is_close(py3, mat3), tested_function+args_str2+"="+str(mat3)+", not "+str(py3)
  
  def convert_args_list_to_str(s,args_list):
    mat_str_ls = [python_to_matlab(arg) for arg in args_list]
    mat_str_ls[1] = python_array_to_cell(args_list[1])
    mat_str_ls[2] = python_array_to_cell(args_list[2])
    return "("+",".join(mat_str_ls)+")"
  
  def compare_struct_to_dict(s,struct,dic):
    skipped = ["getdoc"]
    for key,val in struct.items():
      if key in skipped:
        continue
      mat = struct[key]
      py = getattr(dic,key)
      if key == "sigma":
        # matlab treats [0] as 0 and breaks comparison
        mat = jnp.array(mat).reshape(-1)
      assert is_close(mat,py),"key="+str(key)\
          +","+str(mat)+";"+str(py)\
          +"type(mat)="+str(type(mat))+"; type(py)="\
          +str(type(py))#+str(getattr(dic,key).size)
      # print("passed "+key)

  def Learning_Slow_Manifold_test(self,verbose=False):
    data = {"dt":1,"UpperBound":2,"LowerBound":0}
    Cov_store = jnp.zeros((3,3,3)) # reshaped below
    Mean_store = jnp.zeros((3,3))  # reshaped below
    D = 3
    d = 1
    modify = 0
    args_list = [data,Cov_store,Mean_store,D,d,modify]
    args_str1 = self.convert_args_list_to_str(args_list)
    args_list[1] = Cov_store.reshape(1,3,3,3)
    args_list[2] = Mean_store.reshape(1,3,3)
    tested_func = "Learning_Slow_Manifold"
    # print(tested_func+args_str1)
    mat1 = octave.eval([tested_func+args_str1+";"])
    py1 = Learning_Slow_Manifold(*args_list)
    self.compare_struct_to_dict(mat1,py1)
  
  def weighted_dd2_test(self,verbose=False):
    X0 = jnp.array([[2,0,0]]) # 1 by D row
    D = 3
    d = 1

    c1 = ChartEntry()
    c1.X_int = jnp.array([[1,0,0]])
    c1.sigma = jnp.array([[2]])
    c1.WU = jnp.array([[1],[0],[0]])
    c1.U = jnp.array([[0],[0],[1]])
    c1.b = jnp.array([[0],[0],[2]])
    c1.Lambda = jnp.array([[1]])
    c2 = c1.clone()
    c2.X_int = jnp.array([[1,1,1]])
    c3 = c1.clone()
    c3.X_int = jnp.array([[0,2,1]])
    chart = [c1,c2,c3]
    neigh = [0,2]
    nearest = 0
    connectivity_indices = np.asarray([[0,1,2],[0,1,2],[0,1,2]]) # fully connected
    t0 = 0.1
    chi_p = 5.991
    D = 3
    d =1
    threshold = [1,0.001]
    option = 1
    mode = 1
    args = [X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D, d, threshold, option, mode]
    mat_args = [X0, chart, neigh, nearest, connectivity_indices, t0, chi_p, D, d, threshold, option, mode]
    mat_args[2] = [1,3] # matlab is 1-indexed
    mat_args[4] = np.asarray([[1,2,3],[1,2,3],[1,2,3]])
    mat_args[3] = 1
    args_str = "("+",".join([python_to_matlab(arg) for arg in mat_args])+");"
    pX_proj, pb, pLambda_hat, pH_hat, pT, pnearest, pneigh \
        = weighted_drift_diffusion2(*args)
    func_name = "weighted_drift_diffusion2"
    mX_proj, mb, mLambda_hat, mH_hat, mT, mnearest, mneigh\
        = octave.eval([func_name+args_str],nout=7)
    # assert is_close(mX_proj, pX_proj)
    
    pneigh = [index+1 for index in pneigh]
    pneigh = np.asarray(pneigh)
    # matlab is 1-indexed, python is 0-indexed
    pneigh = pneigh.reshape(1,-1) # matlab outputs a 2D array
    pnearest += 1 # indexing, as per usual
    assert is_close(mX_proj,pX_proj), "mX_proj :"+str(mX_proj)\
      +"; pX_proj:"+str(pX_proj)
    assert is_close(mneigh,pneigh),"mneigh: "+str(mneigh)+"; pneigh: "\
        +str(pneigh)
    assert is_close(mb,pb), "mb :"+str(mb)+"; pb :"+str(pb)
    assert is_close(mLambda_hat,pLambda_hat), "mLambda_hat: "+str(mLambda_hat)\
        +"; pLambda_hat: "+str(pLambda_hat)
    assert is_close(mH_hat,pH_hat), "mH_Hat: "+str(mH_hat)+"; pH_hat: "\
        +str(pH_hat)
    assert is_close(mT,pT), "mT: "+str(mT)+"; pT: "+str(pT)
    assert is_close(mnearest,pnearest), "mnearest: "+str(mnearest)\
    +"; pnearest: "+str(pnearest)
    # return pX_proj, pb, pLambda_hat, pH_hat, pT, pnearest, pneigh
  
  def random_start_test(self,verbose=False):
    # This test will differ from others in that it won't run matlab's code.
    # It mutates a dictionary and is relatively simple
    correct_index = 0
    chart = [ChartEntry() for i in range(9)]
    chart[correct_index].X_int = np.array([[0,0,1]])
    connected_components = [[0,1],[2,3,4,5,6,7,8]]
    chart_sim_parameter = {}
    rng = DummyRNG(0)
    random_start(chart,connected_components,chart_sim_parameter,rng)
    assert chart_sim_parameter["nearest"] == correct_index,\
        str(chart_sim_parameter["nearest"])
    assert is_close(chart[correct_index].X_int,chart_sim_parameter["X_int"])
  
  def landmark_test(self,verbose=False):
    # this test is even more lax than the previous
    # graph (the function) is not implemented in Octave, per octave.eval
    c = ChartEntry()
    c.X_int = np.asarray([[0,0,0]])
    c.sigma = np.asarray([[1]])
    c.WU = np.asarray([[1],[1],[1]])
    c1 = c.clone()
    c2 = c.clone()
    chart = [c,c1,c2]
    t0=0.1
    chi_p = 5.991
    threshold = 1
    connectivity_threshold = 0.3
    chart, connectivity, P, num_connected_components, connected_component_list = \
    landmark(chart,t0,chi_p,threshold,connectivity_threshold)
    assert len(chart) == 1, "len(chart): "+str(len(chart))
    connectivity_correct = np.asarray([[1]])
    assert is_close(connectivity,connectivity_correct)
    assert P.number_of_edges()==1, str(P.number_of_edges())+" edges"
    assert P.number_of_nodes()==1, str(P.number_of_nodes())+" nodes"
    assert 1 == num_connected_components
    assert connected_component_list == [{0}]

  
  def run_all(self,verbose=False):
    for attribute in dir(self):
      if attribute.find("_test")==len(attribute)-5:
        if verbose:
          print("running",attribute)
        test = getattr(self, attribute)
        test(verbose=verbose)
        print(attribute+"."*(constants.passed_string_length-len(attribute))\
          +"passed")
    print("Learning_ini_chart NOT tested")
  
  def Learning_ini_chart_Test(self):
    from ATLAS.Models.Peanut.PeanutParameters import PeanutParameters
    from ATLAS.ATLAS.Learning_ini_chart import Learning_ini_chart
    from ATLAS.Tests.ATLAS.ATLASTests import ATLASTests
    from ATLAS.Tests.Simulator.SimulatorTests import SimulatorTests
    from oct2py import octave

    octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/model/Peanut/")
    octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/ATLAS/")
    octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/simulator/")
    x = octave.eval(["set_parameter;",

                    "X_int=[1,1,1];","K_int=1;N=2;dt=0.01;T_one=2;"\
                    ,"Learning_ini_chart;"])

    octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/ATLAS/")
    octave.eval(["Learning_ini_chart"])
    import numpy as np
    from ATLAS.DummyRNG import DummyRNG

    params = PeanutParameters()

    params.X_int = np.asarray([[1,1,1]])
    params.K_int = 1
    params.N = 2
    params.dt = 0.01
    params.T_one = 2
    y = Learning_ini_chart(params,warmup=0,rng=DummyRNG(1))


