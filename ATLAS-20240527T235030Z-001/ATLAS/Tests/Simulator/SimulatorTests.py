import numpy as np
import jax.numpy as jnp
from ATLAS.Models.Peanut.PeanutParameters import PeanutParameters
# from ATLAS.Models.Butane.ButaneParameters import ButaneParameters
# from ATLAS.Models.Halfmoon.HalfmoonParameters import HalfmoonParameters
from ATLAS.Simulator.simEuler import simEuler
from ATLAS.Simulator.simEuler_one_traj import simEuler_one_traj
from ATLAS.Simulator.simEuler_par import simEuler_par
from ATLAS.Simulator.simEulernonlintrans import simEulernonlintrans
from ATLAS.Simulator.simEulernonlintrans_one_traj import simEulernonlintrans_one_traj
from ATLAS.Simulator.simEulernonlintrans_par import simEulernonlintrans_par
from ATLAS.Tests.is_close import is_close
from ATLAS.Tests.Octave import dict_to_struct,struct_to_dict
from ATLAS.DummyRNG import DummyRNG
from oct2py import octave
from ATLAS import constants
import time
octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/simulator/")



verbose = True

class SimulatorTests:
  def __init__(self):
    self.fake_rng = DummyRNG()
    self.lin_paramss = [] # ss because list of lists. paramses
    self.nonlin_paramss = []
    self.lin_Sim_parameterss = []
    self.nonlin_Sim_parameterss = []
    self.setupPeanut()
    # self.setupHalfmoon()
    # self.setupButane()
    #self.butane_params = ButaneParameters(); self.paramss.append(self.butane_params)
    #self.halfmoon_params = HalfmoonParams(); self.paramss.append(self.halfmoon_params)
  
  def setupPeanut(self):
    params = PeanutParameters()
    keys_to_keep = ["diffusion","drift","D","UpperBound","LowerBound"]
    keys_to_delete = []
    for key in params.RHS_parameter:
      if key not in keys_to_keep:
        keys_to_delete.append(key)
    for key in keys_to_delete:
      del params.RHS_parameter[key]
    self.lin_paramss.append(params)
    Sim_parameter = {}
    Sim_parameter["X_int"] = np.asarray([[1,1,1]])
    Sim_parameter["T_max"] = 20*params.dt #10
    Sim_parameter["N"] = 4 #params.N
    Sim_parameter["dt"] = params.dt
    Sim_parameter["gap"] = 1
    self.lin_Sim_parameterss.append(Sim_parameter)
  
  def setupHalfmoon(self):
    params = HalfmoonParameters()
    self.nonlin_paramss.append(params)
    Sim_parameter = {}
    Sim_parameter["X_int"] = params.X_int
    Sim_parameter["T_max"] = 10
    Sim_parameter["gap"] = 1
    Sim_parameter["dt"] = params.dt
    self.nonlin_Sim_parameterss.append(Sim_parameter)
  
  def setupButane(self):
    params = ButaneParameters()
    self.lin_paramss.append(params)
    Sim_parameter = {}
    Sim_parameter["X_int"] = params.X_int
    Sim_parameter["T_max"] = 1 # 10
    Sim_parameter["N"] = 100 #params.N
    Sim_parameter["dt"] = params.dt
    self.lin_Sim_parameterss.append(Sim_parameter)

  def compute_matlab(self,function_name,RHS_parameter,Sim_parameter,\
    nout=1,verbose=False):
    arg_str="("+dict_to_struct(RHS_parameter,verbose=verbose)\
      +","+dict_to_struct(Sim_parameter,verbose=verbose)+")"
    if verbose:
      print("matlab function call:"+function_name+arg_str)
    matlab = octave.eval([function_name+arg_str],nout=nout)
    # if verbose:
    #   print("matlab:",matlab)
    return matlab

  def error_desc(self,name,mat,py):
    return name+" error: matlab type: "+str(type(mat))+"; matlab: "+str(mat)\
      +"; python type: "+str(type(py))+"; python: "+str(py)
  
  def get_observed(self, tested_function, RHS_parameter, Sim_parameter,verbose=False):
    observed = tested_function(RHS_parameter,Sim_parameter,verbose=verbose,rng=self.fake_rng)
    return observed

  def assertEqualDictionaries(self,test_name,matlab_dict,obs_data,\
    obs_Cov_store,obs_Mean_store,verbose=False):
    index_list = ["X_int","tN","dt","T_max","Tr_store","LowerBound","UpperBound"]
    for i in range(len(index_list)):
      index = index_list[i]
      
      assert is_close(obs_data[index], matlab_dict["data"][index],verbose=verbose), \
        test_name+" test at index"+index_list[i]
    assert is_close(matlab_dict["Cov_store"],obs_Cov_store,verbose=verbose),\
      test_name+" test in Cov_store failed"+self.error_desc("",\
          matlab_dict["Cov_store"],obs_Cov_store)
    assert is_close(matlab_dict["Mean_store"],obs_Mean_store,verbose=verbose),\
      test_name+" test in Mean_store failed"
  
  def run_multiple_traj_Euler_correctnessTest(self,test_name,\
      test_function,lin=True,verbose=False):
    if verbose:
      print("running a test")
    paramss = self.lin_paramss
    Sim_parameterss = self.lin_Sim_parameterss
    if not lin:
      paramss = self.nonlin_paramss
      Sim_parameterss = self.nonlin_Sim_parameterss
    for i in range(len(paramss)):
      params_obj = paramss[i]
      Sim_parameter = Sim_parameterss[i]
      # matlab=self.compute_matlab(test_name,params_obj.RHS_parameter,\
      #   Sim_parameter,nout=3,verbose=verbose)
      obs_data,obs_Cov_store,obs_Mean_store=self.get_observed(simEuler,
        params_obj.RHS_parameter,Sim_parameter)
      data,Cov_store,Mean_store=self.compute_matlab(test_name,params_obj.RHS_parameter,\
        Sim_parameter,nout=3,verbose=verbose)
      matlab = {"data":data,"Cov_store":Cov_store,"Mean_store":Mean_store}
      # return obs_Mean_store,matlab["Mean_store"]
      self.assertEqualDictionaries(test_name,matlab,obs_data,\
        obs_Cov_store,obs_Mean_store,verbose=verbose)
    # print(test_name+" correctness"+" "*(constants.passed_string_length-12-len(test_name))+"passed")
  
  def simEuler_correctness_test(self,verbose=False):
    test_name = "simEuler"
    return self.run_multiple_traj_Euler_correctnessTest(test_name,\
      simEuler,verbose=verbose)
  
  def simEuler_one_traj_correctness_test(self,verbose=False):
    test_name = "simEuler_one_traj_correctness"
    for i in range(len(self.lin_paramss)):
      params_obj = self.lin_paramss[i]
      Sim_parameter = self.lin_Sim_parameterss[i]
      matlab=self.compute_matlab("simEuler_one_traj",\
        params_obj.RHS_parameter,Sim_parameter,verbose=verbose)
      observed=self.get_observed(simEuler_one_traj,params_obj.RHS_parameter,\
        Sim_parameter,verbose=verbose)
      assert is_close(matlab,observed), test_name+" test failed"
  
  def simEuler_par_correctness_test(self,verbose=False):
    test_name = "simEuler_par"
    self.run_multiple_traj_Euler_correctnessTest(test_name,\
      simEuler_par,verbose=verbose)
  
  def simEulernonlintrans_correctness(self,verbose=False):
    test_name = "simEulernonlintrans_correctness"
    self.run_multiple_traj_Euler_correctnessTest(test_name,\
      simEulernonlintrans,verbose=verbose,lin=False)
    
  def simEulernonlintrans_one_traj_correctness(self,verbose=False):
    test_name = "simEulernonlintrans_one_traj_correctness"
    for i in range(self.nonlin_paramss):
      params_obj = self.nonlin_paramss[i]
      Sim_parameter = self.nonlin_Sim_parameterss[i]
      if params_obj.model_name == "Peanut":
        continue
      matlab=self.compute_matlab("simEulernonlintrans_one_traj",\
        params_obj.RHS_parameter,Sim_parameter,\
        verbose=verbose)
      observed=self.get_observed(simEulernonlintrans_one_traj,\
        params_obj.RHS_parameter,Sim_parameter,\
        rng=self.fake_rng,verbose=verbose)
    assert is_close(matlab,observed), test_name+" test failed"
    print(test_name+"."*(constants.passed_string_length-len(test_name))\
      + "passed")

  def simEulernonlintrans_par_correctness(self,verbose=False):
    test_name = "simEulernonlintrans_par_correctness"
    self.run_multiple_traj_Euler_correctnessTest(test_name,\
      simEulernonlintrans_par,verbose=verbose,lin=False)
  
  def simEuler_par_speedup_test(self,verbose=False):
    test_name = "simEuler_par_speedup"
    serial_total_time = 0
    par_total_time = 0
    for j in range(10):
      paramss = self.lin_paramss
      Sim_parameterss = self.lin_Sim_parameterss
      for i in range(len(paramss)):
        params_obj = paramss[i]
        Sim_parameter = Sim_parameterss[i]
        Sim_parameter["N"] = 1000
        serial_start_time = time.time()
        self.get_observed(simEuler,params_obj.RHS_parameter,\
          Sim_parameter,verbose=verbose)
        serial_total_time += (time.time() - serial_start_time)
        par_start_time = time.time()
        self.get_observed(simEuler_par,params_obj.RHS_parameter,\
          Sim_parameter,verbose=verbose)
        par_total_time += (time.time() - par_start_time)
    assert par_total_time < 1*(serial_total_time),\
      test_name+" failed with parallel time "+str(par_total_time)\
      +" and serial time "+str(serial_total_time)
    # print(test_name+"."*(constants.passed_string_length-len(test_name))+"passed")

  def simEulernonlintrans_par_speedup_test(self,verbose=False):
    test_name = "simEulernonlintrans_par_speedup"
    serial_total_time = 0
    par_total_time = 0
    for i in range(10):
      for j in range(len(self.nonlin_paramss)):
        params_obj = self.nonlin_paramss[j]
        Sim_parameter = self.nonlin_Sim_parameterss[j]
        serial_start_time = time.time()
        self.get_observed(simEulernonlintrans,params_obj.RHS_parameter,\
          params_obj.Sim_parameter,verbose=verbose)
        serial_total_time += (time.time() - serial_start_time)
        par_start_time = time.time()
        self.get_observed(simEulernonlintrans_par,params_obj.RHS_parameter,\
          params_obj.Sim_parameter,verbose=verbose)
        par_total_time += (time.time() - serial_start_time)
    assert (par_total_time < 0.5*(serial_total_time)) or serial_total_time<1e-5,\
      test_name+" failed with parallel time "+str(par_total_time)\
      +" and serial time "+str(serial_total_time)
    # print(test_name+" "*(constants.passed_string_length-len(test_name))+"passed")

  def run_all(self,verbose=False):
    for attribute in dir(self):
      if attribute.find("_test")==len(attribute)-5:
        if verbose:
          print("running",attribute)
        test = getattr(self, attribute)
        test(verbose=verbose)
        print(attribute+"."*(constants.passed_string_length-len(attribute))+"passed")
  
  def run_most(self,verbose=False):
    for attribute in dir(self):
      if attribute.find("_test")==len(attribute)-5:
        if attribute.find("nonlintrans") != -1:
          continue
        if attribute.find("speedup") != -1:
          continue
        if verbose:
          print("running",attribute)
        test = getattr(self, attribute)
        test(verbose=verbose)
        print(attribute+"."*(constants.passed_string_length-len(attribute))+"passed")