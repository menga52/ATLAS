import numpy as np
import jax.numpy as jnp
from ATLAS.Tests.is_close import is_close
from ATLAS.Models.Peanut.PeanutParameters import PeanutParameters
from ATLAS.Models.Peanut.get_drift import get_drift
from ATLAS import constants
from oct2py import octave
octave.addpath("/content/drive/MyDrive/ATLAS/Deterministic/MATLAB/model/Peanut/")

seed = 1
default_rng = np.random.default_rng(seed)
verbose = False


class PeanutParametersTest:
  def __init__(self, rng=-1):
    self.params_obj = PeanutParameters(verbose=verbose)
    # for k in [str(type(getattr(self.params_obj,name)))+" "+name+"\n" for name in dir(self.params_obj)]:
    #   print(k)
    self.rng = rng
    if rng==-1:
      self.rng = default_rng
      self.test_args2 = [[2,3],[3.5,3.5],[0.6,0.8]]
      self.test_args3 = [[1,2,3],[3.5,3.5,3.5],[0.9,0.6,0.8]]
      # self.test_args3 = [jnp.array([X]) for X in self.test_args3]
      self.num_rand_tests = 0
      self.rand_arg_bounds2 = [[0,1],[0,1]]
      self.rand_arg_bounds3 = [[0,1],[0,1],[0,1]]
    if self.num_rand_tests > 0:
      self.compute_random_args()
    self.Xs = [[[1,2,3]],[[4,5,6]],[[2,2,2]],[[1,1,1]],[[1.1,1.2,1.3]]]
    self.Xs = [[jnp.array(X).reshape(-1,1)] for X in self.Xs] # column vectors
  

  def compute_random_args(s):
    # s = self
    s.test_args2 = np.zeros((s.num_rand_tests,2))
    s.test_args3 = np.zeros((s.num_rand_tests,3))
    s.test_args2[:,0] = s.rng.uniform(s.rand_arg_bounds[0][0],s.rand_arg_bounds[0][1],num_rand_tests)
    s.test_args2[:,1] = s.rng.uniform(s.rand_arg_bounds[1][0],s.rand_arg_bounds[1][1],num_rand_tests)
    s.test_args3[:,0] = s.rng.uniform(s.rand_arg_bounds[0][0],s.rand_arg_bounds[0][1],num_rand_tests)
    s.test_args3[:,1] = s.rng.uniform(s.rand_arg_bounds[1][0],s.rand_arg_bounds[1][1],num_rand_tests)
    s.test_args3[:,2] = s.rng.uniform(s.rand_arg_bounds[2][0],s.rand_arg_bounds[2][1],num_rand_tests)
  
  def compute_matlab(self, function_name, test_args,verbose=False):
    # function_name - matlab's name for the function being tested
    # test_args - the arguments to which the matlab function will be applied
    num_tests = len(test_args)
    num_args = len(test_args[0])
    function_output = np.empty(num_tests,dtype=object)
    test_args_formatted = ["("+",".join([str(n) for n in ls])+")" for ls in test_args]
    
    for test_index in range(num_tests):
      argstring = test_args_formatted[test_index]
      if verbose:
        print(function_name+argstring)
      function_output[test_index] = octave.eval(["set_parameter", function_name+argstring],verbose=verbose)
    return function_output
  
  def get_observed(self, tested_function, test_args,verbose=verbose):
    num_tests = len(test_args)
    num_args = len(test_args[0])
    observed = np.empty(num_tests, dtype=object)
    for test_index in range(num_tests):
      args = test_args[test_index]
      if verbose:
        print("observed arguments:",*args)
      observed[test_index] = tested_function(*args,verbose)
    return observed
  
  def assert_match(self, test_name, observed_list, matlab_list,verbose=False): 
    for i in range(len(observed_list)):
      if verbose:
        print(type(matlab_list[0]))
        print(type(observed_list[0]))
      assert is_close(observed_list[i], matlab_list[i], verbose), str(observed_list[i])+";\n"+str(matlab_list[i])+";\n"+test_name \
      +" test failed at index "+str(i)+" " +\
      "observed:"+str(observed_list)+";\n"+"matlab:"+str(matlab_list)
    print(test_name+(constants.passed_string_length-len(test_name))*'.'+"passed")

  """
  parameters tests
  """
  def R_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.R, self.test_args2)
    matlab = self.compute_matlab("R", self.test_args2)
    self.assert_match("R",observed,matlab)

  def Rprime_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.Rprime, self.test_args2)
    matlab = self.compute_matlab("Rprime", self.test_args2)
    self.assert_match("Rprime",observed,matlab)
  
  def r_RHS_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.r_RHS, self.test_args3)
    matlab = self.compute_matlab("r_RHS", self.test_args3)
    self.assert_match("r_RHS",observed,matlab)
  
  def theta_RHS_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.theta_RHS, self.test_args3)
    matlab = self.compute_matlab("theta_RHS", self.test_args3)
    self.assert_match("theta_RHS",observed,matlab)

  def phi_RHS_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.phi_RHS, self.test_args3)
    matlab = self.compute_matlab("phi_RHS", self.test_args3)
    self.assert_match("phi_RHS",observed,matlab)
    
  def sigma_r_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.sigma_r, self.test_args3)
    matlab = self.compute_matlab("sigma_r", self.test_args3)
    self.assert_match("sigma_r",observed,matlab)

  def sigma_theta_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.sigma_theta, self.test_args3)
    matlab = self.compute_matlab("sigma_theta", self.test_args3)
    self.assert_match("sigma_theta",observed,matlab)
    
  def sigma_phi_test(self,verbose=False):
    observed = self.get_observed(self.params_obj.sigma_phi, self.test_args3)
    matlab = self.compute_matlab("sigma_phi", self.test_args3)
    self.assert_match("sigma_phi",observed,matlab)
  
  def b_sph_test(self,verbose=False):
    matlab = self.compute_matlab("b_sph", self.test_args3)
    observed = self.get_observed(self.params_obj.b_sph, self.test_args3)
    self.assert_match("b_sph",observed, matlab)

  def Sigma_sph_test(self,verbose=False):
    matlab = self.compute_matlab("Sigma_sph", self.test_args3)
    observed = self.get_observed(self.params_obj.Sigma_sph, self.test_args3)
    self.assert_match("Sigma_sph",observed, matlab)
  
  def drift_test(self,verbose=False):
    matlab = self.compute_matlab("drift", self.Xs,verbose=verbose)
    observed = self.get_observed(self.params_obj.drift,self.Xs)
    self.assert_match("drift",observed, matlab)
  
  def diffusion_test(self,verbose=False):
    matlab = self.compute_matlab("diffusion", self.Xs,verbose=verbose)
    observed = self.get_observed(self.params_obj.diffusion,self.Xs,verbose=verbose)
    self.assert_match("diffusion",observed, matlab,verbose=verbose)
  
  def b_stra_test(self,verbose=False):
    matlab = self.compute_matlab("b_stra",self.test_args2)
    observed = self.get_observed(self.params_obj.b_stra, self.test_args2)
    self.assert_match("b_stra", matlab, observed)
  
  def b_ito_test(self,verbose=False):
    matlab = self.compute_matlab("b_ito",self.test_args2)
    observed = self.get_observed(self.params_obj.b_ito, self.test_args2)
    self.assert_match("b_ito", matlab, observed)

  """
  other tests
  """

  def run_all(self,verbose=False):
    for attribute in dir(self):
      if attribute.find("_test")==len(attribute)-5:
        if verbose:
          print("running",attribute)
        test = getattr(self, attribute)
        test(verbose=verbose)


# test_obj = PeanutParametersTest()
# test_obj.run_all()
