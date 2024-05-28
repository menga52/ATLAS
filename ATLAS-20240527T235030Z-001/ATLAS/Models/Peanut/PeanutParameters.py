import numpy as np
from numpy import sin, cos

from ATLAS.Models.Parameters import *
from ATLAS.Models.Peanut.get_diffusion import get_diffusion
from ATLAS.Models.Peanut.get_drift import get_drift
from ATLAS.Simulator.simEuler import simEuler
from ATLAS.Simulator.simEuler_one_traj import simEuler_one_traj
from ATLAS.Simulator.simEuler_par import simEuler_par

seed = 1
default_rng = np.random.default_rng(seed)
class PeanutParameters(Parameters):

  def __init__(self,verbose=False,rng=-1):
    if rng == -1: self.rng = default_rng
    else:         self.rng = rng
    
    self.init(verbose=verbose)
    self.D = 3 # Dimension of the full system
    self.d = 2 # Dimension of the slow manifold. Later, we can estimate this dimension
    self.chi_p = 5.991 # 95% confidence interval
    self.K_int = 100 # number of initial points on the manifold
    self.N = 400000 # number of short trajectories starting from the same initial point
    self.epsilon = 0.005 # small number
    self.dt = 5*10**(-4) # simulation time-step
    self.T_max = 0.1 # time for short trajectories
    self.t0 = 0.1 # the relaxation time
    self.T_one = 1000 # Time for single long trajectory to learn ini K charts
    self.threshold = [1,0.001] # R_max, determined by the diameter of the slow slow manifold when setting the value of R or the maximum possible layer
    self.connectivity_threshold = 3 # two landmarks are connected if they are within r*sqrt(t0)
    self.explore_threshold = 1.05
    self.LowerBound = 0.05 # Lower bound of the training window
    self.UpperBound = 0.10 # Upper bound of the training window
    self.relearn_option = 0
    self.diffusion = self.peanut_diffusion
    self.drift = self.peanut_drift
    self.model_name = "Peanut"
    
    # parameters for simulation
    self.Nstep = 5*10**6
    
    # parameters of RHS equations
    self.a1 = 4
    self.a2 = 8
    self.c1 = 2
    self.c2 = 0.5
    self.c3 = 0.05
    self.c4 = 0.4
    self.c5 = 0.05
    self.c6 = 0.4 # 0.5
    
    # set up an initial point. TODO: rename to init
    self.theta_int = self.rng.random() * pi # runif(0,pi)
    self.phi_int = self.rng.random() * 2*pi # runif(0,2pi)
    self.R_int = self.R(self.theta_int,self.phi_int)
    self.X_int = np.asarray([[self.R_int*sin(self.theta_int)*cos(self.phi_int),\
                      self.R_int*sin(self.theta_int)*sin(self.phi_int),\
                      self.R_int*cos(self.theta_int)]])
    
    self.setup_RHS_parameter()
    self.setup_Exact_parameter()
    self.setup_chart_sim_parameter()
    self.setup_MSM_parameter()
    self.setup_relearn_parameter()
    
    self.N_IC = 15000
    
    # file path
    chart_fileName = null # [datapath,'chart.mat'];
    chart_part_fileName = null # [datapath,'chart_part.mat'];
    TranM_fileName = null # [datapath,'TranM.mat'];
    FPT_fileName = null # [datapath,'Peanut_FPT.mat'];
    
  def setup_RHS_parameter(self,verbose=False):
    if verbose:
      print("PeanutParameters.setup_RHS_parameter()")
    self.RHS_parameter = {}
    self.RHS_parameter["dt"] = self.dt
    self.RHS_parameter["T_max"] = self.T_max
    self.RHS_parameter["N"] = self.N
    self.RHS_parameter["t0"] = self.t0
    self.RHS_parameter["chi_p"] = self.chi_p
    self.RHS_parameter["threshold"] = self.threshold
    self.RHS_parameter["connectivity_threshold"] = self.connectivity_threshold
    self.RHS_parameter["D"] = self.D
    self.RHS_parameter["d"] = self.d
    self.RHS_parameter["modify"] = self.modify
    self.RHS_parameter["R"] = self.R
    self.RHS_parameter["diffusion"] = self.peanut_diffusion
    self.RHS_parameter["drift"] = self.peanut_drift
    self.RHS_parameter["UpperBound"] = self.UpperBound
    self.RHS_parameter["LowerBound"] = self.LowerBound
    if verbose:
      print("end PeanutParameters.setup_RHS_parameter()")
    
  def setup_Exact_parameter(self):
    self.Exact_parameter = {}
    self.Exact_parameter["H_true_X"]=self.H_trueX
    self.Exact_parameter["d"]=self.d
    self.Exact_parameter["D"]=self.D
    self.Exact_parameter["R_true"]=self.R_true
    self.Exact_parameter["b_true"]=self.b_true
    self.Exact_parameter["H_true"]=self.H_true
    self.Exact_parameter["Lambda_true"]=self.Lambda_true
    
  def setup_chart_sim_parameter(self):
    self.chart_sim_parameter = {}
    self.chart_sim_parameter["X_int"]=[]
    self.chart_sim_parameter["nearest"]=4
    self.chart_sim_parameter["t0"]=self.t0
    self.chart_sim_parameter["dt_s"]=self.t0
    self.chart_sim_parameter["D"]=self.D
    self.chart_sim_parameter["d"]=self.d
    self.chart_sim_parameter["Nstep"]=self.Nstep
    self.chart_sim_parameter["gap"]=self.gap
    self.chart_sim_parameter["connectivity"]=[]
    self.chart_sim_parameter["explore_threshold"]=self.explore_threshold
    self.chart_sim_parameter["N"]=self.N
    
  def setup_MSM_parameter(self):
    self.MSM_parameter = {}
    self.MSM_parameter["step"] = 1
    self.MSM_parameter["N_state"] = 1*10**6
    self.MSM_parameter["dt_s"] = self.t0
    
  def setup_relearn_parameter(self):
    # parameter for relearning
    self.relearn_parameter = {}
    self.relearn_parameter["N_relearn"] = 10*self.N
    self.relearn_parameter["iter"] = 2
    self.relearn_parameter["relative_threshold"] = [0.005,0.02]
    self.relearn_parameter["simulator_par"] = self.simHeun_par1(self.RHS_parameter)
    
  def simHeun_par1(self,param):
    return null # simHeun_par(self.RHS_parameter,param)
  
  # Specify the manifold
  def R(self, theta, phi,verbose=False):
    return (self.a1+self.a2*(cos(theta))**2)**0.5
  # Specify the gradient
  def Rprime(self, theta, phi,verbose=False):
    return -self.a2*cos(theta)*sin(theta)/(self.a1+self.a2*cos(theta)**2)**0.5
    
  # Specify the drift and diffusion in sph coordinate
  def r_RHS(self,r,theta,phi,verbose=False):
    return -(self.c1/self.epsilon)/r*(r-self.R(theta,phi))  
  def theta_RHS(self,r,theta,phi,verbose=False):
    return self.c3*cos(3*theta)/(r*sin(theta)) # -c3*sin(2*theta+pi/4)/r
  def phi_RHS(self,r,theta,phi,verbose=False):
    return self.c5*sin(phi+theta)/r # c5*sin(phi)/r
  def sigma_r(self,r,theta,phi,verbose=False):
    return self.c2/((self.epsilon**0.5)*r)
  def sigma_theta(self,r,theta,phi,verbose=False):
    return self.c4*sin(theta)/r
  def sigma_phi(self,r,theta,phi,verbose=False):
    return self.c6/r
  def b_sph(self,r,theta,phi,verbose=False):
    return np.asarray([[self.r_RHS(r,theta,phi)], [self.theta_RHS(r,theta,phi)], [self.phi_RHS(r,theta,phi)]])
  def Sigma_sph(self,r,theta,phi,verbose=False):
    return np.asarray([[self.sigma_r(r,theta,phi)], [self.sigma_theta(r,theta,phi)], [self.sigma_phi(r,theta,phi)]])
  
  # pre-load parameters to save time
  def peanut_drift(self,X,verbose=False):
    return get_drift(X,self.epsilon, self.a1,self.a2,self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,verbose=verbose)
  def peanut_diffusion(self,X,verbose=False):
    return get_diffusion(X,self.epsilon,self.c2,self.c4,self.c6,verbose=verbose)
  # Specify the drift and diffusion term on the reduced system. The drift term has two parts b_stra and b_ito. The latter is the ito term
  def b_stra(self,theta,phi,verbose=False):
    R = self.R(theta,phi)
    Rp = self.Rprime(theta,phi)
    st = sin(theta)
    ct = cos(theta)
    sp = sin(phi)
    cp = cos(phi) # below are column vectors
    ls1 = [[(Rp*st+R*ct)*cp], [(Rp*st+R*ct)*sp], [(Rp*ct-R*st)]]
    ls2 = [[ -R*st*sp], [R*st*cp],[0]]
    return self.theta_RHS(R,theta,phi,verbose=verbose)*np.array(ls1) + self.phi_RHS(R,theta,phi,verbose=verbose)*np.array(ls2)
  
  # this part is calculated by mathematica
  def b_ito(self,theta,phi,verbose=False):
    a1 = self.a1
    a2 = self.a2
    ct = cos(theta)
    cp = cos(phi)
    st = sin(theta)
    ct = cos(theta)
    c4 = self.c4
    row1 = [(1/2)*(a1+a2*ct**2)**(-5/2)*cp*st*((-1)*self.c6**2*(a1+a2*ct**2)**2+(-1)*c4**2*(a1+a2*ct**2)*(a1+4*a2*ct**2)*st**2+a1*a2*c4**2*st**4)]
    row2 = [(-1/2)*(a1+a2*ct**2)**(-5/2)*st*(self.c6**2*(a1+a2*ct**2)**2+c4**2*(a1+a2*ct**2)*(a1+4*a2*ct**2)*st**2+(-1)*a1*a2*c4**2*st**4)*sin(phi)]
    row3 = [(-1/4)*c4**2*ct*(a1+a2*ct**2)**(-5/2)*(2*a1**2+a2**2+2*a2*(3*a1+a2)*cos(2*theta)+a2**2*cos(4*theta))*st**2]
    return np.asarray([row1,row2,row3])
  
  #TODO: improve names  
  def b_true(self,X, b_ito=None, b_stra=None,verbose=False):
    if b_ito==None:
      b_ito = self.b_ito
    if b_stra==None:
      b_stra = self.b_stra
    x = X[0]
    y = X[1]
    z = X[2]
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    if verbose:
      print("theta:",theta)
      print("phi",phi)
      print("b_ito",b_ito(theta,phi))
      print("b_stra",b_stra(theta,phi))
    b = b_ito(theta, phi) + b_stra(theta, phi)
    return b
    
  def Lambda_true(self,X,H_true=None,verbose=False):
    if H_true==None:
      H_true = self.H_true
    x = X[0]
    y = X[1]
    z = X[2]
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    Lambda = H_true(theta,phi) * H_true(theta, phi)
    return Lambda

  def R_true(self,X,R=None,verbose=False):
    if R==None:
      R = self.R
    x = X[0]
    y = X[1]
    z = X[2]
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    phi = np.arctan2(y,x)
    Rx = R(theta, phi)
    return Rx
    
  def H_trueX(self,X,theta=None,phi=None,verbose=False):
    x = X[0]
    y = X[1]
    z = X[2]
    if theta==None:
      theta = np.arctan2(np.sqrt(x**2+y**2),z)
    if phi==None:
      phi = np.arctan2(y,x)
    st = np.sin(theta) # abbreviations for readability
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    R = self.R(theta,phi)
    Rp = self.Rprime(theta, phi)
    sig_theta = self.sigma_theta(R, theta, phi)
    sig_phi = self.sigma_phi(R, theta, phi)
    ret = np.array([[(Rp*st+R*ct)*cp*sig_theta, -R*st*sp*sig_phi],
    [( Rp*st+R*ct)*sp*sig_theta, R*st*cp*sig_phi],
    [( Rp*ct-R*st)*sig_theta,0 ]])

  def H_true(self,theta,phi,verbose=False):
    R = self.R(theta,phi)
    Rp = self.Rprime(theta,phi)
    st = sin(theta)
    ct = cos(theta)
    sp = sin(phi)
    cp = cos(phi)
    ls = [[(Rp*st+R*ct)*cos(phi)*self.sigma_theta(R,theta,phi), -R*st*sp*self.sigma_phi(R,theta,phi)], [(Rp*st+R*ct)*sp*self.sigma_theta(R,theta,phi),R*st*cp*self.sigma_phi(R,theta,phi)],[(Rp*ct-R*st)*self.sigma_theta(R,theta,phi),0]]
    return np.array(ls)
    
  def simulator_one(self,Sim_parameter,verbose=False,rng=-1):
    return simEuler_one_traj(self.RHS_parameter,\
      Sim_parameter,verbose=verbose,rng=rng)
  def simulator_no_par(self,Sim_parameter,verbose=False,rng=-1):
    return simEuler(self.RHS_parameter,\
      Sim_parameter,verbose=verbose,rng=rng)
  def simulator_par(self,Sim_parameter,verbose=False,rng=-1):
    return simEuler_par(self.RHS_parameter,\
    Sim_parameter,verbose=verbose,rng=rng)
  
  
  
  