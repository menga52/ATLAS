from scipy.stats import norm
import numpy as np
from math import *
from ATLAS.Simulator import simEuler_one_traj, simEuler, simEuler_par

pi = 3.14159265

null = -1 # not yet implemented

# rescales a list. Also converts any iterable to type list
def scale(ls, factor=1):
	print("deprecated")
	return [a*factor for a in ls]

#TODO: verify scaling factor is factor^2
def randn(variance, quantity):
	print("deprecated")
	return scale(norm.rvs(size=quantity), variance**0.15)
	
def componentwise_sum(ls1, ls2):
	# assumes len(ls1)=len(ls2)
	print("deprecated")
	ret = scale(ls1,1)
	for i in range(0,len(ls2)):
		ret[i] += ls2[i]
	return ret
	
class RHSParams:
	def __init__(self,l,k2,k3,theta,c1,c2,c3,beta,sigma):
		print("depcrecated")
		self.l=l
		self.k2=k2
		self.k3=k3
		self.theta=theta
		self.c1=c1
		self.c2=c2
		self.c3=c3
		self.beta=beta
		self.sigma=sigma
		
class AllRHSParams:
	def __init__(self,dt,T_max,N,t0,chi_p,threshold,modify,connectivity_threshold,D,d,parameter,drift,diffusion,UpperBound,LowerBound):
		print("depcrecated")
		self.dt=dt
		self.T_max=T_max
		self.N=N
		self.t0=t0
		self.chi_p=chi_p
		self.threshold=threshold
		self.modify=modify
		self.connectivity_threshold=connectivity_threshold
		self.D=D
		self.d=d
		self.parameter=parameter
		self.drift=drift
		self.diffusion=diffusion
		self.UpperBound=UpperBound
		self.LowerBound=LowerBound

class MSMParameters:
	def __init__(self,step,N_state,dt_s):
		print("depcrecated")
		self.step=step
		self.N_state=N_state
		self.dt_s=dt_s

class ChartSimulatorParams:
	def __init__(self,X_int,nearest,t0,dt_s,D,d,Nstep,gap,connectivity,explore_threshold,N):
		print("depcrecated")
		self.X_int=X_int
		self.nearest=nearest
		self.t0=t0
		self.dt_s=dt_s
		self.D=D
		self.d=d
		self.Nstep=Nstep
		self.gap=gap
		self.connectivity=connectivity
		self.explore_threshold=explore_threshold
		self.N=N
		
class RelearnParams:
	def __init__(self,N_relearn,iter,RHS_parameter,relative_threshold,simulator_par):
		print("depcrecated")
		self.N_relearn=N_relearn
		self.iter=iter
		self.RHS_parameter=RHS_parameter
		self.relative_threshold=relative_threshold
		self.simulator_par=simulator_par
	

class Parameters:
	# looks odd, but allows for easier inheritance
	def __init__(self):
		print("__init__() in Parameters")
		self.init()

	def init(self):
		print("init() in Parameters")
		
		self.setup_basic_params()
		self.setup_parameter()
		
		self.drift = null # @(X) get_drift(X, parameter)
		self.diffusion = null # @(X) get_diffusion(X, parameter)
		
		self.setup_RHS_parameter()
		self.setup_simulators()
		self.setup_MSM_parameter()
		self.setup_chart_sim_parameter()
		self.setup_relearn_parameter()
		
		# parameter for MFPTlength
		self.N_IC = 30000
		
		# file path
		#TODO: should these be public?
		self.chart_fileName                = null
		self.chart_part_fileName           = null
		self.chart_plot_fileName           = null
		self.TranM_fileName                = null
		self.TranM_plot_fileName           = null
		self.chart_relearn_fileName        = null
		self.well_fileName                 = null
		self.FPT_fileName                  = null
		
	def setup_basic_params(self):
		self.D = 6 # Dimension of the full system
		self.d = 1 # The dimension of the slow manifold. Later we can estimate this dimenstion
		self.chi_p = 1.96 # for 95% Confidence interval for 1D Gaussian distribution
		self.K_int = 2 # Number of initial points on the manifold.
		self.N = 1600000  # Number of short traj starting from the same initial point
		self.dt = 1*10**(-6) # Simulation time-step
		self.T_one = 10 # Time for single long trajectory to learn ini K charts. 
		self.relearn_option = 1 # option to relearn
		self.t0 = 10*self.dt # The time as the benchmark. Drift and diffusion varies small within t0
		self.T_max = 50*self.dt # 50*dt #Time for short trajectories.
		self.X_int = componentwise_sum([-1.4461,0.5578,1.5361,1.4268,-2.0593,0.1615], scale(randn(1,6),0.1))
		self.threshold = [1.5, 0.001] # The first one is R_max determined by the diameter of the slow manifold when setting the value of R and the second one threshold for occational prj
		self.connectivity_threshold = 4 #  two landmarks are connected if they are with in r*sqrt(t0)
		self.explore_threshold = 0.95
		self.option = 1 # 1 for fast mode prj, 2 for orthogonal prj
		self.modify = 0 # 0 use the intercept(t=0) 1 use the tau_min point as the landmark
		# The training window is larger than the relaxtion here. It is to ensure
		# the landmark are sufficiently relaxed onto the manifold
		self.LowerBound = 40*self.dt # 40*dt #  Lowerbound of the training window
		self.UpperBound = 50*self.dt # 50*dt #  Upperbound of the training window
		
		# parameters for simulation
		self.Nstep = 2*10**2
		self.gap = 10
		
		# parameters for RHS
		self.l = 1.53
		self.k2 = 319225 # 1.17*10^6;
		self.k3 = 62500
		self.theta = 112/180*pi # 112 degree
		self.c1 = 2037.82
		self.c2 = 158.52
		self.c3 = -3227.70
		self.beta = 4*10^-3
		self.sigma = (2*self.beta**(-1))**0.5
		
	def setup_parameter(self):
		# self.parameter = RHSParams(self.l,self.k2,self.k3,self.theta,self.c1,self.c2,self.c3,self.beta,self.sigma)
		self.parameter = {}
		self.parameter["l"]=self.l
		self.parameter["k2"]=self.k2
		self.parameter["k3"]=self.k3
		self.parameter["theta"]=self.theta
		self.parameter["c1"]=self.c1
		self.parameter["c2"]=self.c2
		self.parameter["c3"]=self.c3
		self.parameter["beta"]=self.beta
		self.parameter["sigma"]=self.sigma
		
	def setup_RHS_parameter(self):
		# self.RHS_parameter = AllRHSParams(self.dt,self.T_max,self.N,self.t0,self.chi_p,self.threshold,self.modify,self.connectivity_threshold,self.D,self.d,self.parameter,self.drift,self.diffusion,self.UpperBound,self.LowerBound)
		self.RHS_parameter = {}
		self.RHS_parameter["dt"]=self.dt
		self.RHS_parameter["T_max"]=self.T_max
		self.RHS_parameter["N"]=self.N
		self.RHS_parameter["t0"]=self.t0
		self.RHS_parameter["chi_p"]=self.chi_p
		self.RHS_parameter["threshold"]=self.threshold
		self.RHS_parameter["modify"]=self.modify
		self.RHS_parameter["connectivity_threshold"]=self.connectivity_threshold
		self.RHS_parameter["D"]=self.D
		self.RHS_parameter["d"]=self.d
		self.RHS_parameter["parameter"]=self.parameter
		self.RHS_parameter["drift"]=self.drift
		self.RHS_parameter["diffusion"]=self.diffusion
		self.RHS_parameter["UpperBound"]=self.UpperBound
		self.RHS_parameter["LowerBound"]=self.LowerBound
	
	def setup_simulators(self):
		# simulators
		self.simulator_one = simEuler_one_traj
		self.simulator_no_par = simEuler
		self.simulator_par = simEuler_par

	def setup_MSM_parameter(self):
		# self.MSM_parameter = MSMParameters(1,1*10**6,10*self.dt)
		self.MSM_parameter = {}
		self.MSM_parameter["step"]=1
		self.MSM_parameter["N_state"]=1*10**6
		self.MSM_parameter["dt_s"]=10*self.dt
	
	def setup_chart_sim_parameter(self):
		#self.chart_sim_parameter = ChartSimulatorParams([],1,self.t0,10*self.dt,self.RHS_parameter.D,self.RHS_parameter.d,self.Nstep,self.gap,[],self.explore_threshold,self.N)
		self.chart_sim_parameter = {}
		self.chart_sim_parameter["X_int"]=[]
		self.chart_sim_parameter["nearest"]=1
		self.chart_sim_parameter["t0"]=self.t0
		self.chart_sim_parameter["dt_s"]=10*self.dt
		self.chart_sim_parameter["D"]=self.RHS_parameter["D"]
		self.chart_sim_parameter["d"]=self.RHS_parameter["d"]
		self.chart_sim_parameter["Nstep"]=self.Nstep
		self.chart_sim_parameter["gap"]=self.gap
		self.chart_sim_parameter["connectivity"]=[]
		self.chart_sim_parameter["explore_threshold"]=self.explore_threshold
		self.chart_sim_parameter["N"]=self.N
	
	def setup_relearn_parameter(self):
		# Addendum: will need to be changed to avoid pointer related errors
		self.RHS_parameter_relearn = self.RHS_parameter # one can change some RHS if you believe the invariant manifold is not changed
		#self.relearn_parameter = RelearnParams(6*self.N,15,self.RHS_parameter_relearn,[0.01,0.05],null)
		self.relearn_parameter = {}
		self.relearn_parameter["N_relearn"]=6*self.N
		self.relearn_parameter["iter"]=15
		self.relearn_parameter["RHS_parameter"]=self.RHS_parameter_relearn
		self.relearn_parameter["relative_threshold"]=[0.01,0.05]
		self.relearn_parameter["simulator_par"]=null
		# last argument above is simEuler_par(RHS_parameter,Sim_parameter)
		
	def testPrint(self):
		print("test print")
		