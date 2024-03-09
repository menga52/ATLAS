from ATLAS.Models.Parameters import *

class HalfmoonParameters(Parameters):
	def __init__(self):
		self.init()
		self.D = 20
		self.K_int = 10
		self.N = 8000000
		self.dt = 0.05
		self.T_one = 800000
		
		self.t0 = 20*self.dt
		self.T_max = 25*self.dt
		self.X_int = np.ones(self.D)
		self.X_int[2] = 0
		self.threshold = [0.5,0.01]
		self.LowerBound = 20*self.dt
		self.UpperBound = 25*self.dt
		self.Nstep = 1*10**5
		self.gap = 1
		
		# not in Butane
		self.a1 = 0
		self.a2 = 5*10**(-3)
		self.a3 = 2.5*10**(-3)
		self.a4 = 0.06
		self.eps = 0.01
		self.b1 = 0.04/self.eps
		self.b2 = 0.035/self.eps**(0.5)
		self.b3 = 0.05/self.eps
		self.b4 = 0.02/self.eps**(0.05)
		self.t  = -1
		
		self.Tran = np.diag(np.ones(self.D-2),1)
		self.Tran[:,0]=np.ones(self.D-1)
		self.Tran = self.Tran[1:,:]
		
		self.Tran_inv = np.diag(np.ones(self.D-2),1)
		self.Tran_inv[:,0] = np.ones(self.D-1)
		self.Tran_inv = self.Tran_inv[:self.D-2,:]
		
		# parameter dictionary is in Butane, but consituents are different
		self.parameter = {}
		self.parameter["a1"] = self.a1
		self.parameter["a2"] = self.a2
		self.parameter["a3"] = self.a3
		self.parameter["a4"] = self.a4
		self.parameter["b1"] = self.b1
		self.parameter["b2"] = self.b2
		self.parameter["b3"] = self.b3
		self.parameter["b4"] = self.b4
		self.parameter["Tran"] = self.Tran
		self.parameter["Tran_inv"] = self.Tran_inv
		
		self.drift = null
		self.diffusion = null
		self.nonlin_trans = null
		self.nonlin_trans_inv = null
		
		self.RHS_parameter = {}
		self.RHS_parameter["T_max"] = self.T_max
		self.RHS_parameter["dt"] = self.dt
		self.RHS_parameter["N"] = self.N
		self.RHS_parameter["t0"] = self.t0
		self.RHS_parameter["chi_p"] = self.chi_p
		self.RHS_parameter["threshold"] = self.threshold
		self.RHS_parameter["connectivity_threshold"] = self.connectivity_threshold
		self.RHS_parameter["D"] = self.D
		self.RHS_parameter["d"] = self.d
		self.RHS_parameter["modify"] = self.modify
		self.RHS_parameter["drift"] = self.drift
		self.RHS_parameter["diffusion"] = self.diffusion
		self.RHS_parameter["nonlin_trans"] = self.nonlin_trans
		self.RHS_parameter["nonlin_trans_inv"] = self.nonlin_trans_inv
		self.RHS_parameter["UpperBound"] = self.UpperBound
		self.RHS_parameter["LowerBound"] = self.LowerBound
		
		self.simulator_one = null
		self.simulator_no_par = null
		self.simulator_par = null
		
		self.MSM_parameter = {}
		self.MSM_parameter["step"] = 1
		self.MSM_parameter["N_state"] = 1*10**6
		self.MSM_parameter["dt_s"] = self.t0
		
		#TODO: this will cause issues with pointers
		self.RHS_parameter_relearn = self.RHS_parameter
		
		self.relearn_parameter = {}
		self.relearn_parameter["N_relearn"] = self.N
		self.relearn_parameter["iter"] = 1
		self.relearn_parameter["RHS_parameter"] = self.RHS_parameter_relearn
		self.relearn_parameter["relative_threshold"] = [0.05,0.1]
		self.relearn_parameter["simulator_par"] = null
		
		self.chart_sim_parameter["nearest"] = 4
		self.chart_sim_parameter["t"] = self.t
		self.chart_sim_parameter["t0"] = self.t0
		self.chart_sim_parameter["dt_s"] = self.t0
		self.chart_sim_parameter["D"] = self.D
		self.chart_sim_parameter["Nstep"] = self.Nstep
		self.chart_sim_parameter["gap"] = self.gap
		self.chart_sim_parameter["N"] = self.N
		