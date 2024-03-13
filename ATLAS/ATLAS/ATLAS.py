def initial_learning(verbose=False):
	if verbose:
		print("Starting initial learning stage")
	Learning_ini_chart()
	
def Learning_ini_chart(model,verbose=False):
	# Find the chart for randomly sampled initial points on the manifold.
	# These initial points are evenly sampled from a long trajectory
	
	chart = np.empty(shape(1,K_int),dtype=dict)
	p = model.parameters
	Sim_parameter = {}
	Sim_parameter["T_max"] = p["T_one"]
	Sim_parameter["dt"] = p["dt"]
	Sim_parameter["gap"] = 1 #???
	Sim_parameter["X_int"] = p["X_int"]
	if verbose:
		print("Initial point:",Sim_parameter["X_int"],"T_max:",Sim_parameter["T_max"])
	
	#TODO: track time
	