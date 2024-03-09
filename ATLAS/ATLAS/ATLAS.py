def initial_learning(verbose=False):
	if verbose:
		print("Starting initial learning stage")
	Learning_ini_chart()
	
def Learning_ini_chart(model):
	# Find the chart for randomly sampled initial points on the manifold.
	# These initial points are evenly sampled from a long trajectory
	
	chart = np.empty(shape(1,K_int),dtype=dict)
	