#TODO: omit?
def sqrt(x):
	return x**0.5

class Model:
	def __init__(self):
		self.init()
	
	def init(self):
		# create parameters object
		parameters = self.createParameters()
	
	def learn(self):
		# initial learning
		self.initial_learning()
		# exploration learning
		self.exploration_learning()
		# simulation learning
		self.simulation_learning()
		# relearning - butane + halfmoon only
		self.relearning()
		# MSM learning
		self.MSM_learning()
		