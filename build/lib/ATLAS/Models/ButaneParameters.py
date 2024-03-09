from ATLAS.Models.Parameters import *

class ButaneParameters(Parameters):
	def __init__(self):
		self.init()
		print(self.D)
		print("should have printed 6")