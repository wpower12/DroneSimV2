import numpy as np

class Wind():
	def __init__(self):
		pass

	def sample_wind(self):
		# Right now, just a 'unit wind' in the x direction
		return np.asarray([1.0, 1.0, 0.0])