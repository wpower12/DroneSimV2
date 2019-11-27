import numpy as np
from . import constants as C

class Wind():
	def __init__(self):
		self.gusting = False
		self.gust_length = 0 # How many ticks (state-advances) a gust lasts
		self.gust_timer  = 0 # How long a gust has been going
		
		# Storing these for debugging, but not strictly needed?
		self.gust_angle  = 0 # Which Direction
		self.gust_mag    = 0 # How hard

		# The above 2 will resolve into a single 3 vector
		self.gust_vect = None

	def sample_wind(self):
		self.advance_state()
		if self.gusting:
			return self.gust_vect
		else: # Not gusting
			return np.asarray([0, 0, 0])

	def advance_state(self):
		if self.gusting:
			self.gust_timer += 1
			if self.gust_timer > self.gust_length:
				self.gusting = False
				print("Gust over!")
		else: # Not Gusting
			if self.sample_start() == 1:
				self.gusting = True
				self.gust_length = self.sample_length()
				self.gust_angle  = self.sample_angle()
				self.gust_mag    = self.sample_mag()
				self.gust_vect   = self.resolve_vector()
				print("Gusting!")
				print(self.gust_length, self.gust_mag)

	#### Keeping the sampling split out in case I want to do anything
	#### 'extra' with them before/after sampling. 
	def sample_start(self):
		return np.random.binomial(1, C.START_P)

	def sample_length(self):
		# TODO - Check if this is an ok way to get a 'discrete' normal
		return int(np.random.normal(C.LENGTH_MEAN, C.LENGTH_VAR))

	def sample_angle(self):
		return np.random.normal(C.ANGLE_MEAN, C.ANGLE_VAR)

	def sample_mag(self):
		return np.random.normal(C.MAG_MEAN, C.MAG_VAR)

	def resolve_vector(self):
		x = self.gust_mag * np.cos(self.gust_angle)
		y = self.gust_mag * np.sin(self.gust_angle)
		return np.asarray([x, y, 0.0])
