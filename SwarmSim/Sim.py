import numpy as np

from . import Drone
from . import Wind
from . import Animator
from . import constants as C

class Sim():
	def __init__(self, num_drones, shape="cube"):
		self.N = num_drones
		self.drones = []
		self.wind = Wind.Wind()
		self.anm = Animator.Animator()
		self.training = True;

		if shape == "cube":
			# For now, if cube, assume num_drones has a perfect
			# cube root. 
			side_len = int(num_drones ** (1/3))
			for layer_number in range(side_len):
				z_loc = C.SEPARATION * layer_number
				for row in range(side_len):
					x_loc = C.SEPARATION * row
					for col in range(side_len):
						y_loc = C.SEPARATION * col
						d = Drone.Drone()
						d.pos = np.asarray([x_loc, y_loc, z_loc])
						d.target = d.pos
						d.init_PIDs()
						self.drones.append(d)

	def tick(self):
		self.anm.plot_drones(self.drones, self.training)

		# All drones see the same 'wind' 
		wind_dev = self.wind.sample_wind() * C.DT
		d.pos += wind_dev

		for d in self.drones:
			if self.training:
				d.update_training()
			else:
				d.update_inference()

		if self.training:
			self.distribute_models()

	def distribute_models(self):
		# Share the models between all drones. 2 Passes?
		# Collect all models
		models = []
		for d in self.drones:
			# TODO - How do we encode them? Just copy a reference?
			pass

		# Share all models
		for d in self.drones:
			# d.models = models # Is it that simple?
			pass

	def set_swarm_target_relative(self, dpos):
		delta = np.asarray(dpos)
		for d in self.drones:
			d.target = d.pos + delta
			d.init_PIDs() 

	def dump_state(self):
		for d in self.drones:
			print(d.pos)