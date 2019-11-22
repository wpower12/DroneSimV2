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
		self.anm.plot_drones(self.drones)
		# All drones see the same 'wind' naive assumption
		wind_dev = self.wind.sample_wind() * C.DT
		for d in self.drones:
			d.pos += wind_dev
			d.update()


	def dump_state(self):
		for d in self.drones:
			print(d.pos)