from . import Swarm as S
from . import Wind  as W
from . import SingleSwarmAnimator as A

class SingleSwarmSim():
	def __init__(self, swarm, animate=True):
		num_drones, swarm_type, color, pos, target = swarm
		self.sim = S.Swarm(num_drones, swarm_type)
		self.sim.color = color
		self.sim.set_swarm_pos_relative(pos)
		self.sim.set_swarm_target_relative(target)
		self.sim.init_drone_PIDs()
		self.wind = W.Wind()
		self.in_training = True
		self.plot_errors = False
		self.animating = animate
		if self.animating:
			self.anm = A.SingleSwarmAnimator()
		
	def animate(self):
		self.anm.plot_drones(self.sim, 
							self.sim.training, 
							self.plot_errors)

	def set_seed(self, n):
		self.wind.set_seed(n)

	def tick(self):
		if self.animating:
			self.animate()
		self.sim.tick(self.wind)

	def start_inference(self, ph):
		self.sim.training = False
		self.sim.use_expansion(ph)

	def dump_drone_locations(self):
		return self.sim.dump_locations()
