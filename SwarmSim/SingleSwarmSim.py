from . import Swarm as S
from . import Wind  as W
from . import SingleSwarmAnimator as A

class SingleSwarmSim():
	def __init__(self, swarm):
		num_drones, swarm_type, color, pos, target = swarm
		self.sim = S.Swarm(num_drones, swarm_type)
		self.sim.color = color
		self.sim.set_swarm_pos_relative(pos)
		self.sim.set_swarm_target_relative(target)
		self.sim.init_drone_PIDs()
		self.wind = W.Wind()
		self.anm = A.SingleSwarmAnimator()
		self.in_training = True
		self.plot_errors = False
		
	def animate(self):
		self.anm.plot_drones(self.sim, 
							self.sim.training, 
							self.plot_errors)

	def tick(self):
		self.sim.tick(self.wind)

	def start_inference(self, ph):
		self.sim.training = False
		self.sim.use_expansion(ph)
