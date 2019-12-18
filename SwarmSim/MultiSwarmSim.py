from . import Swarm as S
from . import Wind  as W
from . import MultiSwarmAnimator as A

class MultiSwarmSim():
	def __init__(self, swarms):
		# Assuming swarms holds a list of swarms:
		# [[num_drones, type, plot_color, inital_position, target],..,[...]]
		self.swarms = []
		for s in swarms:
			num_drones, swarm_type, color, pos, target = s
			new_sim = S.Swarm(num_drones, swarm_type)
			new_sim.color = color
			new_sim.set_swarm_pos_relative(pos)
			new_sim.set_swarm_target_relative(target)
			new_sim.init_drone_PIDs()
			self.swarms.append(new_sim)

		self.wind = W.Wind()
		self.anm  = A.MultiSwarmAnimator()

	def animate(self):
		self.anm.plot_swarms(self.swarms)

	def tick(self):
		for s in self.swarms:
			s.tick(self.wind)

