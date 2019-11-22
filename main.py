from SwarmSim.Sim import *

N = 27
sim = Sim(N, shape="cube")


for t in range(100):
	sim.tick()
	# sim.dump_state()