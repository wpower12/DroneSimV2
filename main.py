from SwarmSim.Sim import *

N = 27
sim = Sim(N, shape="cube")

# Simple waypoint method, set drones target
# as current position plus given vector.
sim.set_swarm_target_relative([20, 60, 0])

for t in range(500):
	sim.tick()
	# sim.dump_state()