from SwarmSim.SingleSwarmSim import *

NUM_TRAINING_STEPS  = 1000
ANIMATE = True

swarm_options = [27, "cube", "g", [0,0,0], [50,50,50]]

sim = SingleSwarmSim(swarm_options, ANIMATE)

for t in range(NUM_TRAINING_STEPS):
	sim.tick()
	print(t)