from SwarmSim.Sim import *

NUM_TRAINING_STEPS  = 20
NUM_INFERENCE_STEPS = 200

N = 27
sim = Sim(N, shape="cube")

# Simple waypoint method, set drones target
# as current position plus given vector.
sim.set_swarm_target_relative([20, 60, 0])

for t in range(NUM_TRAINING_STEPS):
	sim.tick()

sim.training = False

for t in range(NUM_INFERENCE_STEPS):
	sim.tick()