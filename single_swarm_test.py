from SwarmSim.SingleSwarmSim import *

NUM_TRAINING_STEPS  = 100
NUM_INFERENCE_STEPS = 400
PREDICTION_HORZ = 20
N = 27

swarm_options = [N, "cube", "g", [0,0,0], [20,20,20]]
sim = SingleSwarmSim(swarm_options)

for t in range(NUM_TRAINING_STEPS):
	sim.animate()
	sim.tick()

# sim.in_training = False
# sim.use_expansion(PREDICTION_HORZ) 

# for t in range(NUM_INFERENCE_STEPS):
# 	sim.tick()