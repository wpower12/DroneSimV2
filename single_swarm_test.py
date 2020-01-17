from SwarmSim.SingleSwarmSim import *

NUM_TRAINING_STEPS  = 100
NUM_INFERENCE_STEPS = 400
PREDICTION_HORZ = 20
ANIMATE = True

swarm_options = [27, "cube", "g", [0,0,0], [20,20,20]]

sim = SingleSwarmSim(swarm_options, ANIMATE)

for t in range(NUM_TRAINING_STEPS):
	sim.tick()

sim.start_inference(PREDICTION_HORZ)

for t in range(NUM_INFERENCE_STEPS):
	sim.tick()