from SwarmSim.MultiSwarmSim import *

NUM_TRAINING_STEPS  = 100
NUM_INFERENCE_STEPS = 400
PREDICTION_HORZ = 20

swarms = [[9, "planar", [0,0,0], [20,20,20]],
          [27, "cube",  [20,20,20], [-20,-20,-20]]]

m_sim = MultiSwarmSim(swarms)

for t in range(NUM_TRAINING_STEPS):
	m_sim.animate()
	m_sim.tick()