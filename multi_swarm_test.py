from SwarmSim.MultiSwarmSim import *

NUM_TRAINING_STEPS  = 100
NUM_INFERENCE_STEPS = 400
PREDICTION_HORZ = 20

swarms = [[9, "planar", "b", [0,0,0],    [20,20,20]],
          [27, "cube",  "r", [20,20,20], [-20,-20,-20]],
          [65, "cube",  "g", [-20,20,20], [-20,-20,-100]]]

m_sim = MultiSwarmSim(swarms)

for t in range(NUM_TRAINING_STEPS):
	m_sim.animate()
	m_sim.tick()