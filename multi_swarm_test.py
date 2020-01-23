from SwarmSim.MultiSwarmSim import *

NUM_TRAINING_STEPS  = 400
NUM_INFERENCE_STEPS = 400
PREDICTION_HORZ = 20
ANIMATE = True

swarms = [[9, "planar", "b", [0,0,0],    [20,20,20]],
          [27, "cube",  "r", [20,20,20], [-20,-20,-20]]]
          # [65, "cube",  "g", [-20,20,20], [-20,-20,-100]]]

rnd_seed = 0

m_sim = MultiSwarmSim(swarms, rnd_seed, ANIMATE)

for t in range(NUM_TRAINING_STEPS):
	m_sim.animate()
	m_sim.tick()

# m_sim.start_inference(TRUE)
# m_sim.use_expansion(PREDICTION_HORZ)

# for t in range(NUM_INFERENCE_STEPS):
# 	m_sim.animate()
# 	m_sim.tick()