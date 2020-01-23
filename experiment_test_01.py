from SwarmSim.SingleSwarmSim import *
from SwarmSim.ExpHelper      import *

import random

LOG_DIR = "exp_run_results/" 
LOG_FN  = "test_01.txt"
log_f   = open(LOG_DIR+LOG_FN, 'w')

NUM_RUNS   = 2
NUM_TRAINING_STEPS  = 10
NUM_INFERENCE_STEPS = 10
PREDICTION_HORZ     = 20

SWARM_SIZE = 9
SWARM_TYPE = "planar"
START      = [0,0,0]
END        = [50,50,50]
ANIMATE    = False

swarm_options = [SWARM_SIZE, SWARM_TYPE, "b", START, END]
error_total   = 0

log_f.write("run, seed, model error, dr error, diff\n")

for n in range(NUM_RUNS):
	# rnd_seed = 0 # when set to 0 we don't get the LINALG exception
	rnd_seed = random.randint(-10000000, 10000000)

	# baseline
	sim = SingleSwarmSim(swarm_options, ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_INFERENCE_STEPS+NUM_TRAINING_STEPS):
		sim.tick()
	target_locations = sim.dump_drone_locations()

	# with Model
	sim = SingleSwarmSim(swarm_options, ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS):
		sim.tick()
	# starting inference with TRUE means we use the model
	sim.start_inference(True)
	for i in range(NUM_INFERENCE_STEPS):
		sim.tick()
	model_locations = sim.dump_drone_locations()
	model_error = calculate_error(target_locations, model_locations)

	# without Model
	sim = SingleSwarmSim(swarm_options, ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS):
		sim.tick()
	# starting inference with FALSE means we DONT use the model
	sim.start_inference(False)
	for i in range(NUM_INFERENCE_STEPS):
		sim.tick()
	dr_locations = sim.dump_drone_locations()
	dr_error = calculate_error(target_locations, model_locations)

	log_f.write("{}, {}, {}, {}, {}\n".format(n, rnd_seed, model_error, dr_error, dr_error-model_error))

	error_total += (dr_error-model_error)

log_f.write("total error: {}\n".format(error_total))
log_f.write("ave error: {}".format(error_total/SWARM_SIZE))

log_f.close()
