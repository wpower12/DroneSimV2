from SwarmSim.SingleSwarmSim import *
from SwarmSim.ExpHelper      import *
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

LOG_DIR = "exp_run_results/" 
LOG_FN  = "full_run_01.txt"
log_f   = open(LOG_DIR+LOG_FN, 'w')

NUM_RUNS   = 2
NUM_TRAINING_STEPS  = 50
NUM_INFERENCE_STEPS = 50

SWARM_SIZE = 9
SWARM_TYPE = "planar"
START      = [0,0,0]
END        = [5,5,5] #[100,100,100]
ANIMATE    = True

swarm_options = [SWARM_SIZE, SWARM_TYPE, "b", START, END]

log_f.write("n: {}, type: {}, runs: {}, n_train: {}, n_inf: {}, path: {} to {}\n".format(SWARM_SIZE,
																						 SWARM_TYPE,
																						 NUM_RUNS,
																						 NUM_TRAINING_STEPS,
																						 NUM_INFERENCE_STEPS,
																						 START,
																						 END))
log_f.write("run, seed, model error, dr error, diff\n")

error_total   = 0
for n in range(NUM_RUNS):
	print('run:', n)
	
	# rnd_seed = 0 # when set to 0 we don't get the LINALG exception
	#rnd_seed = random.randint(0, 10000000)
	rnd_seed = 1028791
	print('rnd_seed:', rnd_seed)
	print('\n\n')
	
	# =========================================================================
	
	# baseline
	np.random.seed = rnd_seed
	sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, None, 'Ground truth', ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS+NUM_INFERENCE_STEPS):
		sim.tick()

	print('Ground truth trajectories: generated!')
	print('\n\n')
		
	# =========================================================================
	
	# without Model
	np.random.seed = rnd_seed
	sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, None, ' Dead reckoning', ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS):
		sim.tick()
	
	# starting inference with FALSE means we DONT use the model
	# only dead reckoning
	sim.start_inference(False)
	dr_error = np.zeros((3,), dtype=float)
	for i in range(NUM_INFERENCE_STEPS):
		sim.tick()
		
		target_locations = sim.dump_drone_locations(True, None)
		
		dr_locations = sim.dump_drone_locations(False, 'inference')
		dr_error += calculate_error(target_locations, dr_locations)
	dr_error /= NUM_INFERENCE_STEPS
	
	print('DR:', dr_error)
	print('\n\n')
	
	# =========================================================================
	
	# with Model
	np.random.seed = rnd_seed
	sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, False, 'Unstructured regressor', ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS):
		sim.tick()
	
	# starting inference with TRUE means we use the model
	sim.start_inference(True)
	model_error = np.zeros((3,), dtype=float)
	for i in range(NUM_INFERENCE_STEPS):
		sim.tick()
		
		target_locations = sim.dump_drone_locations(True, None)
		
		model_locations = sim.dump_drone_locations(False, 'inference')
		model_error += calculate_error(target_locations, model_locations)
	model_error /= NUM_INFERENCE_STEPS

	print('Unstructured model:', model_error)
	print('\n\n')
	
	# =========================================================================
	
	# with Model
	np.random.seed = rnd_seed
	sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, True, 'Temporal GCRF', ANIMATE)
	sim.set_seed(rnd_seed)
	for i in range(NUM_TRAINING_STEPS):
		sim.tick()
	
	# starting inference with TRUE means we use the model
	sim.start_inference(True)
	model_error = np.zeros((3,), dtype=float)
	for i in range(NUM_INFERENCE_STEPS):
		sim.tick()
		
		target_locations = sim.dump_drone_locations(True, None)
		
		model_locations = sim.dump_drone_locations(False, 'inference')
		model_error += calculate_error(target_locations, model_locations)
	model_error /= NUM_INFERENCE_STEPS

	print('GCRF:', model_error)
	print('\n\n')
	
	
	#log_f.write("{}, {}, {}, {}, {}\n".format(n, rnd_seed, model_error, dr_error, dr_error-model_error))
	
	#error_total += (dr_error-model_error)
	
	break

log_f.write("total error: {}\n".format(error_total))
log_f.write("ave error: {}".format(error_total/SWARM_SIZE))

log_f.close()
print('Done!')
