import numpy as np
from simple_pid import PID
from . import constants as C
  
class Drone():
	def __init__(self):
		# State Vectors
		self.pos = np.zeros((3)) # ACTUAL Location
		self.vel = np.zeros((3)) # Drone Velocity
		self.acc = np.zeros((3)) # Drone Acceleration
		self.pos_estimate = np.zeros((3)) # Estimated via deadreckoning
		self.exp_vector   = np.zeros((3)) # For the expansion procedure

		# Target Vector - Tracking Next Target Position
		self.target       = np.zeros((3)) # Actual position target used for setpoints
		self.saved_target = np.zeros((3)) # Planned target - next waypoint. 'Goal'

		# Controllers
		self.PID_X = None
		self.PID_Y = None
		self.PID_Z = None

		# History Lists
		self.H_pos_est = []
		self.H_pos = []
		self.H_vel = []
		self.H_acc = []

		# Models/External State
		# self.others_pos    = [] # last known pos of other drones
		# self.others_models = [] # model parameters for other drones

	def set_target(self, t):
		self.target = np.copy(t)
		self.init_PIDs()

	def init_PIDs(self):
		# Assuming we call this everytime we update the targets
		x, y, z = self.target
		self.PID_X = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x)
		self.PID_Y = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y)
		self.PID_Z = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z)

	def update_state_from_pos(self, pos):
		# Update output limits manually so that we wouldn't clamp the last integral and the last output value
		self.PID_X.output_limits = (-1.0 * C.MAX_ACC - self.acc[0], C.MAX_ACC - self.acc[0])
		self.PID_Y.output_limits = (-1.0 * C.MAX_ACC - self.acc[1], C.MAX_ACC - self.acc[1])
		self.PID_Z.output_limits = (-1.0 * C.MAX_ACC - self.acc[2], C.MAX_ACC - self.acc[2])

		# Changes in acc are the outputs of the PID controllers
		dAcc_x = self.PID_X(pos[0], dt=C.DT)
		dAcc_y = self.PID_Y(pos[1], dt=C.DT)
		dAcc_z = self.PID_Z(pos[2], dt=C.DT)

		# Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
		self.acc = np.asarray([self.acc[0] + dAcc_x, self.acc[1] + dAcc_y, self.acc[2] + dAcc_z])

		# Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
		n_vel_x = clamp_add(self.vel[0], self.acc[0] * C.DT, C.MAX_VEL)
		n_vel_y = clamp_add(self.vel[1], self.acc[1] * C.DT, C.MAX_VEL)
		n_vel_z = clamp_add(self.vel[2], self.acc[2] * C.DT, C.MAX_VEL)
		self.vel = np.asarray([n_vel_x, n_vel_y, n_vel_z])

	def update_training(self):
		# In training, we use the 'real' position
		self.update_state_from_pos(self.pos)

		# Update position.
		self.pos += self.vel*C.DT
		self.H_pos.append(np.copy(self.pos))

		# Save that in the estimated position
		self.pos_estimate = np.copy(self.pos)

		self.model_update()

	def update_inference(self, model, use_model, X, S, d_index, use_structure):
		if use_model:
			# Apply output of model to predict deviation from Wind
			# self.pos_estimate += model.predict(X, S, d_index) # I think? more on this later.
			self.pos_estimate = model.predict(X, S, d_index, use_structure) # I think? more on this later.

		# In inference, we use the ESTIMATE of pos to update
		self.update_state_from_pos(self.pos_estimate)

		# Finally, update current estimate of position
		self.H_pos_est.append(np.copy(self.pos_estimate))
		self.pos_estimate += self.vel*C.DT

		# We also update the 'real' position, because the real position
		# has already been moved by the wind, the effect of the PID's 
		# changes to the acceleration are still impacting the true
		# location ONTOP of the wind moving it. 
		self.H_pos.append(np.copy(self.pos))
		self.pos += self.vel*C.DT

	def model_update(self):
		# TODO
		pass

	def model_predict(self):
		# TODO 
		return np.zeros((3))

	def model_variance(self):
		# TODO
		return 1.0

	def has_reached_target(self, epsilon):
		# return true if distance to target is within epsilon
		return abs(np.linalg.norm(self.pos_estimate-self.target)) < epsilon

# Returns a value that is max(abs(max_a), abs(a+b))
def clamp_add(a, b, max_a):
	if (a + b) > max_a:
		return max_a
	if (a + b) < -1.0*max_a:
		return -1.0*max_a
	return a + b