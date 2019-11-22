import numpy as np
from simple_pid import PID
from . import constants as C

class Drone():
	def __init__(self):
		self.maxv = C.MAX_VEL
		self.maxa = C.MAX_ACC

		self.m    = 1.0 # Unit Mass

		# State Vectors 
		self.pos = np.zeros((3))
		self.vel = np.zeros((3))
		self.acc = np.zeros((3))

		# Target Vector - Tracking Next Target Position
		self.target = np.zeros((3))

		# Controllers
		self.PID_X = None
		self.PID_Y = None
		self.PID_Z = None

		# History Lists
		self.H_pos = []
		self.H_vel = []
		self.H_acc = []

		# Models/External State
		# self.others_pos    = [] # last known pos of other drones
		# self.others_models = [] # model parameters for other drones

		# self.model = None # The ML model we are using?

	def init_PIDs(self):
		# Assuming we call this everytime we update the targets
		x, y, z = self.target
		self.PID_X = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x)
		self.PID_Y = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y)
		self.PID_Z = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z)

	def update(self):
		dAcc_x = self.PID_X(self.pos[0])
		dAcc_y = self.PID_Y(self.pos[1])
		dAcc_z = self.PID_Z(self.pos[2])
		# print(dAcc_x, dAcc_y, dAcc_z)

		# Update acc's by clamp adding the above to them.
		n_acc_x = clamp_add(self.acc[0], dAcc_x, C.MAX_ACC)
		n_acc_y = clamp_add(self.acc[1], dAcc_y, C.MAX_ACC)
		n_acc_z = clamp_add(self.acc[2], dAcc_z, C.MAX_ACC)
		self.acc = np.asarray([n_acc_x, n_acc_y, n_acc_z])

		# Update vel's by clamp adding the acc's to them.
		# Note we are adding acc*DT to the vel
		n_vel_x = clamp_add(self.vel[0], n_acc_x*C.DT, C.MAX_VEL)
		n_vel_y = clamp_add(self.vel[1], n_acc_y*C.DT, C.MAX_VEL)
		n_vel_z = clamp_add(self.vel[2], n_acc_z*C.DT, C.MAX_VEL)
		self.vel = np.asarray([n_vel_x, n_vel_y, n_vel_z])

		# Update position.
		self.pos += self.vel

# Returns a value that is max(abs(max_a), abs(a+b))
def clamp_add(a, b, max_a):
	if (a + b) > max_a:
		return max_a
	if (a + b) < -1.0*max_a:
		return -1.0*max_a
	return a + b