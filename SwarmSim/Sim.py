import numpy as np

from . import Drone
from . import Wind
from . import Animator
from . import constants as C

class Sim():
	def __init__(self, num_drones, shape="cube"):
		self.N = num_drones
		self.drones = []
		self.wind = Wind.Wind()
		self.anm = Animator.Animator()
		self.training = True
		self.using_expansion = False # For experiments
		self.pred_horz = 0
		self.expansion_timer = 0
		self.expansion_state = C.EXP_OFF

		if shape == "cube":
			# For now, if cube, assume num_drones has a perfect
			# cube root. 
			side_len = int(num_drones ** (1/3))
			for layer_number in range(side_len):
				z_loc = C.SEPARATION * layer_number
				for row in range(side_len):
					x_loc = C.SEPARATION * row
					for col in range(side_len):
						y_loc = C.SEPARATION * col
						d = Drone.Drone()
						d.pos = np.asarray([x_loc, y_loc, z_loc])
						d.target = d.pos
						d.init_PIDs()
						self.drones.append(d)

	def tick(self):
		self.anm.plot_drones(self.drones, self.training, self.using_expansion)

		# All drones see the same 'wind' 
		wind_dev = self.wind.sample_wind() * C.DT

		# 'State Machine' for expansion procedure
		# runs when the expansion flag is set, and when 
		# in inference mode. 
		if self.using_expansion and not self.training:	
			if self.expansion_state == C.EXP_OFF:
				self.expansion_timer += 1
				if self.expansion_timer > self.pred_horz:
					self.exp_hover()
					self.expansion_state = C.EXP_HOVER
					print("Expansion: drones switch to hover mode")
			elif self.expansion_state == C.EXP_HOVER:
				# Check if drones are at targets
				if self.drones_at_targets():
					self.exp_expand()
					self.expansion_state = C.EXP_EXPANDING
					print("Expansion: drones expanding")
			elif self.expansion_state == C.EXP_EXPANDING:
				if self.drones_at_targets():
					self.exp_correct_targets()
					self.expansion_state == C.EXP_OFF
					self.expansion_timer = 0
					print("Expansion: drones update targets")

		for d in self.drones:
			d.pos += wind_dev
			if self.training:
				d.update_training()
			else:
				d.update_inference()

		# For now, not doing this like this.
		# Going to just share a list of 'other' drones 
		if self.training:
			self.distribute_models()

	def use_expansion(self, pred_horz):
		self.using_expansion = True
		self.pred_horz = pred_horz
		self.expansion_timer = pred_horz # So we run immediatly the first time

	def exp_hover(self):
		# Save current target and set target to current pos estimate. 
		for d in self.drones:
			d.saved_target = d.target
			d.set_target(d.pos_estimate)

	def exp_expand(self):
		# Calculate 'center' of the swarm
		poss = []
		for d in self.drones:
			poss.append(d.pos_estimate)
		center = np.mean(poss, axis=0)

		# TODO - Calculate/determine 'max' variance, use
		# to determine the magnitude of expansion vector

		# Move each drone away from center
		for d in self.drones:
			delta = d.pos_estimate - center
			d.exp_vector = (delta / np.linalg.norm(delta))
			d.exp_vector *= C.TEST_VAR_RADIUS
			d.set_target(d.pos_estimate+d.exp_vector)

	def exp_correct_targets(self):
		for d in self.drones:
			d.set_target(d.saved_target+d.exp_vector)

	def drones_at_targets(self):
		have_reached = True
		for d in self.drones:
			have_reached = have_reached and d.has_reached_target(C.TARGET_EPSILON)
		return have_reached

	def distribute_models(self):
		# Share the models between all drones. 2 Passes?
		# Collect all models
		models = []
		for d in self.drones:
			# TODO - How do we encode them? Just copy a reference?
			pass

		# Share all models
		for d in self.drones:
			# d.models = models # Is it that simple?
			pass

	def set_swarm_target_relative(self, dpos):
		delta = np.asarray(dpos)
		for d in self.drones:
			d.target = d.pos + delta
			d.init_PIDs() 

	def dump_state(self):
		for d in self.drones:
			print(d.pos)