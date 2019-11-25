import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Animator():
	def __init__(self):
		plt.ion()
		self.fig = plt.figure()
		self.ax  = self.fig.add_subplot(111, projection='3d')
		self.xlim = [0,100]
		self.ylim = [0,100]
		self.zlim = [0,7]

	def plot_drones(self, drones, in_training=True):
		plt.cla()
		for d in drones:
			self.plot_drone(d, in_training)

		# Hardcoding for now, should fix later.
		# read the limits from the waypoints? idk.
		plt.xlim(self.xlim[0], self.xlim[1])
		plt.ylim(self.ylim[0], self.ylim[1])
		self.ax.set_zlim(self.zlim[0], self.zlim[1])
		plt.pause(0.001)

	def plot_drone(self, d, in_training=True):
		if in_training:
			# Draw drone as sphere at d.pos
			# Draw history in H_pos as TODO_PICK_COLOR
			x, y, z = d.pos
			self.ax.plot([x], [y], [z], 'k.')

			if len(d.H_pos) > 0:
				s_hist = np.vstack(d.H_pos)
				self.ax.plot(s_hist[:,0], s_hist[:,1], s_hist[:,2], 'k:')
	
		else: # If in inference:
			# Draw drone at d.pos
			# Draw smaller dot at d.pos_est
			# Draw history of ACTUAL    pos from d.H_pos
			# Draw history of ESTIMATED pos from d.H_pos_est
			x, y, z = d.pos
			self.ax.plot([x], [y], [z], 'k.')
			
			x, y, z = d.pos_estimate
			self.ax.plot([x], [y], [z], 'r.')

			if len(d.H_pos) > 0:
				s_hist = np.vstack(d.H_pos)
				self.ax.plot(s_hist[:,0], s_hist[:,1], s_hist[:,2], 'k:')
			if len(d.H_pos_est) > 0:
				s_hist = np.vstack(d.H_pos_est)
				self.ax.plot(s_hist[:,0], s_hist[:,1], s_hist[:,2], 'r:')


