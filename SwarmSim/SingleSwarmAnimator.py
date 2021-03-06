import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class SingleSwarmAnimator():
	def __init__(self, fig_title):
		plt.ion()
		self.fig = plt.figure(num=fig_title)
		self.ax  = self.fig.add_subplot(111, projection='3d')
		self.xlim = [0,100]
		self.ylim = [0,100]
		self.zlim = [0,7]
		self.color_str = ""

	def plot_drones(self, sim, in_training=True, plot_errors=False):
		plt.cla()
		self.color_str = sim.color + "."
		for d in sim.drones:
			self.plot_drone(d, in_training)

		# Hardcoding for now, should fix later.
		# read the limits from the waypoints? idk.
		# plt.xlim(self.xlim[0], self.xlim[1])
		# plt.ylim(self.ylim[0], self.ylim[1])
		# self.ax.set_zlim(self.zlim[0], self.zlim[1])
		
		plt.pause(0.001)

	def plot_drone(self, d, in_training=True):
		x, y, z = d.pos
		
		self.ax.plot([x], [y], [z], self.color_str)

		if len(d.H_pos) > 0:
			s_hist = np.vstack(d.H_pos)
			self.ax.plot(s_hist[:,0], s_hist[:,1], s_hist[:,2], 'k:')

		if not in_training: # If in inference:
			x, y, z = d.pos_estimate
			self.ax.plot([x], [y], [z], 'r.')

			if len(d.H_pos_est) > 0:
				s_hist = np.vstack(d.H_pos_est)
				self.ax.plot(s_hist[:,0], s_hist[:,1], s_hist[:,2], 'r:')




