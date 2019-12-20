import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class MultiSwarmAnimator():
	def __init__(self):	
		plt.ion()
		self.fig = plt.figure()
		self.ax  = self.fig.add_subplot(111, projection='3d')
		self.xlim = [0,100]
		self.ylim = [0,100]
		self.zlim = [0,7]
		self.color_str = ""

	def plot_swarms(self, swarms, in_training=True, plot_errors=False):
		plt.cla()
		for s in swarms:
			self.color_str = s.color + "."
			for d in s.drones:
				self.plot_drone(d, s.training)
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



