import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class Animator():
	def __init__(self):
		plt.ion()
		self.fig = plt.figure()
		self.ax  = self.fig.add_subplot(111, projection='3d')
		self.xlim = [-5,5]
		self.ylim = [-5,5]
		self.zlim = [ 0,10]

	def plot_drones(self, drones):
		plt.cla()
		for d in drones:
			self.plot_drone(d)

		# Hardcoding for now, should fix later.
		# read the limits from the waypoints? idk.
		plt.xlim(self.xlim[0], self.xlim[1])
		plt.ylim(self.ylim[0], self.ylim[1])
		self.ax.set_zlim(self.zlim[0], self.zlim[1])
		plt.pause(0.001)

	def plot_drone(self, d):
		x, y, z = d.pos
		self.ax.plot([x], [y], [z], 'k.')