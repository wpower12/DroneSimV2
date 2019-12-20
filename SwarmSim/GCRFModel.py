import numpy as np
from . import constants as C
from .Regressors.MyMultiOutputRegressor import MyMultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.multioutput  import MultiOutputRegressor

from scipy.optimize import minimize

class GCRFModel():
	def __init__(self):

		### Regressors ###
		self.regs = []
		self.regs.append(MultiOutputRegressor(LinearRegression))
		# self.regs.append(MyMultiOutputRegressor(LinearRegression)) # Broken. 

		self.K = len(self.regs)
		alpha = np.ones((self.K,))
		beta  = 0
		self.theta = np.array([alpha, beta])

	def train(self, S, X, Y):
		T, N, F = np.shape(X) # (Timesteps, Num_Nodes, Num_Features)	

		R = []
		# For r in Regressors
			# Fit X and Y on r for all T
			# Get output of R(X) after fitting
			# Append to R	
		for reg in self.regs:
			# reg.fit(X, Y)
			# R.append(reg.predict(X))
			pass

		# Should have a [N, K, Num_Features] slice for each time step in R

		# If S is distances right now, then we need to convert it to an adjacency matrix
		# We threshold. 
		S = threshold(S)
		S = (S / np.sum(np.sum(S)))*N

		# Now we need to pass the R, L, and Y values to an optimizer
		# that will find the set of A, b that yields the minimum theta.

	def predict(self):
		return np.zeros((3))

def GCRF_obj(theta, L, R, Y):
	# theta - [a0, a1, ... an, beta]
	# L - lapacian of S matrix
	# R - 'data' shape -> [Nodes, Features, Timesteps]
	# Y - target - shape -> [Nodes, Features]
	N, K, T = np.shape(R)
	alpha   = theta[:K]
	beta    = theta[K]

# TODO
def threshold(S):
	return S


