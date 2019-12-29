import numpy as np
from . import constants as C
from .Regressors.MyMultiOutputRegressor import MyMultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.multioutput  import MultiOutputRegressor

from scipy.optimize import minimize

class GCRFModel():
	def __init__(self, K):
		self.K = K
		### Weak Learners ###
		self.regs = []
		for i in range(K):
			self.regs.append(MultiOutputRegressor(LinearRegression))
			# self.regs.append(LinearRegression())

		alpha = np.ones((self.K,))
		beta  = 0
		self.theta = np.array([alpha, beta])

	def train(self, S, X, Y):
		print(np.shape(X), np.shape(Y))
		R = self.fit_weak_learners(Y, X)

		(N,K,_) = R.shape
		S = (S/sum(sum(S))) * N
		L = np.diag(sum(S)) - S
		
		cons = ({'type': 'ineq', 'fun' : lambda theta: sum(theta[0:K])},
				{'type': 'ineq', 'fun' : lambda theta: theta[K]})

		# Keep 'updating' theta? 
		theta = self.theta
		res = minimize(GCRF_objective, theta, args=(L,R,Y), #jac=GCRF_objective_deriv,
				  	   constraints=cons, method='SLSQP', options={'maxiter': 1000, 'disp': False})
		theta = res.x
		return theta

	def fit_weak_learners(self, Y, X):
		R_train = []
		for wl in self.regs:
			wl.fit(X, Y)
			R_train.append(wl.predict(X))

		# Should return a (N, K, T) tensor
		# Currently a (K, N, T, F) tensor
		return np.array(R_train)

	def predict(self):
		return np.zeros((3))

def GCRF_objective( theta, L, R, Y ):
	(N,K,T) = R.shape
	alpha = theta[0:K]
	beta = theta[K]
	epsilon = 1e-6
	
	gamma = sum(alpha)
	Q = beta*L + gamma*np.eye(N)
	Q_inv = np.linalg.inv(Q)
	
	neg_ll = 0
	for t in range(0,T):
		b = np.dot(R[:,:,t],alpha.T)
		mu = np.linalg.solve(Q,b)
		e = Y[:,t] - mu
		neg_ll = neg_ll - np.dot(np.dot(e.T, Q), e) - 0.5*np.log(np.linalg.det(Q_inv) + epsilon)
	neg_ll = -neg_ll
	
	return neg_ll

# TODO
def threshold(S):
	return S


