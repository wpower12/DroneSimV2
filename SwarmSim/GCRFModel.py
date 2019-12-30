import numpy as np
from . import constants as C
from .Regressors.MyMultiOutputRegressor import MyMultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.multioutput  import MultiOutputRegressor

from scipy.optimize import minimize

class GCRFModel():
	def __init__(self, K, T):
		self.K = K
		self.T = T

		### Weak Learners ###
		self.regs = []
		for i in range(K):
			# self.regs.append(MultiOutputRegressor(LinearRegression))
			self.regs.append(LinearRegression())

		alpha = np.ones((self.K,))
		beta  = 0
		self.theta = np.array([alpha, beta])

	def train(self, S, X, Y):
		X      = flatten_shape(np.array(X))
		Y_flat = flatten_shape(np.array(Y))
		Y      = tensorize_Y(np.array(Y))

		(N, _) = np.shape(S)
		S = (S/sum(sum(S))) * N
		L = np.diag(sum(S)) - S
		
		R = self.fit_weak_learners(X, Y_flat)
		(_,_,D) = np.shape(R)
		R = tensorize_R(R, (N, D, self.K, self.T))

		cons = ({'type': 'ineq', 'fun' : lambda theta: sum(theta[0:K])},
				{'type': 'ineq', 'fun' : lambda theta: theta[K]})

		theta = self.theta

		print(np.shape(L), np.shape(R), np.shape(Y))
		res = minimize(GCRF_objective, theta, args=(L,R,Y), #jac=GCRF_objective_deriv,
				  	   constraints=cons, method='SLSQP', options={'maxiter': 1000, 'disp': False})
		theta = res.x
		return theta

	def fit_weak_learners(self, X, Y):
		R_train = []
		for wl in self.regs:
			wl.fit(X, Y)
			R_train.append(wl.predict(X))
		return np.array(R_train)

	def predict(self):
		return np.zeros((3))

def GCRF_objective( theta, L, R, Y ):
	(N,D,K,T) = R.shape
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

def flatten_shape(X):
	[T, N, d] = np.shape(X)
	flat = np.zeros((T*N, d), dtype=float)
	for t in range(0,T):
		flat[t*N:t*N+N, :] = X[t, :, :]
	return flat

def tensorize_Y(Y_old):
	# Just reshaping Y - it is in shape (T, N, D)
	# Need it to be (N, D, T)
	(T, N, D) = np.shape(Y_old)
	Y = np.zeros((N, D, T))
	for n in range(N):
		for d in range(D):
			for t in range(T):
				Y[n][d][t] = Y_old[t][n][d]
	return Y


def tensorize_R(R_flat, dim):
	# R_flat    is shape (K, N*T, D)
	# Want R to be shape (N, D, K, T)?
	[N, D, K, T] = dim
	R = np.zeros((N, D, K, T), dtype=float)
	for d in range(D):
		for k in range(K):
			for n in range(N):
				for t in range(T):
					R[n][d][k][t] = R_flat[k][n*t][d]
	return R


def threshold(S):
	(N, _) = np.shape(S)
	S_thresh = np.zeros((N, N))
	for i in range(N):
		for j in range(N):
			if S[i][j] < (C.SEPARATION - 0.1):
				S_thresh[i][j] = 1
	return S_thresh


