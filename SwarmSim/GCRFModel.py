import numpy as np
from . import constants as C
from .Regressors.MyMultiOutputRegressor import MyMultiOutputRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
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
			# self.regs.append(MultiOutputRegressor(LinearRegression()))
			self.regs.append(MultiOutputRegressor(MLPRegressor()))
			#self.regs.append(LinearRegression())

	def train(self, S, X, Y):
		# Construct the L of S
		(N, _) = np.shape(S)
		S = (S/sum(sum(S))) * N
		L = np.diag(sum(S)) - S

		# Reshape the data.
		X          = flatten_shape(np.array(X))
		Y_expanded = np.expand_dims(Y, 1)
		Y_flat     = broadcast_y(flatten_shape(Y_expanded), N)
		Y          = tensorize_Y(Y_expanded)

		
		R = self.fit_weak_learners(X, Y_flat)
		(_,_,D) = np.shape(R)
		R = tensorize_R(R, (N, D, self.K, self.T))
		
		alpha = np.ones((self.K,))
		beta  = np.zeros((1,))
		params = np.append(alpha, beta)

		cons = ({'type': 'ineq', 'fun' : lambda params: sum(params[0:self.K])},
				{'type': 'ineq', 'fun' : lambda params: params[self.K]})

		# print(np.shape(L), np.shape(R), np.shape(Y))
		res = minimize(GCRF_objective, params, args=(L,R,Y), #jac=GCRF_objective_deriv,
				  	   constraints=cons, method='SLSQP', options={'maxiter': 1000, 'disp': False})
		
		self.theta = res.x

	def fit_weak_learners(self, X, Y):
		print(np.shape(X), np.shape(Y))
		R_train = []
		for wl in self.regs:
			wl.fit(X, Y)
			R_train.append(wl.predict(X))
		return np.array(R_train)

	def predict(self, X, S, d_index):
		#return np.zeros((3))
		
		# predict(self, X, S):
		
		# if we are predicting the drones' positions at t,
		# then X should contain the (t-1-w : t - 1) windows for all drones
		
		# Call the predict() method for the weak learners to calculate R
		# ...
		
		X = np.array(X)[-1, :, :]
		
		#R = self.regs[0].predict(X)
		#for k in range(1, len(self.regs)):
		#	R = np.concatenate((R, self.regs[k].predict(X)), axis=0)
		
		R = np.zeros((X.shape[0], X.shape[1], self.K), dtype=float)
		for k in range(0, self.K):
			R[:,:,k] = self.regs[k].predict(X)
		
		
		(N,d_out,K) = R.shape
		S = (S/sum(sum(S))) * N
		L = np.diag(sum(S)) - S

		alpha = self.theta[0:K]
		gamma = sum(alpha)
		beta = self.theta[K]

		Q = beta*L + gamma*np.eye(N)
		mu = np.linalg.solve(Q,np.dot(R,alpha))
		
		#return mu[d_index]
		return R[d_index, :, 0]
		#return R[d_index, :, 1]

def GCRF_objective(params, L, R, Y):
	(N,d_out,K,T) = R.shape
	alpha = params[0:K]
	beta = params[K]
	epsilon = 0.0#1e-8
	
	gamma = sum(alpha)
	Q = beta*L + gamma*np.eye(N)
	#Q_inv = np.linalg.inv(Q)
	Q_inv = np.linalg.pinv(Q) # Optional
	
	neg_ll = 0
	for t in range(0,T):
		for j in range(0,d_out):
			b = np.dot(R[:,j,:,t],alpha.T)
			#mu = np.linalg.solve(Q,b)
			mu = np.dot(Q_inv, b)
			e = Y[:,j,t] - mu
			neg_ll = neg_ll - np.dot(np.dot(e.T, Q), e) - 0.5*np.log(np.linalg.det(Q_inv) + epsilon)
	neg_ll = -neg_ll
	
	return neg_ll

def flatten_shape(X):
	[T, N, d] = np.shape(X)
	flat = np.zeros((T*N, d), dtype=float)
	for t in range(0,T):
		flat[t*N:t*N+N, :] = X[t, :, :]
	return flat

def broadcast_y(Y, N):
	(T, d) = np.shape(Y)
	broad_y = np.zeros((T*N, d), dtype=float)
	for t in range(T):
		for n in range(N):
			broad_y[t+n] = Y[t]
	return broad_y


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


