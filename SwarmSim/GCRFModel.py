import numpy as np
from . import constants as C
from .Regressors.MyMultiOutputRegressor import MyMultiOutputRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree         import DecisionTreeRegressor
from sklearn.multioutput  import MultiOutputRegressor

from scipy.optimize import minimize

class GCRFModel():
	def __init__(self):

		### Weak Learners ###
		self.weak_learners = []
		self.weak_learners.append(MultiOutputRegressor(LinearRegression()))
		#self.weak_learners.append(MultiOutputRegressor(Ridge()))
		#self.weak_learners.append(MultiOutputRegressor(MLPRegressor()))
	
			
	def train(self, S, X, Y):
		
		#print('==== train() ====')
		#print(S.shape)
		#print(X.shape)
		#print(Y.shape)
		
		[N, d_in, T] = X.shape
		[_, d_out, _] = Y.shape
		
		self.N = N
		self.d_in = d_in
		self.T = T
		self.d_out = d_out
		self.K = len(self.weak_learners)
		
		# =========================================================================
		
		# Unfold X and Y by vertically stacking their 
		# slices across all training timesteps
		X_unfolded = flatten_tensor(X)
		Y_unfolded = flatten_tensor(Y)
		
		# Train one (or multiple) weak learners
		K = len(self.weak_learners)
		R_unfolded = np.zeros((Y_unfolded.shape[0],Y_unfolded.shape[1],K), dtype=float)
		for k in range(0,K):
			self.weak_learners[k].fit(X_unfolded, Y_unfolded)
			R_unfolded[:,:,k] = self.weak_learners[k].predict(X_unfolded)
		
		# Fold R_unfolded into a tensor of T slices, 
		# each being of shape N x d_out x K
		R = tensorize_R(R_unfolded, [N, d_out, K, T])
		# =========================================================================
		
		# Temporal GCRF training
		# print('Temporal GCRF training ...')
		# theta <- GCRF_TRAIN(Y,S,R)
		
		S = (S/sum(sum(S))) * N
		L = np.diag(sum(S)) - S
		
		# Initialize params
		alpha = np.ones(K)
		beta = 0
		params = np.append(alpha, beta)

		cons = ({'type': 'ineq', 'fun' : lambda params: sum(params[0:K])},
				{'type': 'ineq', 'fun' : lambda params: params[K]})
		
		res = minimize(GCRF_objective, params, args=(L,R,Y), #jac=GCRF_objective_deriv,
				  	   constraints=cons, method='SLSQP', options={'maxiter': 10000, 'disp': False})
		
		self.theta = res.x
	
	
	def predict(self, X, S, d_index, use_structure):
		
		R = np.zeros((X.shape[0], self.d_out, self.K), dtype=float)
		for k in range(0, self.K):
			R[:,:,k] = self.weak_learners[k].predict(X)
			
		# =========================================================================
			
		# mu <- GCRF_PREDICT(theta, S, R)
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


def flatten_tensor(tensor):
	[N, d, T] = tensor.shape
	tensor_flat = np.zeros((T*N, d), dtype=float)
	for t in range(0,T):
		tensor_flat[t*N:t*N+N, :] = tensor[:, :, t]
	return tensor_flat


def tensorize_R(R_flat, dim):
	[N, d, K, T] = dim
	R = np.zeros((N, d, K, T), dtype=float)
	for t in range(0,T):
		R[:,:,:,t] = R_flat[t*N:t*N+N, :, :]
	return R

		
