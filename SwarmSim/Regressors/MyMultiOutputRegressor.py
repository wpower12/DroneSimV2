from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, ClassifierMixin

class MyEnsembleRegressor(BaseEstimator, ClassifierMixin):
	def __init__(self, M=10, sampling_method=None, eta=None):
		self.M = M
		self.sampling_method = sampling_method
		self.eta = eta
		self.base_learners = []
		
		for m in range(self.M):
			#self.base_learners.append(SGDRegressor(loss='squared_loss', l1_ratio=1, alpha=0.1))
			self.base_learners.append(SGDRegressor(loss='squared_loss', l1_ratio=1, alpha=0.05))
			#self.base_learners.append(MLPRegressor())
	
	def fit(self, X, y):
		self.X_ = X
		self.y_ = y
		
		N = X.shape[0]
		data_indices = range(0,N)
		
		for m in range(self.M):
			
			if self.sampling_method == 'subsampling':
				Nsub = int(np.round(self.eta*N))
				sample_indices = np.random.choice(data_indices, Nsub, replace=False)
				X_base = X[sample_indices, :]
				y_base = y[sample_indices]
				self.base_learners[m].fit(X_base, y_base)
			elif self.sampling_method == 'bootstrap':
				sample_indices = np.random.choice(data_indices, N, replace=True)
				#print(X.shape)
				#print(y.shape)
				X_base = X[sample_indices, :]
				y_base = y[sample_indices]
				self.base_learners[m].fit(X_base, y_base)
			else:
				self.base_learners[m].fit(X,y)
			
		return self
	
	def partial_fit(self, X, y):
		for m in range(self.M):
			self.base_learners[m].partial_fit(X,y)
		return self
	
	def predict(self, X):
		preds = np.zeros((X.shape[0],self.M), dtype=float)
		for m in range(self.M):
			preds[:,m] = self.base_learners[m].predict(X)
		pred_mean = np.mean(preds, axis=1)
		pred_std = np.std(preds, axis=1)
		
		return pred_mean, pred_std

class MyMultiOutputRegressor(BaseEstimator, ClassifierMixin):
	def __init__(self, num_outs):
		self.num_outputs = num_outs
		self.output_regressors = []
		for k in range(self.num_outputs):
			self.output_regressors.append(MyEnsembleRegressor(M=10, sampling_method='subsampling', eta=0.2))
			#self.output_regressors.append(BaggingRegressor(base_estimator=SGDRegressor(), n_estimators=30))

	def fit(self, X, y):
		self.X_ = X
		self.y_ = y
		for k in range(self.num_outputs):
			self.output_regressors[k].fit(X,y[:,k])
		return self
	
	def partial_fit(self, X, y):
		for k in range(self.num_outputs):
			self.output_regressors[k].partial_fit(X,y[:,k])
		return self
	
	def predict(self, X):
		pred_means = np.zeros((X.shape[0],self.num_outputs), dtype=float)
		pred_stds = np.zeros((X.shape[0],self.num_outputs), dtype=float)
		
		for k in range(self.num_outputs):
			curr_mean, curr_std = self.output_regressors[k].predict(X)
			pred_means[:,k] = curr_mean
			pred_stds[:,k] = curr_std
		
		return pred_means, pred_stds	
		

