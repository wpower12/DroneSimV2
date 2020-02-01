import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_error(target, observed):
	'''
	n, d = np.shape(target)
	diff = target-observed
	ret = 0
	for i in range(n):
		ret += np.linalg.norm(diff[i])
	return ret
	'''
	return np.array([ mean_squared_error(target[:,0], observed[:,0]),
			 		  mean_squared_error(target[:,1], observed[:,1]),
			 		  mean_squared_error(target[:,2], observed[:,2]) ])