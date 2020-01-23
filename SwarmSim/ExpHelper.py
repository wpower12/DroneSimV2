import numpy as np

def calculate_error(target, observed):
	'''
	n, d = np.shape(target)
	diff = target-observed
	ret = 0
	for i in range(n):
		ret += np.linalg.norm(diff[i])
	return ret
	'''
	return np.mean( np.power(np.array(target)-np.array(observed), 2) )