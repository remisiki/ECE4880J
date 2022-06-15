import numpy as np

def find_local_maxima(H, neighbor_size=5):
	"""
	This is a predefined function tool for you to use to find the local maxinum point in a matrix
	Inputs:
	- H: An 2-D numpy array of shape (H,W)

	Returns:
	- peaks: A 2-D numpy array of shape (H,W), where peaks have original value in H, otherwise it is 0

	"""
	H_copy = np.copy(H)
	ssize = int((neighbor_size-1)/2)
	peaks = np.zeros(H_copy.shape)
	h, w = H_copy.shape
	for y in range(ssize, h-ssize):
		for x in range(ssize, w-ssize):
			val = H_copy[y, x]
			if val > 0:
				neighborhood = np.copy(H_copy[y-ssize:y+ssize+1, x-ssize:x+ssize+1])
				neighborhood[ssize, ssize] = 0
				if val > np.max(neighborhood):
					peaks[y, x] = val
	return peaks

def Handwrite_HoughLines(Im, num_lines):
	neighbor_size = 5
	peaks = find_local_maxima(Im, neighbor_size)
	# YOUR CODE HERE
	height, width = peaks.shape
	maximums = peaks.flatten().argsort()[-num_lines:]
	rhos = maximums // width
	# thetas = np.deg2rad(maximums % width)
	thetas = maximums % width

	return rhos, thetas
