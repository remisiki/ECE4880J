import scipy
from scipy import linalg, optimize, io, ndimage
import numpy as np


def w1(A,b):
	"""
	Solves the linear equation set a * x = b for the unknown x for square a matrix.
	Input:
	- A: A numpy array with shape [M,N]
	- b: A numpy array with shape [M,]

	Returns:
	- Solved result X with shape [N,]

	Hint: Use linalg.solve
	"""
	return scipy.linalg.solve(A, b).flatten()

def w2(A):
	"""
	Compute the inverse of a matrix   
	 Input:
	- A: A numpy array with shape [N,N]

	Returns:
	- Inverse of A with shape [N,N]

	Hint: Use linalg.inv
	"""
	return scipy.linalg.inv(A)

def w3(A,b):
	"""
	Solve argmin_x || Ax - b ||_2 for x>=0. 

	Input:
	- A: A numpy array with shape [M,N]
	- b: A numpy array with shape [M,]

	Returns:
	- Solution vector x with shape [N,]

	Hint: Use scipy.optimize.nnls
	"""
	return scipy.optimize.nnls(A, b)[0]

def w4(a):
	"""
	Save a dictionary of names and arrays into a .mat file.

	Input:
	- a: Dictionary from which to save matfile variables.
	e.g: a = {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]), 'label': 'experiment'}

	Hint: Use scipy.io.savemat
	"""
	file_name = 'matrix.mat'
	scipy.io.savemat(file_name, a)
	return file_name

def w5(A):
	"""
	Rotate an array for 180 degree on the 0-th axis.

	The array is rotated in the plane defined by the two axes given by the axes parameter using spline interpolation of the requested order.
	
	Input:
	- A: The input array as an image

	Returns:
	- Rotated image

	Hint: Use scipy.ndimage.rotate
	"""
	return scipy.ndimage.rotate(A, 180)

