from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, projections
from mpl_toolkits.mplot3d import Axes3D

def ReLU(x:np.ndarray, weight:np.ndarray, bias:np.ndarray) -> np.ndarray:
	"""A ReLU neuron

	Args:
		weight (np.ndarray): a numpy array with shape (1, N)
		bias (np.ndarray):  a numpy array with shape (1, N)

	Returns:
		np.ndarray: the activated result
	"""
	activated = weight * x + bias
	activated[activated < 0] = 0
	return activated


def Sigmoid(x:np.ndarray, weight:np.ndarray, bias:np.ndarray) -> np.array:
	"""A Sigmoid neuron

	Args:
		weight (np.ndarray): a numpy array with shape (1, N)
		bias (np.ndarray):  a numpy array with shape (1, N)

	Returns:
		np.ndarray: the activated result
	"""
	activated = weight * x + bias
	activated = 1.0 / (1.0 + np.exp(- activated))
	return activated


def L1_loss(predicted:np.ndarray, y:np.ndarray) -> np.ndarray:
	"""The L1 loss function, AKA the Mean Absolute Error (MAE) loss

	Args:
		predicted (np.ndarray): The predicted y based on the input x. 
		y (np.ndarray): The ground truth y.

	Returns:
		np.ndarray: the L1 loss
	"""
	return np.abs(predicted - y)


def L2_loss(predicted:np.ndarray, y:np.ndarray) -> np.ndarray:
	"""The L2 loss function, AKA Mean Squared Error (MSE) loss

	Args:
		predicted (np.ndarray): The predicted y based on the input x. 
		y (np.ndarray): The ground truth y.

	Returns:
		np.ndarray: the L2 loss
	"""
	L2_loss = np.power(predicted - y, 2)
	return L2_loss


def CrossEntropy_loss(predicted:np.ndarray, y:np.ndarray) -> np.ndarray:
	"""The cross-entropy loss

	Args:
		predicted (np.ndarray): The predicted y based on the input x. 
		y (np.ndarray): The ground truth y.

	Returns:
		np.ndarray: the L2 loss
	"""

	cross_entropy = - (y * np.log(predicted) + (1 - y) * np.log(1 - predicted))
	return cross_entropy


def plot_Q1(
	activationFunc: Callable,
	lossFunc: Callable,
	funcName: str,
	plotType: str,
	fileName: str = "hw5_Q1_sample.png",
) -> None:
	# implement your own plot function
	weight = np.arange(-2., 2.1, 0.1)
	bias = np.arange(-2., 2.1, 0.1)
	weight, bias = np.meshgrid(weight, bias)
	x = np.ones(weight.shape)
	y = 0.5 * np.ones(weight.shape)

	predicted = activationFunc(x, weight, bias)

	fig = plt.figure()

	if (plotType == "activation"):
		plotTitle = f"{funcName} function"
		z = predicted
	elif (plotType == "loss"):
		plotTitle = f"{funcName} loss"
		loss = lossFunc(predicted, y)
		z = loss
	elif (plotType == "gradient"):
		plotTitle = f"Gradient of {funcName} loss"
		loss = lossFunc(predicted, y)
		gradient = np.gradient(loss, 0.1)[0]
		z = gradient
	else:
		raise Exception(f"Invalid plot type '{plotType}'.")

	ax = fig.add_subplot(111, projection = '3d')
	surf = ax.plot_surface(
		weight,
		bias,
		z,
		cmap = cm.coolwarm,
		linewidth = 0,
		antialiased = False
	)
	ax.set_xlabel("weight")
	ax.set_ylabel("bias")
	ax.set_title(plotTitle)
	fig.colorbar(surf)

	plt.savefig(fileName)

def plot_demo(activation_func:Callable, loss_func:Callable, 
						fname:str="hw5_Q1_sample.png", 
						savefig:bool=True) -> None:
	"""This function will reproduce the sample plot."""
	weight = np.arange(-2., 2.1, 0.1)
	bias = np.arange(-2., 2.1, 0.1)
	weight, bias = np.meshgrid(weight, bias)
	x = np.ones(weight.shape)
	y = 0.5 * np.ones(weight.shape)

	predicted = activation_func(x, weight, bias)
	loss = loss_func(predicted, y)
	gradient = np.gradient(loss, 0.1)[0]

	fig = plt.figure(figsize=plt.figaspect(0.33))

	ax = fig.add_subplot(1,3,1, projection='3d')
	surf = ax.plot_surface(weight, bias, predicted, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel("weight")
	ax.set_ylabel("bias")
	ax.set_title("ReLU function")
	fig.colorbar(surf, location='bottom', shrink=0.5, aspect=10)

	ax = fig.add_subplot(1,3,2, projection='3d')
	surf = ax.plot_surface(weight, bias, loss, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel("weight")
	ax.set_ylabel("bias")
	ax.set_title("L1 loss")
	fig.colorbar(surf, location='bottom', shrink=0.5, aspect=10)

	ax = fig.add_subplot(1,3,3, projection='3d')
	surf = ax.plot_surface(weight, bias, gradient, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel("weight")
	ax.set_ylabel("bias")
	ax.set_title("Gradient of L1 loss")
	fig.colorbar(surf, location='bottom', shrink=0.5, aspect=10)

	if savefig:
		plt.savefig(fname)

	plt.show()


if __name__ == "__main__":
	plot_Q1(Sigmoid, L2_loss, "L2", "gradient")