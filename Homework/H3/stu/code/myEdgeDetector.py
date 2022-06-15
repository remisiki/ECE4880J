from contextlib import suppress
from typing import Tuple, Union 

import matplotlib.pyplot as plt
import numpy as np

from myConvolution import myConv2D


def GaussianKernel(sigma: float = 1.):
	"""Generate a gaussian kernel with the given standard deviation.

	The kernel size is decided by 2*ceil(3*sigma) + 1
	"""
	# TODO: calculate the kernel size and initialize
	size = int(2 * np.ceil(3 * sigma) + 1)
	ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)

	# TODO: generate the gaussian kernel
	gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
	kernel = np.outer(gauss, gauss)
	kernel /= np.sum(kernel)

	return kernel


def SobelFilter(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	# the Sobel operators on x-direction and y-direction
	Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]], np.float32)
	Ky = np.array([[1,2,1], [0,0,0], [-1,-2,-1]], np.float32)

	# TODO: calculate the intensity gradient
	Gx = myConv2D(img, Kx, padding = 1)
	Gy = myConv2D(img, Ky, padding = 1)
	G = np.sqrt(np.square(Gx) + np.square(Gy))
	theta = np.arctan2(Gy, Gx)

	return G, theta


def NMS(img: np.ndarray, angles: np.ndarray) -> np.ndarray:
	"""Process the image with non-max suppression algorithm
	"""
	# map the gradient angle to the closest of 4 cases, where the line is sloped at 
	#   almost 0 degree, 45 degree, 90 degree, and 135 degree

	height, width = img.shape
	suppressed = np.zeros((height, width))
	angles *= (180 / np.pi)
	angles %= 360

	for i in range(height):
		for j in range(width):
			if (((i + 1) >= height) or ((j + 1) >= width) or ((i - 1) < 0) or ((j - 1) < 0)):
				# Out of boundary
				continue
			angle = angles[i][j]
			if ((angle >= 337.5 or angle < 22.5) or (angle >= 157.5 and angle < 202.5)):
				# 0 degree
				if (img[i][j] >= img[i][j + 1] and img[i][j] >= img[i][j - 1]):
					suppressed[i][j] = img[i][j]
			elif ((angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5)):
				# 45 degree
				if (img[i][j] >= img[i - 1][j + 1] and img[i][j] >= img[i + 1][j - 1]):
					suppressed[i][j] = img[i][j]
			elif ((angle >= 67.5 and angle < 112.5) or (angle >= 247.5 and angle < 292.5)):
				# 90 degree
				if (img[i][j] >= img[i - 1][j] and img[i][j] >= img[i + 1][j]):
					suppressed[i][j] = img[i][j]
			elif ((angle >= 112.5 and angle < 157.5) or (angle >= 292.5 and angle < 337.5)):
				# 135 degree
				if (img[i][j] >= img[i - 1][j - 1] and img[i][j] >= img[i + 1][j + 1]):
					suppressed[i][j] = img[i][j]
	return suppressed

def myCanny(
	img: np.ndarray,
	sigma: Union[float, Tuple[float,float]] = 1.,
	threshold: Tuple[int,int] = (100, 150)
) -> np.ndarray:
	"""Apply Canny algorithm to imgect the edge in an image.

	Returns: The edge imgection result whose size is the same as the input image.
	"""
	# denoise the image by a convolution with the gaussian filter
	#   TODO: implement the Gaussian kernel generator
	gaussian_kernel = GaussianKernel(sigma)
	padding = np.floor(np.array(gaussian_kernel.shape) / 2)
	denoised = myConv2D(img, gaussian_kernel, padding = (int(padding[0]), int(padding[1])))
	plt.imsave("./zebra_denoised.jpg", denoised, cmap = "Greys_r")

	# find the intensity gradient of the image
	#   TODO: implement the Sobel filter
	gradient, angles = SobelFilter(denoised)
	plt.imsave("./zebra_gradient.jpg", gradient, cmap = "Greys_r")

	# find the edge candidates by non-max suppression
	#   TODO: implement the non-max suppression function
	nms = NMS(gradient, angles)
	plt.imsave("./zebra_nms.jpg", nms, cmap = "Greys_r")

	# TODO: imgermine the potential edges by the hysteresis threshold

	nms *= 255 / np.max(nms)
	output = np.where(nms > threshold[1], img, 0)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if ((nms[i][j] <= threshold[1]) and (nms[i][j] >= threshold[0])):
				if (
					output[i - 1][j] or output[i - 1][j - 1] or output[i - 1][j + 1] or
					output[i][j - 1] or output[i][j + 1] or
					output[i + 1][j] or output[i + 1][j - 1] or output[i + 1][j + 1]
				):
					output[i][j] = img[i][j]
	plt.imsave("./zebra_edge.jpg", output, cmap="Greys_r")
	return output