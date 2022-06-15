import numpy as np
import math

def Handwrite_HoughTransform(img, rhostep, thetastep):
	# YOUR CODE HERE
	height = img.shape[0]
	width = img.shape[1]
	rho_max = np.sqrt(np.square(height) + np.square(width))
	theta_range = np.arange(0, 180, np.rad2deg(thetastep))
	theta_int = np.int64(theta_range)
	theta_cos = np.cos(np.deg2rad(theta_range))
	theta_sin = np.sin(np.deg2rad(theta_range))
	rho_range = np.arange(0, rho_max, rhostep)
	h_table = np.zeros((len(rho_range), len(theta_range)))

	for x in range(height):
		for y in range(width):
			if (img[x][y] > 0):
				r = np.int64(y * theta_cos + x * theta_sin)
				for rho, theta in zip(r, theta_int):
					i = (np.abs(rho_range - np.abs(rho))).argmin()
					j = (np.abs(theta_range - theta)).argmin()
					if (i > 0 and j > 0):
						h_table[i][j] += 1

	return [h_table, rho_range, np.deg2rad(theta_range)]
