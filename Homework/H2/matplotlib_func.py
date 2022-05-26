import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def w1():
	"""
	Draw a line chart and save as a jpg file

	Hint: plt.subplot(), ax.plot()
	"""
	data_x = [1, 2, 3, 4]
	data_y = [1, 4, 2, 3]
	file_name = 'line_chart.jpg'
	ax = plt.subplot(111)
	ax.plot(data_x, data_y)
	plt.savefig(file_name)
	return file_name

def w2():
	"""
	Draw a scatter chart and save as a jpg file

	Hint: Use ax.scatter()
	"""
	data_x = [1, 2, 3, 4]
	data_y = [1, 4, 2, 3]
	file_name = 'scatter_chart.jpg'
	ax = plt.subplot(111)
	ax.scatter(data_x, data_y)
	plt.savefig(file_name)
	return file_name

def w3():
	"""
	Draw a chart with multiple lines of x,x^2,x^3 and save as a jpg file

	Hint: Trust plt.plot()
	"""
	x = np.arange(0., 5., 0.2)
	file_name = 'multiple_lines.jpg'
	ax = plt.subplot(111)
	ax.plot(x, x)
	ax.plot(x, x ** 2)
	ax.plot(x, x ** 3)
	plt.savefig(file_name)
	return file_name

def w4():
	"""
	Draw a histogram chart values-names and save as a jpg file
	
	Hint: plt.bar()
	"""
	names = ['group_a', 'group_b', 'group_c']
	values = [1, 10, 100]
	file_name = 'histogram_chart.jpg'
	ax = plt.subplot(111)
	ax.bar(names, values)
	plt.savefig(file_name)
	return file_name

def w5():
	"""
	Read and save an image with B channel (R G B channel) as a jpg file

	Hint: matplotlib.image.imread()
	"""
	img_path = 'img.jpg'
	file_name = 'img_blue.jpg'
	img_arr = matplotlib.image.imread(img_path)
	matplotlib.image.imsave(file_name, img_arr.T[2].T)
	return file_name

