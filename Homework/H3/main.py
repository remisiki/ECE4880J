import stu.code.houghScript as test
# import stu.code.compression as test
import numpy as np
import matplotlib.image as mpimg

if __name__ == '__main__':
	pass
	# test.main()
	# image = mpimg.imread('./stu/img/lena.jpg').astype('float')
	# image = mpimg.imread('./stu/img/hw3_zebra.jpg').astype('float')
	# edge = test.myCanny(image)
	# print(kernel)
	# kernel = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
	# kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
	# kernel = np.array(
	# 	[
	# 		[0, 0, 1, 2, 1, 0, 0],
	# 		[0, 3, 13, 22, 13, 3, 0],
	# 		[1, 13, 59, 97, 59, 13, 1],
	# 		[2, 22, 97, 159, 97, 22, 2],
	# 		[1, 13, 59, 97, 59, 13, 1],
	# 		[0, 3, 13, 22, 13, 3, 0],
	# 		[0, 0, 1, 2, 1, 0, 0]
	# 	]
	# ) / 1003
	# kernel = np.ones((3, 3)) - np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
	# conv = test.myConv2D(image, kernel, stride = (1, 3), padding = (5, 7))
	# conv = test.myConv2D(image, kernel)
	# file_name = "lena_conv.jpg"
	# mpimg.imsave(file_name, conv, cmap = 'gray')
