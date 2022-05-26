# import numpy as np
# import matplotlib_func
import matplotlib.pyplot as plt

if __name__ == '__main__':
	# main()
	# epoch_x = [1, 2, 4, 8, 16]
	# epoch_result = [63.1, 73.1, 65.0, 59.0, 64.9]
	# epoch_result_mse = [45.6, 66.6, 71.0, 68.5, 64.1]
	# ax = plt.subplot(111)
	# ax.scatter(epoch_x, epoch_result)
	# ax.scatter(epoch_x, epoch_result_mse)
	# plt.xlabel('# Epoch')
	# plt.ylabel('Plane accuracy')
	# plt.savefig('epoch_accuracy.png')
	lr_x = range(-5, 1)
	lr_result = [40.4, 55.6, 74.5, 4.4, 0.0, 0.0]
	lr_result_mse = [41.6, 68.4, 58.4, 0.0, 0.0, 0.0]
	ax = plt.subplot(111)
	ax.scatter(lr_x, lr_result)
	# ax.scatter(lr_x, lr_result_mse)
	ax.set_xlabel('log(Learning rate)')
	ax.set_ylabel('Ship accuracy')
	plt.xticks(lr_x)
	plt.savefig('lr_accuracy.png')