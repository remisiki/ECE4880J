from typing import Tuple, Union
import numpy as np


def myConv2D(
	img: np.ndarray,
	kernel: np.ndarray,
	stride: Union[int, Tuple[int, int]] = 1,
	padding: Union[int, Tuple[int, int]]=0
) -> np.ndarray:
	"""Convolve two 2-dimensional arrays.

	Args:
		img (np.ndarray): A grayscale image.
		kernel (np.ndarray): A odd-sized 2d convolution kernel.
		stride (int or tuple, optional): The parameter to control the movement of the kernel. 
			An integer input k will be automatically converted to a tuple (k, k). Defaults to 1.
		padding (int or tuple, optional): The parameter to control the amount of padding to the image. 
			An integer input k will be automatically converted to a tuple (k, k). Defaults to 0.

	Returns (np.ndarray): The processed image.
	"""
	# TODO: check the datatype of stride and padding
	# hint: isinstance()
	if (isinstance(stride, int)):
		stride = (stride, stride)
	if (isinstance(padding, int)):
		padding = (padding, padding)

	# TODO: define the size of the output and initialize
	height, width = img.shape
	kernel_height, kernel_width = kernel.shape
	kernel_padding_top = (kernel_height - 1) // 2
	kernel_padding_left = (kernel_width - 1) // 2
	extra_padding_top = padding[0]
	extra_padding_left = padding[1]
	stride_height = stride[0]
	stride_width = stride[1]
	output = np.zeros((
		(height + extra_padding_top * 2 - kernel_height + stride_height) // stride_height,
		(width + extra_padding_left * 2 - kernel_width + stride_width) // stride_width
	))

	# TODO: add padding to the image
	img_padded = np.pad(img,
		(
			(kernel_padding_top + extra_padding_top, kernel_padding_left + extra_padding_top),
			(kernel_padding_top + extra_padding_left, kernel_padding_left + extra_padding_left)
		)
	)

	# TODO: implement your 2d convolution
	row_end = (height + extra_padding_top * 2 - kernel_padding_top * 2) // stride_height
	column_end = (width + extra_padding_left * 2 - kernel_padding_left * 2) // stride_width

	for i in range(0, row_end):
		for j in range(0, column_end):

			kernel_row_start = i * stride_height + kernel_padding_top
			kernel_row_end = kernel_row_start + kernel_height
			kernel_column_start = j * stride_width + kernel_padding_left
			kernel_column_end = kernel_column_start + kernel_width

			output[i][j] = np.sum(
				img_padded[kernel_row_start:kernel_row_end, kernel_column_start:kernel_column_end] * kernel
			)


	return output