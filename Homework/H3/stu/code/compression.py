import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.image import imsave

def DCT_transform(img):
	"""
	Inputs:
	- img: An 2-D numpy array of shape (H,W)

	Returns:
	- img_dct: the dct transformation result, a 2-D numpy array of shape (H, W)

	Hint: implement the dct transformation basis matrix
	"""
	H,W = img.shape

	img_dct = np.zeros((H, W))
	c_table_h = np.zeros((H, W))
	c_table_w = np.zeros((H, W))
	s_table = np.zeros((H, W))
	f_table = np.zeros((H, W))
	for i in range(H):
		if (i == 0):
			c_table_h[i].fill(1)
			c_table_w[i].fill(1)
			continue
		for j in range(W):
			c_table_h[i][j] = np.sqrt(2) * np.cos((2*j+1)*i*np.pi/(2*H))
			c_table_w[i][j] = np.sqrt(2) * np.cos((2*j+1)*i*np.pi/(2*W))
	for x in range(H):
		for v in range(W):
			s_table[x][v] = np.multiply(img[x], c_table_w[v]).sum()
	for u in range(H):
		for v in range(W):
			f_table[u][v] = np.dot(c_table_h[u], s_table[:, v])
	f_table /= np.sqrt(H*W)

	img_dct = f_table

	return img_dct

def iDCT_transform(img_dct):
	"""
	Inputs:
	- img_dct: An 2-D numpy array of shape (H,W)

	Returns:
	- img_recover: recoverd image, a 2-D numpy array of shape (H, W)

	Hint: use the same dct transformation basis matrix but do the reverse operation
	"""
	H,W = img_dct.shape

	img_recover = np.zeros((H, W))
	c_table_h = np.zeros((H, W))
	c_table_w = np.zeros((H, W))
	s_table = np.zeros((H, W))
	f_table = np.zeros((H, W))
	for i in range(H):
		if (i == 0):
			c_table_h[i].fill(1)
			c_table_w[i].fill(1)
			continue
		for j in range(W):
			c_table_h[i][j] = np.sqrt(2) * np.cos((2*j+1)*i*np.pi/(2*H))
			c_table_w[i][j] = np.sqrt(2) * np.cos((2*j+1)*i*np.pi/(2*W))
	for u in range(H):
		for y in range(W):
			s_table[u][y] = np.dot(img_dct[u], c_table_w[:, y])
	for x in range(H):
		for y in range(W):
			f_table[x][y] = np.multiply(s_table[:, y], c_table_h[:, x]).sum()
	f_table /= np.sqrt(H*W)

	img_recover = f_table

	return img_recover


def main():
	#############################################
	############ Global compression #############
	#############################################

	image = mpimg.imread('./stu/img/lena.jpg')
	image = image.astype('float')
	H,W = image.shape

	image_dct = DCT_transform(image)

	### Visualize the log map of dct (image_dct_log) here ###
	image_dct_log = np.log(abs(image_dct))
	file_name = 'dct_log_map.png'
	x0 = np.zeros(512, dtype = int)
	y0 = np.arange(512)
	x = x0
	y = y0
	for i in range(511):
		x = np.append(x, x0 + i + 1)
		y = np.append(y, y0)
	ax = plt.subplot(111)
	ax.xaxis.tick_top()
	im = ax.scatter(x, y, s = 1, c = image_dct_log)
	plt.colorbar(im)
	plt.gca().invert_yaxis()
	plt.savefig(file_name)

	### Compress the dct result here by preserving 1/4 data (H/2,W/2) in image_dct and set others to zero here ###
	image_dct_compress = np.zeros((H, W))
	image_dct_compress[0:(H//2), 0:(W//2)] = image_dct[0:(H//2), 0:(W//2)]

	image_recover = iDCT_transform(image_dct)
	image_compress_recover = iDCT_transform(image_dct_compress)
	file_name = "lena_recover.jpg"
	imsave(file_name, image_recover, cmap = 'gray')
	file_name = "lena_compress.jpg"
	imsave(file_name, image_compress_recover, cmap = 'gray')

	image_dct_compress = np.zeros((H, W))
	image_dct_compress[0:(H//4), 0:(W//4)] = image_dct[0:(H//4), 0:(W//4)]
	image_compress_recover = iDCT_transform(image_dct_compress)
	file_name = "lena_compress_16.jpg"
	imsave(file_name, image_compress_recover, cmap = 'gray')


	#############################################
	########## Blockwise compression ############
	#############################################
	image = mpimg.imread('./stu/img/lena.jpg')
	image = image.astype('float')

	H,W = image.shape

	patches_num_h = int(H/8)
	patches_num_w = int(W/8)

	img_recover = np.zeros(image.shape)
	image_recover_compress = np.zeros(image.shape)
	image_dct_log = np.zeros(image.shape)

	for i in range(patches_num_h):
		for j in range(patches_num_w):
			### divide the image into 8x8 pixel image patches here ###
			patch = image[i*8:(i+1)*8, j*8:(j+1)*8]
			patch_dct = DCT_transform(patch)

			patch_dct_log = np.log(abs(patch_dct))

			### Compress the dct result here by preserving 1/4 data (H/2,W/2) in image_dct and set others to zero here
			patch_dct_compress = np.zeros((8, 8))
			patch_dct_compress[0:4, 0:4] = patch_dct[0:4, 0:4]

			patch_recover = iDCT_transform(patch_dct)
			patch_compress_recover = iDCT_transform(patch_dct_compress)

			### put patches together here
			img_recover[i*8:(i+1)*8, j*8:(j+1)*8] = patch_recover
			image_recover_compress[i*8:(i+1)*8, j*8:(j+1)*8] = patch_compress_recover
			image_dct_log[i*8:(i+1)*8, j*8:(j+1)*8] = patch_dct_log

	file_name = "lena_patches_compress.jpg"
	imsave(file_name, image_recover_compress, cmap = 'gray')
	file_name = "lena_patches_recover.jpg"
	imsave(file_name, img_recover, cmap = 'gray')

if __name__ == "__main__":
	main()


 
