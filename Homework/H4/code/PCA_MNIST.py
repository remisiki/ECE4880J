import numpy as np  
import struct  
import matplotlib.pyplot as plt  
import operator
from time import time
import sys

# Read the image data
filename = 'data/t10k-images-idx3-ubyte'  
binfile = open(filename , 'rb')  
buf = binfile.read()  
index = 0  
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)  
index += struct.calcsize('>IIII')  

# Read the labels
filename1 =  'data/t10k-labels-idx1-ubyte'  
binfile1 = open(filename1 , 'rb')  
buf1 = binfile1.read()  
  
index1 = 0  
magic1, numLabels1 = struct.unpack_from('>II' , buf , index)  
index1 += struct.calcsize('>II')  

# Initialization
DataNumbers = 10000
datamat = np.zeros((DataNumbers,28*28))
datalabel = []

# Collect data
for i in range(DataNumbers):
	im = struct.unpack_from('>784B' ,buf, index)  
	index += struct.calcsize('>784B')  
	im = np.array(im) 
	datamat[i]=im
	numtemp = struct.unpack_from('1B' ,buf1, index1) 
	label = numtemp[0]
	index1 += struct.calcsize('1B')
	datalabel.append(label)

## Start your PCA ###

datalabel = np.array(datalabel)
# np.save('data/t10k-images-idx3-ubyte.npy', datamat)
# np.save('data/t10k-labels-idx1-ubyte.npy', datalabel)
# sys.exit(0)
# datamat = np.load('data/t10k-images-idx3-ubyte.npy')
# datalabel = np.load('data/t10k-labels-idx1-ubyte.npy')
digitOneImages = datamat[datalabel == 1]
# np.save('data/t10k-images-idx3-ubyte-1.npy', digitOneImages)
# sys.exit(0)
# digitOneImages = np.load('data/t10k-images-idx3-ubyte-1.npy')
meanPoint = np.mean(digitOneImages, axis = 0)
plt.imsave('img/pca_average_one.png', meanPoint.reshape(28, 28), cmap = 'gray')
centeredPoints = digitOneImages - meanPoint
start = time()
covarianceMatrix = np.dot(centeredPoints, centeredPoints.T) / (centeredPoints.shape[0] - 1)
eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
sortedEigenIndex = np.argsort(eigenValues)[::-1]
end = time()
print(end - start)
w = np.abs(eigenVectors[sortedEigenIndex[:5]])
for i, pcaVector in enumerate(w):
	plt.imsave(f'img/pca_vector_{i}.png', pcaVector[:784].reshape(28, 28), cmap = 'gray')

for n in [1, 2, 5, 10, 20]:
	w = np.abs(eigenVectors[sortedEigenIndex[:n]])
	projectedPoints = np.dot(w, centeredPoints)
	reconstructedPoints = np.dot(w.T, projectedPoints) + meanPoint
	plt.imsave(f'img/pca_reconstruct_{n}.png', reconstructedPoints[0].reshape(28, 28), cmap = 'gray')
	MSE = ((reconstructedPoints[0] - digitOneImages[0]) ** 2).sum(axis = 0).mean()
	print(f"MSE: {MSE}")
