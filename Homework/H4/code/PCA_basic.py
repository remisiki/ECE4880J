import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

def pca_basic():
	### Data
	c1 = np.array([[2,2,2],[1,2,3]])
	c2 = np.array([[4,5,6],[3,3,4]])
	c = np.concatenate((c1,c2),axis=1)

	### Calculate w and w0 here
	meanPoint = np.mean(c, axis = 1).reshape(2, 1)
	centeredPoints = c - meanPoint
	covarianceMatrix = np.cov(c)
	eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix)
	sortedEigenIndex = np.argsort(eigenValues)[::-1]
	w = np.abs(eigenVectors[sortedEigenIndex[0]].reshape(2, 1))
	w0 = -np.dot(w.T, meanPoint)

	print('w is:', w)
	print('w0 is:', w0)

	### Plot the reconstructed points here
	projectedPoints = np.dot(w.T, centeredPoints)
	reconstructedPoints = np.dot(w, projectedPoints) + meanPoint
	plt.figure(figsize=(9, 9))
	plt.xlim(0, 10)
	plt.ylim(0, 10)
	plt.scatter(c1[0], c1[1])
	plt.scatter(c2[0], c2[1])
	plt.scatter(reconstructedPoints[0], reconstructedPoints[1])
	plt.axline(meanPoint.flatten(), slope = - w[0] / w[1])
	plt.axline(meanPoint.flatten(), slope = w[1] / w[0])
	plt.savefig('img/pca_basic.png')



	### Calculate MSE here
	MSE = ((reconstructedPoints - c) ** 2).sum(axis = 0).mean()


	print('MSE is:', MSE)
	### Calculate the Fisher Ratio here
	projectedPoints = [projectedPoints[0][:3], projectedPoints[0][3:]]
	FR = ((np.mean(projectedPoints[0])-np.mean(projectedPoints[1])) ** 2) / (np.var(projectedPoints[0])+np.var(projectedPoints[1]))


	print('Fisher Ratio is:', FR)
	return w,w0,MSE,FR


pca_basic()


