import numpy as np
import matplotlib.pyplot as plt

dataFileName = 'kmeans_array2'
data = np.load(f'data/{dataFileName}.npy')

# Visualize data
# plt.scatter(data[:, 0], data[:, 1], c='c',s=30,marker='o')

### Start your K-means ###
clusterCount = 5
iterationCount = 16

centers = np.array([
	np.random.uniform(low = data.T[0].min(), high = data.T[0].max(), size = (clusterCount,)),
	np.random.uniform(low = data.T[1].min(), high = data.T[1].max(), size = (clusterCount,))
]).T

for k in range(iterationCount):
	delta = np.zeros((data.shape[0], clusterCount))
	for i in range(data.shape[0]):
		clusterId = np.argmin(((data[i] - centers) ** 2).sum(axis = 1))
		delta[i][clusterId] = 1

	for j in range(clusterCount):
		centers[j] = (np.repeat(delta[:, j], data.shape[1]).reshape(data.shape) * data).sum(axis = 0)
		sumOfDelta = delta[:, j].sum()
		if (sumOfDelta == 0):
			centers[j] = np.array([
				np.random.uniform(low = data.T[0].min(), high = data.T[0].max(), size = (1,)),
				np.random.uniform(low = data.T[1].min(), high = data.T[1].max(), size = (1,))
			]).T
		else:
			centers[j] /= sumOfDelta

colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
for i in range(clusterCount):
	clusterData = data[delta[:, i] == 1]
	color = colors[i % len(colors)]
	plt.scatter(clusterData[:, 0], clusterData[:, 1], color = color, s = 30, marker = 'o')
	plt.scatter(centers[i][0], centers[i][1], color = color, s = 100, marker = '*')
plt.savefig(f'img/{dataFileName}_i{iterationCount}_c{clusterCount}.png')