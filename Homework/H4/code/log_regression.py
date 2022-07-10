import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# NOTE: The things to complete are marked by "##### ADD YOUR CODE HERE".

""" PLEASE use this completed function for for plot generation,
	for consistency among submitted plots. """
def plot(train_losses, train_accs, test_losses, test_accs, prefix):
	num_epochs = len(train_losses)
	plt.plot(range(num_epochs),train_losses,label='train loss', marker='o',linestyle='dashed',linewidth=1,markersize=2)
	plt.plot(range(num_epochs),test_losses,label='test loss', marker='o',linestyle='dashed',linewidth=1,markersize=2)
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('log loss')
	plt.savefig('./{}_loss.png'.format(prefix))
	plt.clf()
	plt.plot(range(num_epochs),train_accs,label='train acc',marker='o',linestyle='dashed',linewidth=1,markersize=2)
	plt.plot(range(num_epochs),test_accs,label='test acc',marker='o',linestyle='dashed',linewidth=1,markersize=2)
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.savefig('./{}_acc.png'.format(prefix))

def sigmoid(x, params):
	z = np.dot(x, params[0]) + params[1]
	return 1 / (1 + np.exp(-z))

def loss(params, xs, ys):
	""" This function computes the logistic loss on (xs, ys).
	Please compute the MEAN over the dataset vs. the SUM,
	the latter of which was used in the preceding theory
	section."""
	##### ADD YOUR CODE HERE
	z = sigmoid(xs, params)
	return (-(ys * np.log(z)) - (1 - ys) * np.log(1 - z)).mean()


def acc(params, xs, ys):
	""" This function computes the accuracy on (xs, ys)."""
	##### ADD YOUR CODE HERE
	z = sigmoid(xs, params)
	return np.count_nonzero(np.abs(ys - z) < 0.5) / ys.shape[0]



def get_gradients(params, xs, ys):
	""" This function should compute the gradient of the
	logistic loss with respect to the parameters.
	The loss should be a MEAN over the dataset vs. the SUM,
	the latter of which was used in the preceding theory
	section. """
	##### ADD YOUR CODE HERE
	a = sigmoid(xs, params)
	return np.dot(xs.T, a - ys) / ys.shape[0], np.dot(np.ones(ys.shape).T, a - ys) / ys.shape[0]


def apply_gradients(params, gradients, learning_rate):
	""" This function applies a gradient descent update
	step to params, using the provided computed gradients
	and learning_rate, and returns the new params."""
	##### ADD YOUR CODE HERE
	return params[0] - learning_rate * gradients[0], params[1] - learning_rate * gradients[1]

def load_data(data_dir='regression.pkl'):
	data = pkl.load(open(data_dir, 'rb'))
	train_xs = data['XTrain']
	train_ys = data['yTrain']
	test_xs = data['XTest']
	test_ys = data['yTest']
	
	# The labels are converted to {0,1}.
	# Convert y to {0,1}. Map 1-->1 and 2-->0. Recall that original labels are in {1,2}.
	for i in range(len(train_ys)): train_ys[i,0] = 1 if train_ys[i,0] == 1 else 0
	for i in range(len(test_ys)): test_ys[i,0] = 1 if test_ys[i,0] == 1 else 0

	return train_xs, train_ys, test_xs, test_ys

def train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate):
	""" This function runs gradient descent on (train_xs, train_ys)
	for num_epochs, using the specified learning_rate.
	The parameters are initialized at 0.
	The final metrics, along with parameters, are returned.
	YOU DO NOT NEED TO MODIFY ANY PARTS OF THIS FUNCTION. """
	
	# Initialize w and b at 0.
	d = train_xs.shape[1]
	w = np.zeros((d,1))
	b = 0.0
	
	train_losses, test_losses, train_accs, test_accs = [], [], [], []
	for epoch in range(num_epochs):
		# Perform an update step.
		gradients = get_gradients((w,b), train_xs, train_ys)
		w,b = apply_gradients((w,b), gradients, learning_rate)
	
		# Compute and store train and test metrics.
		train_losses.append(loss((w,b), train_xs, train_ys))
		train_accs.append(acc((w,b), train_xs, train_ys))
		test_losses.append(loss((w,b), test_xs, test_ys))
		test_accs.append(acc((w,b), test_xs, test_ys))

	return train_losses, test_losses, train_accs, test_accs, w, b

if __name__ == '__main__':
	num_epochs = 3000    # 3000
	learning_rate = 0.1
	train_xs, train_ys, test_xs, test_ys = load_data(data_dir='data/regression.pkl')
	train_losses, test_losses, train_accs, test_accs, w, b = train(train_xs, train_ys, test_xs, test_ys, num_epochs, learning_rate)
	plot(train_losses, train_accs, test_losses, test_accs, prefix='{}'.format(num_epochs))