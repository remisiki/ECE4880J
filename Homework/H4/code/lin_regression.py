import numpy as np

# Template code for gradient descent, to make the implementation
# a bit more straightforward.
# Please fill in the missing pieces, and then feel free to
# call it as needed in the functions below.
def gradient_descent(X, y, lr, num_iters):
	losses = []
	n, d = X.shape
	w = np.zeros((d,1))
	for i in range(num_iters):
		grad = np.dot(X.T, (np.dot(X, w) - y))
		w = w - lr * grad
		loss = ((np.dot(X, w) - y) ** 2).sum()
		losses.append(loss)
	return losses, w

# Code for (2-5)
def linear_regression():
	X = np.array([[1,1],[2,3],[3,3]])
	Y = np.array([[1],[3],[3]])    

	##### ADD YOUR CODE FOR ALL PARTS HERE
	l = 1e-5
	w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + l * np.identity(X.shape[1])), X.T), Y)
	print(w)
	losses, w = gradient_descent(X, Y, 0.01, 10000)
	print(w)
