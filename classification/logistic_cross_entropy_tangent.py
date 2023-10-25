from helper.lib import *


class CustomLogisticRegressionTangent:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def tanh(self, x):
        x = np.clip(x, -740, 700) # https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function to avoid overflow
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        n = y_pred.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1/ n * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def tangent_gradient(self, X: npt.NDArray, y: npt.NDArray[np.float64], theta: npt.NDArray):
        n = X.shape[0]
        z = X.dot(theta)
        y_pred = self.tanh(z)
        # Gradient of the loss function with respect to the i-th weight using tanh activation
        gradient = (((y/y_pred) * -1) + (1 - y)/(1 - y_pred)) * (1 - y_pred**2)
        gradient = np.dot(X.T, gradient)
        return gradient / n

    def fit(self, X, y):
        m, n = X.shape
        self.thetas = np.random.uniform(-1, 1, size=n)

        for _ in range(self.num_iterations):
            z = np.dot(X, self.thetas)
            y_pred = self.tanh(z)
            loss = self.binary_cross_entropy(y, y_pred)
            gradient = self.tangent_gradient(X, y, self.thetas)
            self.thetas = self.thetas - (self.learning_rate / m) * gradient

    def predict(self, X):
        z = np.dot(X, self.thetas)
        y_pred = self.tanh(z)
        return np.round(y_pred)
    
# Create and train the custom logistic regression model
# model = CustomLogisticRegressionTangent()
# model.fit(X_train, y)
# y_pred = model.predict(X_train)