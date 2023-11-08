# write logistic regression with newton method with using numpy only

import numpy as np

from helper.lib import *

# for explanation, https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/solving-logreg-newtons-method
# another one, https://github.com/jrios6/Math-of-Intelligence/blob/master/2-Second_Order_Optimization/Logistic%20Regression%20with%20Newton's%20Method.ipynb


class CustomLogisticRegressionNewtonMethod:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, x: npt.NDArray) -> npt.NDArray:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, z: npt.NDArray) -> npt.NDArray:
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # also log likelihood
    def binary_cross_entropy(self, theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray):
        n = X.shape[0]
        y_pred = self.sigmoid(X.dot(theta))
        cost = -1 / n * np.sum(y * np.log(y_pred) +
                               (1 - y) * np.log(1 - y_pred))
        return cost

    def logisitic_gradient(self, X: npt.NDArray, y_pred: npt.NDArray, y: npt.NDArray):
        n = X.shape[0]
        return 1 / n * X.T.dot(y_pred - y)

    def hessian(self, X: npt.NDArray, y_pred: npt.NDArray) -> npt.NDArray:
        n = X.shape[0]
        # W is diagonal matrix consisting of P * (1 - P)
        return 1 / n * X.T.dot(np.diag(y_pred * (1 - y_pred))).dot(X)

    def fit(self, X, y):
        m, n = X.shape
        self.thetas = np.random.uniform(-1, 1, size=n)

        for _ in range(self.num_iterations):
            z = np.dot(X, self.thetas)
            y_pred = self.sigmoid(z)
            gradient = self.logisitic_gradient(X, y_pred, y)
            hessian = self.hessian(X, y_pred)
            self.thetas = self.thetas - np.linalg.inv(hessian).dot(gradient)

    def predict(self, X):
        z = np.dot(X, self.thetas)
        y_pred = self.sigmoid(z)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
