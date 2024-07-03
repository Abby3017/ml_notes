from typing import Callable

import numpy.typing as npt

from helper.data_generator import augment_data
from helper.lib import *
from plotting.plot_simple import plot_dataset


# for sigmoid, labels are 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(z: npt.NDArray) -> npt.NDArray:
    return sigmoid(z) * (1 - sigmoid(z))


def calculate_accuracy(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray, activation: Callable):
    y_pred = activation(X.dot(theta))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return np.mean(y_pred == y)

# log loss (this works)


def cost_function_sigmoid(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray):
    n = X.shape[0]
    y_pred = sigmoid(X.dot(theta))
    cost = -1 / n * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


def logisitic_gradient(X: npt.NDArray, y_pred: npt.NDArray, y: npt.NDArray):
    n = X.shape[0]
    return 1 / n * X.T.dot(y_pred - y)


def gradient_descent(theta, X, y, alpha, num_iters, activation_gradient: Callable, activation: Callable, cost_function: Callable):
    n = X.shape[0]
    J_history = np.zeros(num_iters)
    J_accuracy = np.zeros(num_iters)
    best_theta = theta
    best_accuracy = 0
    for i in range(num_iters):
        y_pred = sigmoid(X.dot(theta))
        theta = theta - alpha * activation_gradient(X, y, theta)
        J_history[i] = cost_function(X, y, theta)
        J_accuracy[i] = calculate_accuracy(theta, X, y, activation)
        if best_accuracy < J_accuracy[i]:
            best_accuracy = J_accuracy[i]
            best_theta = theta

    return best_theta, J_history, J_accuracy

# mean square loss with normalising gradient


def cost_function_sigmoid_mean_square(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray):
    n = X.shape[0]
    y_pred = sigmoid(X.dot(theta))
    cost = 1 / n * np.sum(np.square(y_pred - y))
    return cost


def logistic_mean_square_gradient(X: npt.NDArray, y_pred: npt.NDArray, y: npt.NDArray):
    n = X.shape[0]
    return 2 / n * X.T.dot((y_pred - y) * (y_pred * (1 - y_pred)))


def normalised_gradient_descent(theta, X, y, alpha, num_iters, activation: Callable):
    n = X.shape[0]
    J_history = np.zeros(num_iters)
    J_accuracy = np.zeros(num_iters)
    best_theta = theta
    best_accuracy = 0
    for i in range(num_iters):
        y_pred = sigmoid(X.dot(theta))
        gradient = logistic_mean_square_gradient(X, y_pred, y)
        normalised = np.linalg.norm(gradient) + 1e-8
        gradient = gradient / normalised
        theta = theta - alpha * gradient
        J_history[i] = cost_function_sigmoid_mean_square(X, y, theta)
        J_accuracy[i] = calculate_accuracy(theta, X, y, activation)
        if best_accuracy < J_accuracy[i]:
            best_accuracy = J_accuracy[i]
            best_theta = theta

    return best_theta, J_history, J_accuracy

# for tangent, labels are -1 and 1


def tangent(x: npt.NDArray):
    # https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function to avoid overflow
    x = np.clip(x, -740, 700)
    y_pred = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # simple transformation to map the range (-1, 1) to (0, 1):
    output = 0.5 * (y_pred + 1)
    return y_pred, output


def tangent_derivative(x: npt.NDArray):
    return 1 - np.square(tangent(x))


def cost_function_tang(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray):
    n = X.shape[0]
    y_pred, _ = tangent(X.dot(theta))
    y_pred = np.clip(y_pred, 1e-7, None)  # to avoid log(0)
    cost = -1 / n * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


def tangent_gradient(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray):
    n = X.shape[0]
    z = X.dot(theta)
    y_pred, output = tangent(z)
    # Gradient of the loss function with respect to the i-th weight using tanh activation
    gradient = (((y/y_pred) * -1) + (1 - y)/(1 - y_pred)) * (1 - y_pred**2)
    gradient = np.dot(X.T, gradient)
    return gradient / n


def get_wrong_prediction(theta, X, y, activation: Callable):
    X_train = augment_data(X)
    y_pred = activation(X_train.dot(theta))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    X_wrong_pred = X[y_pred != y]
    y_wrong_pred = y[y_pred != y]
    print("Number of wrong predictions: ", X_wrong_pred.shape[0])
    plot_dataset(X_wrong_pred, y_wrong_pred, theta)

# mean square loss with normalising gradient


def cost_function_tang_mean_square(X: npt.NDArray, y: npt.NDArray, theta: npt.NDArray):
    n = X.shape[0]
    y_pred, _ = tangent(X.dot(theta))
    cost = 1 / n * np.sum(np.square(y_pred - y))
    return cost


def tangent_mean_square_gradient(X: npt.NDArray, y_pred: npt.NDArray, y: npt.NDArray):
    n = X.shape[0]
    return 2 / n * X.T.dot((y_pred - y) * (1 - y_pred**2))


def calculate_accuracy_tang(theta: npt.NDArray, X: npt.NDArray, y: npt.NDArray, activation: Callable):
    _, output = activation(X.dot(theta))
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    return np.mean(output == y)


def normalised_gradient_descent_tangent(theta, X, y, alpha, num_iters, activation: Callable):
    n = X.shape[0]
    J_history = np.zeros(num_iters)
    J_accuracy = np.zeros(num_iters)
    best_theta = theta
    best_accuracy = 0
    for i in range(num_iters):
        y_pred = sigmoid(X.dot(theta))
        gradient = logistic_mean_square_gradient(X, y_pred, y)
        normalised = np.linalg.norm(gradient) + 1e-8
        gradient = gradient / normalised
        theta = theta - alpha * gradient
        J_history[i] = cost_function_tang_mean_square(X, y, theta)
        J_accuracy[i] = calculate_accuracy_tang(theta, X, y, activation)
        if best_accuracy < J_accuracy[i]:
            best_accuracy = J_accuracy[i]
            best_theta = theta

    return best_theta, J_history, J_accuracy
