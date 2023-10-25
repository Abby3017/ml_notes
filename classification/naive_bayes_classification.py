from typing import Callable

import numpy.typing as npt
from scipy.special import logsumexp
from scipy.stats import bernoulli, norm

from helper.data_generator import augment_data
from helper.lib import *
from plotting.plot_simple import plot_dataset


# Distribution for continuous features
class ContFeatureParam:
    def estimate(self, X):
        mean = X.mean()
        var = X.var(ddof=0)
        return mean, var

    def get_probability(self, val, params):
        var_smoothing = 1e-9
        sigma = (params[1] + var_smoothing)**0.5
        
        return norm.pdf(val, loc=params[0], scale=sigma) + var_smoothing


# Distribution for binary features
class BinFeatureParam:
    def estimate(self, X):
        num_samples = X.shape[0]
        unique_vals, counts = np.unique(X, return_counts=True)
        # counts[0] is the number of zeros, counts[1] is the number of ones

        # compute prob of success (p)
        prob = 1 - counts[0]/num_samples
        
        return prob

    def get_probability(self, val, prob):
        # returns the density value of the input value val
        result = np.ones_like(val) * prob
        zeros = np.argwhere(val != 1)
        result[zeros] = 1-prob # prob of failure for zeros
        return result + 1e-9


# Distribution for categorical features
class CatFeatureParam:

    def estimate(self, X):

        # Estimate the parameters for the Multinoulli distribution
        prob_vec = np.ones((30)) * 1e-10
        num_samples = X.shape[0]
        unique_vals, counts = np.unique(X, return_counts=True)
        # probability of each unique value is its count over the total number of samples
        for i in range(len(unique_vals)):
            prob_vec[i] = counts[i]/num_samples + 1e-10
        return prob_vec

    def get_probability(self, val, prob_vec):
        # returns the density value of the input value val
        return prob_vec[val]

class NBC:
    # Inputs:
    #   feature_types: the array of the types of the features, e.g., feature_types=['r', 'r', 'r', 'r']
    #   num_classes: number of classes of labels
    def __init__(self, feature_types=[], num_classes=0):
        
        # probability vector of each class
        self.y_prob = [] 
        
        # parameters we will use later in fit and predict
        self.feature_types = np.array(feature_types)
        self.num_classes = num_classes
        self.num_features = len(feature_types)

        # initiating three instances of the classes to test
        self.cont = ContFeatureParam()
        self.bin = BinFeatureParam()
        self.cat = CatFeatureParam()

    # The function uses the input data to estimate all the parameters of the NBC
    # You should use the parameters based on the types of the features
    def fit(self, X, y):
      
      self.y_prob = []
      
      # unique values of label y
      self.y_unique_vals = np.unique(np.array(y))
      print(self.y_unique_vals)
      
      X = np.array(X)
      # probability vector for each type r, b, and c
      self.prob_vec_r = np.zeros((self.num_classes, self.num_features, 2))
      self.prob_vec_b = np.zeros((self.num_classes, self.num_features, 1))
      self.prob_vec_c = np.zeros((self.num_classes, self.num_features, 30))

      # loop over each class
      for c in range(self.num_classes):

        # compute the probability of each label and append to to y_prob list
        label_c_prob = np.sum(y==self.y_unique_vals[c]) / y.size
        self.y_prob.append(label_c_prob)

        # compute probability of each feature
        for f in range(self.num_features):
            sample = np.array(X[y == self.y_unique_vals[c], f])
            # if continous - compute mean and variance
            if self.feature_types[f] == 'r':
                self.prob_vec_r[c, f] = self.cont.estimate(sample)
            # if bi_class - compute probability of success p
            if self.feature_types[f] == 'b':
                self.prob_vec_b[c, f] = self.bin.estimate(sample)
            # if categorical - compute probability of each class in a vector in this shape [p1, p2, p3, ...........]
            if self.feature_types[f] == 'c':
                self.prob_vec_c[c, f] = self.cat.estimate(sample)

    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):
        
        # p(x | class) * p(class) (numerator)
        numerator = np.ones((X.shape[0], self.num_classes)) * np.log(np.array(self.y_prob)+1e-9)
        # computing the numerator by summing the log of probabilities instead of product of probabilities
        # this will happen to each class so we loop num_classes times
        for c in range(self.num_classes):
        # loop over each feature we have and call get_probability func on each
            for f in range(self.num_features):
                if self.feature_types[f] == 'r':
                    numerator[:, c] += np.log(self.cont.get_probability(X[:, f], self.prob_vec_r[c, f]))
                if self.feature_types[f] == 'b':
                    numerator[:, c] += np.log(self.bin.get_probability(X[:, f], self.prob_vec_b[c, f]))
                if self.feature_types[f] == 'c':
                    numerator[:, c] += np.log(self.cat.get_probability(X[:, f], self.prob_vec_c[c, f]))
        # the predicted class is the argmax of numerator.
        # return self.y_unique_vals[np.argmax(numerator , axis=1)]
        return np.argmax(numerator , axis=1)
