from helper.lib import *


def generate_data(n, k, num_outliers = 0, outlier_both = True):
    if outlier_both:
        n = n - (num_outliers * 2)
    nneg, npos = n // 2, n - n // 2
    if num_outliers > 0 and not outlier_both:
        nneg = nneg - num_outliers
    X_pos = np.random.uniform(1, 5, size=(npos, k))
    X_neg = np.random.uniform (-5, -1, size=(nneg, k))
    # loc = 2.0 # mean
    # scale = 0.5 # standard deviation
    # X_pos = np.random.normal(loc, scale, size=(npos, k)) + np.full((npos, k), 3.0)
    # X_neg = np.random.normal(loc, scale, size=(nneg, k)) + np.full((nneg, k), -3.0)
    if outlier_both:
        outlier_pos = np.random.uniform(-8, -3, size=(num_outliers, k))
        X_pos = np.vstack((X_pos, outlier_pos))
    if num_outliers > 0:
        outlier_neg = np.random.uniform(3, 8, size=(num_outliers, k))
        X_neg = np.vstack((X_neg, outlier_neg))
    
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    return X, y

def normalise_dataset(X, type='minmax'):
    if type == 'minmax':
        X = (X - X.min()) / (X.max() - X.min())
    elif type == 'zscore':
        X = (X - X.mean()) / X.std()
    elif type == 'unit':
        X = X / np.linalg.norm(X)
    return X

def split_dataset(X, y, split_ratio=0.8):
    n = X.shape[0]
    n_train = int(n * split_ratio)
    idx = np.random.permutation(n)
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    return X_train, y_train, X_test, y_test

def generate_batch(X, y, batch_size):
    n = X.shape[0]
    iter = n // batch_size
    idx = np.random.permutation(n)
    for i in range(iter):
        idx_batch = idx[i*batch_size:(i+1)*batch_size]
        X_batch, y_batch = X[idx_batch], y[idx_batch]
        yield X_batch, y_batch

def augment_data(X):
    n = X.shape[0]
    X_aug = np.hstack((np.ones((n, 1)), X))
    return X_aug