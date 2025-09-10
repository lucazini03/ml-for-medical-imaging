import numpy as np
import matplotlib.pyplot as plt


def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def regression(weights, X) :
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    y_pred = np.dot(X, weights)
    return y_pred


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

