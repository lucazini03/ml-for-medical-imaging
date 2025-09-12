import numpy as np


def lsq(X, y):
    """
    Least squares linear regression.

    Args:
        X (array): Input data matrix.
        y (array): Target vector.
    
    Returns estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


def regression(weights, X):
    """
    Performs regression on the data using the calculated weights.

    Args:
        weights (array): Coefficients.
        X (array): Input data matrix
    
    Returns predicted y    
    """
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    y_pred = np.dot(X, weights)
    return y_pred


def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error for the predicted vector and the target vector.

    Args:
        y_true (array): Target vector.
        y_pred (array): Predicted vector.

    Returns calculated MSE value
    """
    return np.mean((y_true - y_pred) ** 2)