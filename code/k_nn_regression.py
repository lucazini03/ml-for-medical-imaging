import numpy as np
import scipy.spatial


def knn_regression(x_data, y_data, x_test, k):
    """
    Performs k-nearest neighbor regression. Calculates distances between test nodes and all training nodes.
    Then finds the nearest k nodes and takes average of their target values.

    Args:
        x_data (array): Matrix with training nodes and features.
        y_data (array): Matrix with target values of training nodes
        x_test (array): Matrix with test nodes and features
        k (int): Number of nearest neighbors.
    
    Returns:
        y_pred (array): Array with predicted values for test data.
    """
    # calculate distances between test points and all training points
    distance_matrix = scipy.spatial.distance.cdist(x_test, x_data, metric='euclidean')
    
    # get indices of k nearest neighbors for each test point
    k_indices = np.argpartition(distance_matrix, k, axis=1)[:, :k]
    
    # get target values of k nearest neighbors and calculate mean
    y_neighbors = y_data[k_indices]
    y_pred = np.mean(y_neighbors, axis=1)
    
    return y_pred