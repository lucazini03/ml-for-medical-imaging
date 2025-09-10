import scipy
import numpy as np
import matplotlib.pyplot as plt


def knn(x_data, y_data, k):
    """
    Performs k-nearest neighbor algorithm. First calculates the distances between all nodes. 
    Then finds the nearest k nodes and takes majority vote of the class with division [0.0,0.5), [0.5,1.0].

    Args:
        x_data (array): Matrix with the nodes and features.
        y_data (array): Matrix with the assigned classes of the nodes
        k (int): Number of nearest neighbors.
    
    Returns:
        accuracy (float): Float describing the accuracy of the knn class predictions.
    """
    range_data = range(len(x_data))

    # calculate distances
    distance_matrix = scipy.spatial.distance.cdist(x_data, x_data, metric='euclidean') # matrix of distance between each X_data, diagonal is 0
    # get k+1 nearest value and then save the values of those lower
    k_distances = [distance_matrix[i][distance_matrix[i] < [np.sort(distance_matrix[i])][0][k]] for i in range_data]
    # get index corresponding to k nearest values
    k_indices = np.array([[np.where(distance_matrix[i] == k_distances[i][j])[0][0] for j in range(k)] for i in range_data])
    # sum corresponding y values and divide by k
    y_normalized = [sum([y_data[k_indices[i][j]][0] for j in range(k)])/k for i in range_data]
    # assign classes [0,1] based on normalized y
    y_pred = [[0 if y < 0.5 else 1] for y in y_normalized]

    accuracy = list(y_pred == y_data).count(True) / len(y_data)

    return accuracy


def plot_knn(k_list, train_accuracy_list, test_accuracy_list):
    """
    Visualizes the train and test accuracy over the different values of k.

    Args:
        k_list (list): List that contains the k-values.
        train_accuracy_list (list): List that contains the accuracies of the train data.
        test_accuracy_list (list): List that contains the accuracies of the test data.

    Returns None    
    """
    fig,ax = plt.subplots(1,2,sharex=True, figsize=(15,5))
    ax[0].grid()
    ax[0].plot(k_list, train_accuracy_list, c="royalblue", label="Train")
    ax[0].plot(k_list, test_accuracy_list, c="violet", label ="Test")
    ax[0].set_xlabel("k value")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim([0,1])
    ax[0].set_title("Accuracy vs k")
    ax[0].legend()

    ax[1].grid() # plot zoomed in from y 0.9 to y 1. 
    ax[1].plot(k_list, train_accuracy_list, c="royalblue", label="Train")
    ax[1].plot(k_list, test_accuracy_list, c="violet", label ="Test")
    ax[1].set_xlabel("k value")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim([0.9,1])
    ax[1].set_title("Accuracy vs k, zoomed in")
    ax[1].legend()

    plt.show()
    return None