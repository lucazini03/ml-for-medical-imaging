import matplotlib.pyplot as plt

def plot_mse_vs_k(k_list, mse_list, mse_lin=None, title="Mean Squared Error vs k value"):
    """
    Plots the Mean Squared Error (MSE) for different k values.
    
    Args:
        k_list (array-like): List of k values.
        mse_list (array-like): List of MSE values corresponding to k_list.
        mse_lin (float, optional): MSE of linear regression to plot as a reference line.
        title (str, optional): Title for the plot.
    
    Returns None
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid()
    ax.plot(k_list, mse_list, 'b-', label='k-NN Regression')
    if mse_lin is not None:
        ax.axhline(y=mse_lin, color='r', linestyle='--', label='Linear Regression')
    ax.set_xlabel("k value")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return None