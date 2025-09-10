def compute_class_stats(X, y):
    means = []
    stds = []
    for idx in range(X.shape[1]):
        data_feature = X[:, idx]
        data_feature_benign = data_feature[y == 1]
        data_feature_malignant = data_feature[y == 0]
        means.append([np.mean(data_feature_malignant), np.mean(data_feature_benign)])
        stds.append([np.std(data_feature_malignant), np.std(data_feature_benign)])
    return np.array(means), np.array(stds)

def plot_class_conditional(X, y, feature_names, means, stds):
    n_features = X.shape[1]
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for idx in range(n_features):
        data_feature = X[:, idx]
        data_feature_benign = data_feature[y == 1]
        data_feature_malignant = data_feature[y == 0]
        x_range = np.linspace(data_feature.min(), data_feature.max(), 200)
        # Gaussians
        gaussian_benign = (1/(stds[idx,1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - means[idx,1])/stds[idx,1])**2)
        gaussian_malignant = (1/(stds[idx,0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - means[idx,0])/stds[idx,0])**2)
        plt.subplot(n_rows, n_cols, idx+1)
        plt.hist(data_feature_benign, bins=30, density=True, alpha=0.5, label='Benign')
        plt.hist(data_feature_malignant, bins=30, density=True, alpha=0.5, label='Malignant')
        plt.plot(x_range, gaussian_benign, color='blue', label='Gaussian Benign')
        plt.plot(x_range, gaussian_malignant, color='red', label='Gaussian Malignant')
        plt.title(feature_names[idx])
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    plt.tight_layout()
    plt.show()