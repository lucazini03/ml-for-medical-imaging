from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_knn_roc_curves(X_train, X_test, y_train, y_test, k_values, scaler=None):
    """
    Plot ROC curves for k-NN classifier with different k values
    
    Parameters:
    X_train: Training features
    X_test: Test features  
    y_train: Training labels
    y_test: Test labels
    k_values: List of k values to test
    scaler: Scaler object (default: StandardScaler)
    """

    # use StandardScaler if no scaler is provided
    if scaler is None:
        scaler = StandardScaler()
    
    # create a figure for the ROC curves
    plt.figure(figsize=(10, 8))
    
    # store AUC values for each k
    auc_values = {}
    
    # iterate over different k values
    for k in k_values:
        # Create a k-NN classifier with the current k value
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        
        # Create the pipeline with scaler and classifier
        model_knn = Pipeline([
            ("scaler", scaler),
            ("knn", knn)
        ])
        
        # train the model using the training data
        model_knn.fit(X_train, y_train)
        
        # get prediction probabilities for the positive class
        y_score = model_knn.predict_proba(X_test)[:, 1] # make sure to use predict_proba instead of predict for ROC curve
        
        # calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
        
        # calculate the AUC
        roc_auc = auc(fpr, tpr)
        auc_values[k] = roc_auc
        
        # plot the ROC curve for this k value
        plt.plot(fpr, tpr, lw=2, label=f'k = {k} (AUC = {roc_auc:.3f})')
    
    # plot the diagonal line (random classifier baseline)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier (AUC = 0.500)')
    
    # Configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for k-NN with Different k Values')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # display the plot
    plt.show()
    