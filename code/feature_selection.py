from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np


def lasso_cross_validation(X_train, X_test, y_train, y_test, n_cross_validation):
    """
    Performs cross validation to get the best alpha for Lasso and shows the remaining features and their coefficients.

    Args:
        X_train (DataFrame): Df with the features of the training data.
        y_train (DataFrame): Df with the target values of the training data.
        X_test (DataFrame): Df with the features of the test data.
        y_test (DataFrame): Df with the target values of the test data.
        n_cross_validation (int): Amount of partitions to use in cross validation.
    
    Returns None
    """
    # initialize alpha range and model
    params = {"alpha":np.arange(0.00001, 1, 0.01)} # 100 steps
    lasso = Lasso(tol=1e-2) # increased tolerance to aid convergence

    # GridSearchCV with model, params and folds.
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv = n_cross_validation, scoring="neg_mean_squared_error") # Change in number of folds
    lasso_cv.fit(X_train, y_train)

    print("Best training score: {}, alpha = {}".format(lasso_cv.best_score_,lasso_cv.best_params_))
    print("Corresponding test score: {}".format(lasso_cv.score(X_test,y_test)))


    # plot the coefficients of remaining features
    lasso1 = Lasso(alpha=lasso_cv.best_params_["alpha"])
    lasso1.fit(X_train, y_train)
    lasso1_coef = np.abs(lasso1.coef_)

    names = X_train.columns.values[(np.where(lasso1_coef != 0.0))]

    plt.figure(figsize=[7,5])
    plt.bar(names, lasso1_coef[np.where(lasso1_coef != 0.0)])
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Features remaining with alpha={}".format(lasso_cv.best_params_["alpha"]))
    plt.xlabel("Features")
    plt.ylabel("Coefficients")
    plt.show()

    return None