import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


def polynomial_regression_grid_search(X_train, X_test, y_train, y_test, degrees):
    """
    Perform polynomial regression with grid search cross-validation.
    
    Parameters:
    X_train, X_test: Training and test features
    y_train, y_test: Training and test targets
    max_degree: Maximum polynomial degree to test (default=8)
    
    Returns:
    grid_search: Fitted GridSearchCV object
    """
    # Create pipeline
    pipeline = Pipeline([
        #('scaler', StandardScaler()), MIGHT HELP, ACCORDING TO THE VERSION OF SCIKITLEARN. IN MY VERSION IT DOES NOT HELP
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression())
    ])

    # Define parameter grid
    param_grid = {'poly__degree': degrees}

    # Grid search with cross-validation using R^2 score (regression)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Best model performance
    print(f"Best polynomial degree: {grid_search.best_params_['poly__degree']}")
    print(f"Best CV R^2: {grid_search.best_score_:.4f}")

    # Test set performance
    test_r2 = grid_search.score(X_test, y_test)
    print(f"Test set R^2: {test_r2:.4f}")
    
    return grid_search

def plot_learning_curve(grid_search, degrees):
    """
    Plot learning curve showing cross-validation R^2 score vs polynomial degree.
    
    Parameters:
    grid_search: Fitted GridSearchCV object
    max_degree: Maximum polynomial degree tested
    """
    cv_scores = [grid_search.cv_results_['mean_test_score'][i] for i in range(len(degrees))]

    plt.figure(figsize=(8, 6))
    plt.plot(degrees, cv_scores, 'o-', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation R^2 Score')
    plt.title('Cross-Validation R^2 Score vs Polynomial Order')
    plt.grid(True)
    plt.xticks(degrees)
    plt.show()

def plot_dataset_regression(X, y, model):
    """
    Plot dataset and regression curve.
    
    Parameters:
    X, y: Features and targets
    model: Fitted regression model
    """
    plt.figure(figsize=(8, 6))
    plt.plot(X, y, 'r.', markersize=12, label='Data')
    
    # Generate points for regression curve
    X_fit = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    y_fit = model.predict(X_fit)
    
    plt.plot(X_fit, y_fit, 'b-', label='Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.show()