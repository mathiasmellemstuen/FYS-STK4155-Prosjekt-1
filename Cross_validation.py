from tkinter import W
import numpy as np
from FrankeFunction import FrankeFunctionNoised
from design_matrix import create_design_matrix
from sklearn.model_selection import KFold
from mean_square_error import MSE
import matplotlib.pyplot as plt
from linear_model import LinearModel, LinearModelType


def calculate_stats_with_crossvalidation(X, z, k, linear_model): 

    k_fold = KFold(n_splits=k)

    current_error_values = []
    current_bias_values = []
    current_variance_values = []

    for train_inds, test_inds in k_fold.split(X):

        X_train = X[train_inds]
        X_test = X[test_inds]
        z_train = z[train_inds]
        z_test = z[test_inds]
        
        linear_model.fit(X_train, z_train)
        y_tilde = linear_model.predict(X_test).ravel()

        error = np.mean(MSE(z_test, y_tilde))
        current_error_values.append(error)
    
    
    # TODO: Return bias and variance
    error = np.mean(current_error_values)
    bias = None
    variance = None
    return error, bias, variance


if __name__ == "__main__": 

    np.random.seed(1234)

    # Making data
    x = np.arange(0, 1, 0.075)
    y = np.arange(0, 1, 0.075)
    x, y = np.meshgrid(x,y)
    z = FrankeFunctionNoised(x, y, 0.01)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    k = 15
    max_degree = 11

    lm = LinearModel(LinearModelType.OLS)
    mse_values = np.zeros(max_degree)

    for degree in range(0, max_degree): 
        X = create_design_matrix(x, y, degree)
        error, bias, variance = calculate_stats_with_crossvalidation(X, z, k, lm)
        mse_values[degree] = error

    plt.figure()
    plt.plot(np.arange(0, degree + 1, 1), mse_values)
    plt.show()