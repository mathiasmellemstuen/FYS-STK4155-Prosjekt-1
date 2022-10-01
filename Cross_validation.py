from tkinter import W
import numpy as np
from FrankeFunction import FrankeFunctionNoised
from design_matrix import create_design_matrix
from sklearn.model_selection import KFold
from mean_square_error import MSE
import matplotlib.pyplot as plt
from linear_model import LinearModel, LinearModelType


def calculate_stats_with_crossvalidation(X, z, k, linear_model : LinearModel): 

    k_fold = KFold(n_splits=k)

    current_error_values = []

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

    return error
