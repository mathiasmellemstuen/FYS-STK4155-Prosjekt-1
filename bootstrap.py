import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from design_matrix import create_design_matrix
from FrankeFunction import FrankeFunctionNoised
from linear_model import LinearModel, LinearModelType

def calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, linear_model : LinearModel): 

    z_pred = np.empty((len(z_test), n_bootstraps))

    X_test = create_design_matrix(x_test,y_test, degree)
    X_train = create_design_matrix(x_train,y_train, degree)

    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        linear_model.fit(X_, z_)
        z_pred[:, i] = linear_model.predict(X_test).ravel()

    error= np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    print('Polynomial degree:', degree)
    print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

    return degree, error, bias, variance