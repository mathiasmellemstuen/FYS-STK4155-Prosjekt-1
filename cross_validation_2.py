from tkinter import W
import numpy as np
from FrankeFunction import FrankeFunctionNoised
from ordinary_least_squares import calc_beta
from design_matrix import create_design_matrix
from sklearn.model_selection import KFold
from mean_square_error import MSE
import matplotlib.pyplot as plt

# Making data
x = np.arange(0, 1, 0.075)
y = np.arange(0, 1, 0.075)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x, y, 0.01)
x = x.ravel()
y = y.ravel()
z = z.ravel()

k = 15
k_fold = KFold(n_splits=k)
max_degree = 11

mse_values = np.zeros(max_degree)

for degree in range(0, max_degree):

    current_mse_values = []
    X = create_design_matrix(x, y, degree)

    for train_inds, test_inds in k_fold.split(X):

        X_train = X[train_inds]
        X_test = X[test_inds]
        z_train = z[train_inds]
        z_test = z[test_inds]
        
        beta = calc_beta(X_train, z_train)
        y_tilde = X_test @ beta
        current_mse_values.append(np.mean(MSE(z_test, y_tilde)))

    mse_values[degree] = np.mean(current_mse_values)


plt.figure()
plt.plot(np.arange(0, degree + 1, 1), mse_values)
plt.show()