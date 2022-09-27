import numpy as np
from FrankeFunction import FrankeFunctionNoised, calc_beta
from design_matrix import create_design_matrix
from sklearn.model_selection import KFold
from mean_square_error import MSE
import matplotlib.pyplot as plt


# Scale data
# Implement k-fold cross-validation algorithm 
# Evaluate MSE function from test folds
# Compare cross-validation MSE with bootstrap MSE

# Making data, 20 samples
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x, y, 0.01).ravel()
x.ravel()
y.ravel()

k = 5
k_fold = KFold(n_splits=k)
max_degree = 11

mse_values = np.zeros(max_degree)

for degree in range(1, max_degree):

    i = 0
    current_mse_values = []

    for train_inds, test_inds in k_fold.split(x):
        x_train = x[train_inds]
        y_train = y[train_inds]
        z_train = z[train_inds]

        x_test = x[test_inds]
        y_test = y[test_inds]
        z_test = z[test_inds]

        X_train = create_design_matrix(x_train, y_train, degree)
        X_test = create_design_matrix(x_test, y_test, degree)

        beta = calc_beta(X_train, z_train)
        y_tilde = X_test @ beta

        print(y_tilde.shape)

        current_mse_values.append(np.mean(MSE(z_test, y_tilde)))

    mse_values[degree] = np.mean(current_mse_values)

plt.figure()
plt.plot(np.arange(0, degree + 1, 1), mse_values)
plt.show()