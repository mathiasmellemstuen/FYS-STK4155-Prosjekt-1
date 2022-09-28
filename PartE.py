import numpy as np
from ridge_regression import ridge_beta
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunction, FrankeFunctionNoised
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x, y, 0.01)
max_polynomial = 5

nlambdas = 10
MSE_values_predict = np.zeros((max_polynomial, nlambdas))
MSE_values_train = np.zeros((max_polynomial, nlambdas))

lambdas = np.logspace(-3, 1, nlambdas)

for current_polynomial in range(1, max_polynomial + 1):
    X = create_design_matrix(x, y, current_polynomial)
    X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)
    for i in range(nlambdas):
        lam = lambdas[i]
        Ridge_Beta = ridge_beta(X_train, z_train, lam)

        z_tilde = X_train @ Ridge_Beta
        z_predict = X_test @ Ridge_Beta

        MSE_values_predict[current_polynomial-1][i] = MSE(z_test, z_predict)
        MSE_values_train[current_polynomial-1][i] = MSE(z_train, z_tilde)

plt.figure()
for i in range(max_polynomial):
    plt.plot(np.log10(lambdas), MSE_values_train[i], label = 'MSE train, pol = ' + str(i+1))
    plt.plot(np.log10(lambdas), MSE_values_predict[i], label = 'MSE test, pol = ' + str(i+1))

plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.show()