import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from design_matrix import create_design_matrix
from ordinary_least_squares import calc_beta
from sklearn.utils import resample

from ridge_regression import ridge_beta

def bootstrap(x, y, z, max_degree, n_bootstraps):

    np.random.seed(1234)

    error = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    variance = np.zeros(max_degree)
    polydegree = np.zeros(max_degree)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in range(max_degree):

        z_pred = np.empty((len(z_test), n_bootstraps))
        X_test = create_design_matrix(x_test,y_test,degree)
        X_train = create_design_matrix(x_train,y_train,degree)


        for i in range(n_bootstraps):
            X_, z_ = resample(X_train, z_train)
            beta = calc_beta(X_, z_)
            ans = X_test @ beta
            z_pred[:, i] = ans.ravel()

        polydegree[degree] = degree
        error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
        print('Polynomial degree:', degree)
        print('Error:', error[degree])
        print('Bias^2:', bias[degree])
        print('Var:', variance[degree])
        print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.ylabel("MSE, Variance, Bias^2")
    plt.xlabel("Polynomial degree")
    plt.legend()
    plt.show()

def bootstrap_ridge(x, y, z, lam, max_degree, n_bootstraps):

    np.random.seed(1234)
        
    error = np.zeros((max_degree, len(lam)))
    bias = np.zeros((max_degree, len(lam)))
    variance = np.zeros((max_degree, len(lam)))
    polydegree = np.zeros((max_degree, len(lam)))

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in range(max_degree):

        z_pred = np.empty((len(z_test), len(lam), n_bootstraps))
        X_test = create_design_matrix(x_test,y_test,degree)
        X_train = create_design_matrix(x_train,y_train,degree)

        for i in range(len(lam)):
            lmb = lam[i]
            for j in range(n_bootstraps):
                X_, z_ = resample(X_train, z_train)
                beta = ridge_beta(X_, z_, lmb)
                ans = X_test @ beta
                z_pred[:, i, j] = ans.ravel()

            polydegree[degree, i] = degree
            error[degree, i] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
            bias[degree, i] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance[degree, i] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
            print('Polynomial degree:', degree)
            print('Error:', error[degree])
            print('Bias^2:', bias[degree])
            print('Var:', variance[degree])
            print('{} >= {} + {} = {}'.format(error[degree], bias[degree], variance[degree], bias[degree]+variance[degree]))

    for i in range(len(lam)):
        plt.plot(lam, error[i], label='Error')
        plt.plot(lam, bias[i], label='bias')
        plt.plot(lam, variance[i], label='Variance')
    plt.ylabel("MSE, Variance, Bias^2")
    plt.xlabel("lambda")
    plt.legend()
    plt.show()



