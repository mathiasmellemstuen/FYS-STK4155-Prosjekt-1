import numpy as np
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunctionNoised, create_data_samples_with_franke
from linear_model import LinearModel, LinearModelType
import matplotlib.pyplot as plt
from bootstrap import calculate_stats_with_bootstrap
from Cross_validation import calculate_stats_with_crossvalidation

if __name__ == "__main__": 

    np.random.seed(1234)

    x, y, z = create_data_samples_with_franke()
    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)

    max_polynomial = 10
    n_bootstraps = 500
    nlambdas = 10
    k = 10

    heatmap_bootstrap = np.zeros((max_polynomial, nlambdas))
    heatmap_crossvalidation = np.zeros((max_polynomial, nlambdas))

    lambdas = np.logspace(-3, 1, nlambdas)

    lm = LinearModel(LinearModelType.RIDGE)

    for current_polynomial in range(1, max_polynomial + 1):
        X = create_design_matrix(x, y, current_polynomial)
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)
        for i in range(nlambdas):
            lam = lambdas[i]
            lm.set_lambda(lam)
            bootstrap_degree, bootstrap_error, bootstrap_bias, bootstrap_variance = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, current_polynomial, lm)
            heatmap_bootstrap[current_polynomial - 1][i] = bootstrap_error

            crossvalidation_error = calculate_stats_with_crossvalidation(X, z, k, lm)
            heatmap_crossvalidation[current_polynomial - 1][i] = crossvalidation_error

    # Plotting heatmap for bootstrap
    plt.figure()
    plt.title(r"Heatmap of polynomial degree over $\lambda$ with bootstrap resampling")
    plt.imshow(heatmap_bootstrap, cmap="inferno")
    plt.xlabel(r"$\lambda$")
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()

    # Plotting heatmap for crossvalidation
    plt.figure()
    plt.title(r"Heatmap of polynomial degree over $\lambda$ with crossvalidation resampling")
    plt.imshow(heatmap_crossvalidation, cmap="inferno")
    plt.xlabel(r"$\lambda$")
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()

    plt.show()