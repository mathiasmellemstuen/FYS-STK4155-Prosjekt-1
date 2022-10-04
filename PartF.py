import numpy as np
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from sample_data import create_data_samples, DataSamplesType
from linear_model import LinearModel, LinearModelType
import matplotlib.pyplot as plt
from bootstrap import calculate_stats_with_bootstrap
from Cross_validation import calculate_stats_with_crossvalidation

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__": 

    np.random.seed(1234)

    real_data = False
    
    if not real_data:
        name_file1 = "figures\Heatmap_MSE_Bootstrap_Lasso.pdf"
        name_file2 = "figures\Heatmap_MSE_Crossvalidation_Lasso.pdf"
        name_file3 = "figures\Heatmap_Variance_Bootstrap_Lasso.pdf"
        name_file4 = "figures\Heatmap_Bias_Bootstrap_Lasso.pdf"
        x, y, z = create_data_samples(DataSamplesType.TEST)

    else:
        name_file1 = "figures\Heatmap_MSE_Bootstrap_Lasso_Real.pdf"
        name_file2 = "figures\Heatmap_MSE_Crossvalidation_Lasso_Real.pdf"
        name_file3 = "figures\Heatmap_Variance_Bootstrap_Lasso_Real.pdf"
        name_file4 = "figures\Heatmap_Bias_Bootstrap_Lasso_Real.pdf"
        x, y, z = create_data_samples(DataSamplesType.REAL, real_data_n=100)

    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)

    max_polynomial = 13
    n_bootstraps = 500
    nlambdas = 13
    k = 10

    heatmap_bootstrap_variance = np.zeros((max_polynomial, nlambdas))
    heatmap_bootstrap_bias = np.zeros((max_polynomial, nlambdas))

    heatmap_bootstrap = np.zeros((max_polynomial, nlambdas))
    heatmap_crossvalidation = np.zeros((max_polynomial, nlambdas))

    lambdas = np.logspace(-3, 1, nlambdas)

    lm = LinearModel(LinearModelType.LASSO)

    for current_polynomial in range(1, max_polynomial + 1):
        X = create_design_matrix(x, y, current_polynomial)
        x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)
        
        for i in range(nlambdas):
            lam = lambdas[i]
            lm.set_lambda(lam)
            bootstrap_degree, bootstrap_error, bootstrap_bias, bootstrap_variance = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, current_polynomial, lm)
            crossvalidation_error = calculate_stats_with_crossvalidation(X, z, k, lm)

            heatmap_bootstrap[current_polynomial - 1][i] = bootstrap_error
            heatmap_bootstrap_variance[current_polynomial - 1][i] = bootstrap_variance
            heatmap_bootstrap_bias[current_polynomial - 1][i] = bootstrap_bias
            heatmap_crossvalidation[current_polynomial - 1][i] = crossvalidation_error

    # Plotting heatmap for bootstrap
    plt.figure()
    plt.title(r"Heatmap of MSE with bootstrap resampling and lasso regression")
    plt.imshow(heatmap_bootstrap, cmap="inferno")
    plt.xlabel(r"$\alpha$")
    plt.gcf().autofmt_xdate()
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()
    plt.savefig(f"{name_file1}")

    # Plotting heatmap for crossvalidation
    plt.figure()
    plt.title(r"Heatmap of MSE with crossvalidation resampling and lasso regression")
    plt.imshow(heatmap_crossvalidation, cmap="inferno")
    plt.xlabel(r"$\alpha$")
    plt.gcf().autofmt_xdate()
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()
    plt.savefig(f"{name_file2}")

    # Plotting heatmap for variance, bootstrap
    plt.figure()
    plt.title(r"Heatmap of variance with bootstrap resampling and lasso regression")
    plt.imshow(heatmap_bootstrap_variance, cmap="inferno")
    plt.xlabel(r"$\alpha$")
    plt.gcf().autofmt_xdate()
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()
    plt.savefig(f"{name_file3}")

    # Plotting heatmap for bias, bootstrap
    plt.figure()
    plt.title(r"Heatmap of bias with bootstrap resampling and lasso regression")
    plt.imshow(heatmap_bootstrap_bias, cmap="inferno")
    plt.xlabel(r"$\alpha$")
    plt.gcf().autofmt_xdate()
    lambdas = np.around(lambdas, decimals=5)
    plt.xticks(np.arange(0, nlambdas), labels=lambdas)
    plt.ylabel("Polynomial degree")
    plt.colorbar()
    plt.savefig(f"{name_file4}")

    plt.show()