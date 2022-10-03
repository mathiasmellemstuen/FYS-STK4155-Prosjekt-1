from re import A
import numpy as np
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunctionNoised, create_data_samples_with_franke
import matplotlib.pyplot as plt
from linear_model import LinearModel, LinearModelType
from Cross_validation import calculate_stats_with_crossvalidation
from bootstrap import calculate_stats_with_bootstrap

if __name__ == "__main__": 

    np.random.seed(1234)

    # Making data
    x, y, z = create_data_samples_with_franke()

    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)

    n_bootstraps = 500
    k = 18
    max_degree = 11

    lm = LinearModel(LinearModelType.OLS)

    crossvalidation_error = np.zeros(max_degree)
    bootstrap_error = np.zeros(max_degree)

    # For bootstrap
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    for degree in range(0, max_degree): 
        X = create_design_matrix(x, y, degree)
        crossvalidation_error[degree] = calculate_stats_with_crossvalidation(X, z, k, lm)

        bootstrap_degree, bootstrap_error[degree], bootstrap_bias, bootstrap_variance = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, lm)
        
    plt.figure()
    plt.plot(np.arange(0, degree + 1, 1), crossvalidation_error, label="Crossvalidation error")
    plt.plot(np.arange(0, degree + 1, 1), bootstrap_error, label="Bootstrap error")
    plt.legend()
    plt.xlabel("Polynomials")
    plt.ylabel("MSE")
    plt.title("Comparison of crossvalidation and bootstrap")
    plt.savefig("figures/crossvalidation.pdf")
    plt.show()