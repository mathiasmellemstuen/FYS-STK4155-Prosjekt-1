from re import A
import numpy as np
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sample_data import create_data_samples, DataSamplesType
from linear_model import LinearModel, LinearModelType
from Cross_validation import calculate_stats_with_crossvalidation
from bootstrap import calculate_stats_with_bootstrap

if __name__ == "__main__": 

    # Making data
    np.random.seed(1234)

    real_data = True
    
    if not real_data:
        name_file = "KFOLD_"
        x, y, z = create_data_samples(DataSamplesType.TEST)

    else:
        name_file = "KFOLD_Real_"
        x, y, z = create_data_samples(DataSamplesType.REAL, real_data_n=100)
      
    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)

    n_bootstraps = 500
    k = 5
    max_degree = 12

    lm = LinearModel(LinearModelType.OLS)

    crossvalidation_error = np.zeros(max_degree)
    bootstrap_error = np.zeros(max_degree)

    # For bootstrap
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    x_train = (x_train - np.mean(x_train))/np.std(x_train)
    x_test = (x_test - np.mean(x_test))/np.std(x_test)
    y_train = (y_train - np.mean(y_train))/np.std(y_train)
    y_test = (y_test - np.mean(y_test))/np.std(y_test)

    for degree in range(0, max_degree): 
        X = create_design_matrix(x, y, degree)
        crossvalidation_error[degree] = calculate_stats_with_crossvalidation(X, z, k, lm)

        bootstrap_degree, bootstrap_error[degree], bootstrap_bias, bootstrap_variance = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, lm)
        
    plt.figure()
    plt.plot(np.arange(0, degree + 1, 1), crossvalidation_error, label="Crossvalidation error")
    plt.plot(np.arange(0, degree + 1, 1), bootstrap_error, label="Bootstrap error")
    plt.legend()
    plt.xlabel(r"Polynomials")
    plt.ylabel(r"MSE")
    plt.title(r"Comparison of crossvalidation and bootstrap for $k=5$ folds")
    plt.tight_layout()
    plt.savefig(f"figures\{name_file}{str(k)}.pdf")
    plt.show()