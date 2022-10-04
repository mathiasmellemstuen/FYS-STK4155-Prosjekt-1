from sample_data import create_data_samples, DataSamplesType
import matplotlib.pyplot as plt
import numpy as np
from linear_model import LinearModel, LinearModelType
from sklearn.model_selection import train_test_split
from bootstrap import calculate_stats_with_bootstrap
from FrankeFunction import FrankeFunctionNoised

if __name__ == "__main__": 

    np.random.seed(1234)

    n_bootstraps = 500
    start_degree = 0
    max_degree = 14

    # Making data
    
    real_data = True
    
    if not real_data:
        name_file = "bootstrap.pdf"
        x, y, z = create_data_samples(DataSamplesType.TEST)
    else:
        name_file = "bootstrap_Real.pdf"
        x, y, z = create_data_samples(DataSamplesType.REAL, real_data_n=100)

    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    lm = LinearModel(LinearModelType.OLS)

    error = np.zeros(max_degree)
    bias = np.zeros(max_degree)
    variance = np.zeros(max_degree)
    polydegree = np.zeros(max_degree)

    for degree in range(start_degree, max_degree):
        polydegree[degree], error[degree], bias[degree], variance[degree] = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, lm)

    plt.plot(polydegree, error, label=r'Error')
    plt.plot(polydegree, bias, label=r'bias')
    plt.plot(polydegree, variance, label=r'Variance')
    plt.ylabel(r"MSE, Variance, $Bias^2$")
    plt.xlabel(r"Polynomial degree")
    plt.legend()
    plt.savefig(f"figures\{name_file}")
    plt.show()