import numpy as np
from mean_square_error import MSE
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from FrankeFunction import FrankeFunctionNoised, create_data_samples_with_franke
import matplotlib.pyplot as plt
from linear_model import LinearModel, LinearModelType
from cross_validation import calculate_stats_with_crossvalidation

if __name__ == "__main__": 

    np.random.seed(1234)

    # Making data
    x, y, z = create_data_samples_with_franke()

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    k = 15
    max_degree = 11

    lm = LinearModel(LinearModelType.OLS)
    mse_values = np.zeros(max_degree)

    for degree in range(0, max_degree): 
        X = create_design_matrix(x, y, degree)
        error, bias, variance = calculate_stats_with_crossvalidation(X, z, k, lm)
        mse_values[degree] = error

    plt.figure()
    plt.plot(np.arange(0, degree + 1, 1), mse_values)
    plt.show()