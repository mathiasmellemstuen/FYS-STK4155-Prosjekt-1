from FrankeFunction import FrankeFunctionNoised
from mean_square_error import MSE
from r2_score import R2score
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from linear_model import LinearModel, LinearModelType

if __name__ == "__main__":

    # Making data, 20 samples
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x,y)
    z = FrankeFunctionNoised(x, y, 0.01)

    # 20% of data is used for test, 80% training
    test_size = 0.2

    max_polynomial = 5

    mse_values_test = []
    mse_values_train = []
    r2_score_values_test = []
    r2_score_values_train = []
    
    lm = LinearModel(LinearModelType.OLS)

    # Doing calculations for each polynomial
    for current_polynominal in range(1, max_polynomial + 1): 

        X = create_design_matrix(x, y, current_polynominal)
        X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=test_size)
        
        # Scaling of the data
        X_train = (X_train - np.mean(X_train))/np.std(X_train)
        X_test = (X_test - np.mean(X_test))/np.std(X_test)
        y_train = (y_train - np.mean(y_train))/np.std(y_train)
        y_test = (y_test - np.mean(y_test))/np.std(y_test)

        # Using training data to create beta
        lm.fit(X_train, y_train)


        # Using beta and test data to pretict y
        y_tilde_test = lm.predict(X_test)
        y_tilde_train = lm.predict(X_train)

        # Calculating mean square error and R2 score for each polynomial
        mse_values_test.append(np.mean(MSE(y_test, y_tilde_test)))
        r2_score_values_test.append(np.mean(R2score(y_test, y_tilde_test)))
        mse_values_train.append(np.mean(MSE(y_train, y_tilde_train)))
        r2_score_values_train.append(np.mean(R2score(y_train, y_tilde_train)))

    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=5.0)

    # Plotting mean square error for each polynomial
    axs[0].plot(np.arange(1, max_polynomial + 1, 1), mse_values_test, label="MSE test")
    axs[0].plot(np.arange(1, max_polynomial + 1, 1), mse_values_train, label="MSE train")
    axs[0].legend()
    axs[0].set_title("MSE")
    axs[0].set_xlabel("Polynomials")
    axs[0].set_ylabel("MSE")
    
    # Plotting R2 score for each polynomial
    axs[1].plot(np.arange(1, max_polynomial + 1, 1), r2_score_values_test, label="R2 score test")
    axs[1].plot(np.arange(1, max_polynomial + 1, 1), r2_score_values_train, label="R2 score train")
    axs[1].legend()
    axs[1].set_title("R2 score")
    axs[1].set_xlabel("Polynomials")
    axs[1].set_ylabel("R2 score")

    plt.show()