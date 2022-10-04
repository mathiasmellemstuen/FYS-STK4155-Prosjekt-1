import numpy as np
import matplotlib.pyplot as plt
from FrankeFunction import FrankeFunctionNoised
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from mean_square_error import MSE


def plot_MSE_variance(x,y):
    num = 14
    val = 0.1
    mse_arr_train = np.zeros(num-1)
    mse_arr_test = np.zeros(num-1)
    for i in range(1,num):
        z = FrankeFunctionNoised(x,y, val)
        z = z.ravel()
        X = create_design_matrix(x,y, i)
        X_train, X_test, z_train, z_test= train_test_split(X, z, test_size=0.2)
        #Scaling
        #X_train = (X_train - np.mean(X_train))/np.std(X_train)
        #X_test = (X_test - np.mean(X_test))/np.std(X_test)
        #z_train = (z_train - np.mean(z_train))/np.std(z_train)
        #z_test = (z_test - np.mean(z_test))/np.std(z_test)
        
        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        mse_arr_train[i-1] = MSE(z_train, z_tilde_train)
        mse_arr_test[i-1] = MSE(z_test, z_tilde_test)

    n_arr = np.array([i for i in range(1,num)])
    plt.plot(n_arr, mse_arr_train, label= "MSE_train")
    plt.plot(n_arr, mse_arr_test, label= "MSE_test")
    plt.xlabel(r"Polynomials")
    plt.ylabel(r"MSE")
    plt.legend()
    plt.savefig("MSE_test_train.pdf")
    plt.show()


if __name__ == "__main__":
    np.random.seed(12)
    x = np.arange(0, 1, 0.01)  
    y = np.arange(0, 1, 0.01)
    x, y = np.meshgrid(x,y)
    plot_MSE_variance(x,y)
