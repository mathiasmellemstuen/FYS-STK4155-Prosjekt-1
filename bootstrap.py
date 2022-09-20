import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import Ridge

from FrankeFunction import make_X, MSE, FrankeFunctionNoised, calc_beta, create_X, FrankeFunction


def Frankevalues(x, noise=0.01):
    length = x.shape[0]
    y = np.zeros(length)
    for i in range(length):
        y[i] = FrankeFunction(x[i,0], x[i,1]) + np.random.normal(0, noise)
    return y


np.random.seed(522553)

#x = np.arange(0, 1, 0.1)
#y = np.arange(0, 1, 0.1)
#x, y = np.meshgrid(x,y)

n = 500
n_bootstraps = 100
degree = 5  # A quite high value, just to show.
noise = 0.01

x = np.random.rand(n)
y = np.random.rand(n)

z = FrankeFunctionNoised(x,y,noise)

# Combine x transformation and model into one operation.
# Not neccesary, but convenient.
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)


# The following (m x n_bootstraps) matrix holds the column vectors y_pred
# for each bootstrap iteration.
for p in range(1,degree+1):

    model = make_pipeline(PolynomialFeatures(degree=p), LinearRegression(fit_intercept=False))
    model = LinearRegression(fit_intercept=False)
    #model = make_pipeline(PolynomialFeatures(degree=p), Ridge(alpha=1))

    #model = Ridge(alpha=1)
    X_train = make_X(x_train,y_train,p)
    X_test = make_X(x_test,y_test,p)

    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)
    #X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    #Scaling
    #X_train = (X_train - np.mean(X_train))/np.std(X_train)
    #X_test = (X_test - np.mean(X_test))/np.std(X_test)

    #z_train = (z_train - np.mean(z_train))/np.std(z_train)
    #z_test = (z_test - np.mean(z_test))/np.std(z_test)

    y_pred = np.empty((z_test.shape[0], n_bootstraps))
    print(y_pred.shape, z_test.shape)
    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        #print(i)
        # Evaluate the new model on the same test data each time.
        model.fit(X_,z_)
        #print(model.predict(X_test).shape)
        y_pred[:, i] = model.predict(X_test)

    length = len(z_test)
    z_test = np.repeat(z_test, n_bootstraps, axis=0)
    z_test = z_test.reshape((length, n_bootstraps))

    # Note: Expectations and variances taken w.r.t. different training
    # data sets, hence the axis=1. Subsequent means are taken across the test data
    # set in order to obtain a total value, but before this we have error/bias/variance
    # calculated per data point in the test set.
    # Note 2: The use of keepdims=True is important in the calculation of bias as this 
    # maintains the column vector form. Dropping this yields very unexpected results.

    #error = np.mean( np.mean((z_test - y_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )
    print("degree = ", n)
    #print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    #print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

#plt.plot(x[::5, :], y[::5, :], label='f(x)')
#plt.scatter(X_test, np.mean(y_pred, axis=1), label='Pred')
#plt.legend()
#plt.show()