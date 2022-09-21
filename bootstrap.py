from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import Ridge

from FrankeFunction import make_X, MSE, FrankeFunctionNoised, calc_beta, create_X, FrankeFunction

"""
def Frankevalues(x, noise=0.01):
    length = x.shape[0]
    y = np.zeros(length)
    for i in range(length):
        y[i] = FrankeFunction(x[i,0], x[i,1]) + np.random.normal(0, noise)
    return y
"""

np.random.seed(1234)

n = 40
n_bootstraps = 500
start_degree = 0
maxdegree = 11


# Make data set.
x = np.arange(0, 1, 0.075)
y = np.arange(0, 1, 0.075)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x,y,0.01)
x = x.ravel()
y = y.ravel()
z = z.ravel().reshape(-1,1)

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

#length = z_test.shape[0]*z_test.shape[1]
#z_test1 = np.repeat(z_test, n_bootstraps, axis=0)
#z_test1 = z_test1.reshape((length, n_bootstraps))

for degree in range(start_degree,maxdegree):
    #model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    #z_pred = np.empty((z_test.shape[0] * z_test.shape[1], n_bootstraps))
    z_pred = np.empty((len(z_test), n_bootstraps))

    X_test = create_X(x_test,y_test,degree)
    X_train = create_X(x_train,y_train,degree)
    
    #scaling
    #X_train = (X_train - np.mean(X_train))/np.std(X_train)
    #X_test = (X_test - np.mean(X_test))/np.std(X_test)

    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        beta = calc_beta(X_, z_)
        #y_pred[:, i] = model.fit(X_, z_).predict(X_test)
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
plt.savefig("bootstrap.png")
plt.show()