from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from FrankeFunction import make_X, MSE, FrankeFunctionNoised, calc_beta, create_X

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

"""
def Frankevalues(x, noise=0.01):
    length = x.shape[0]
    y = np.zeros(length)
    for i in range(length):
        y[i] = FrankeFunction(x[i,0], x[i,1]) + np.random.normal(0, noise)
    return y
"""

np.random.seed(1234)

nsamples = 100
start_degree = 0
maxdegree = 11


# Make data set.
x = np.arange(0, 1, 0.075)
y = np.arange(0, 1, 0.075)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x,y,0.01).ravel()
x = x.ravel()
y = y.ravel()


"""np.random.shuffle(x)
np.random.shuffle(y)
np.random.shuffle(z)
x = np.array_split(z, k)
y = np.array_split(z, k)
z = np.array_split(z, k)"""


poly = PolynomialFeatures(degree = 6)
degrees = 7
# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

# Initialize a KFold instance
k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))

i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)
    j = 0
    for train_inds, test_inds in kfold.split(x):
        xtrain = x[train_inds]
        ytrain = y[train_inds]
        ztrain = z[train_inds]

        xtest = x[test_inds]
        ytest = y[test_inds]
        ztest = z[test_inds]


        #Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        #ridge.fit(Xtrain, ytrain[:, np.newaxis])

        #Xtest = poly.fit_transform(xtest[:, np.newaxis])
        #ypred = ridge.predict(Xtest)

        Xtrain = create_X(xtrain, ytrain, degrees)
        Xtest = create_X(xtest, ytest, degrees)
        beta = calc_beta(Xtrain, ztrain)

        ridge.fit(Xtrain, ztrain)
        zpred = ridge.predict(Xtest)
        scores_KFold[i,j] = np.sum((zpred - ztest[:, np.newaxis])**2)/np.size(zpred)

        j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    #X = poly.fit_transform(x[:, np.newaxis])
    #estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)
    X = create_X(x,y,degrees)
    #ridge.fit(X,z)
    estimated_mse_folds = cross_val_score(ridge, X, z[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot and compare the slightly different ways to perform cross-validation

plt.figure()

#plt.plot(np.log10(lambdas), estimated_mse_sklearn*50, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()