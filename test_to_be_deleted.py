import numpy as np
from FrankeFunction import FrankeFunctionNoised
from bootstrap_function import bootstrap, bootstrap_ridge

np.random.seed(1234)
n = 40
n_bootstraps = 500
start_degree = 0
maxdegree = 11
nlambdas = 10
lambdas = np.logspace(-3, 1, nlambdas)

# Make data set.
x = np.arange(0, 1, 0.075)
y = np.arange(0, 1, 0.075)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x,y,0.01)
x = x.ravel()
y = y.ravel()
z = z.ravel().reshape(-1,1)

#bootstrap(x, y, z, maxdegree, n_bootstraps)

bootstrap_ridge(x, y, z, lambdas, maxdegree, n_bootstraps)