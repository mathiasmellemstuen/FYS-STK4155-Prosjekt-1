from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
#x_mesh, y_mesh = np.meshgrid(x,y)
x, y = np.meshgrid(x,y)


x_noise = np.random.normal(0,1,20)
y_noise = np.random.normal(0,1,20)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

z = FrankeFunction(x,y)

#for loop for permutations
#DELETE IF NOT NEEDED
def permutations(n):
    count = 0
    perm = np.zeros((n*(n-1),2))
    for i in range(n+1):
        for j in range(n-i,-1, -1):
            if i>0 or j>0:
                perm[count,0]=i
                perm[count,1]=j

                count += 1

    return perm

def FrankeFunctionNoised(x, y, max_noise): 

    ff = FrankeFunction(x, y)
    noise = np.random.normal(0, 0.1, len(x)*len(x))
    noise = noise.reshape(len(x), len(x))
    return ff + noise

#FrankeFunctionNoised(x,y,0.1)


def make_X(x,y,n): 
    x = x.ravel()
    y = y.ravel() 
    X = np.c_[np.ones(len(x)),		  
                        x,y,
                        x**2,x*y,y**2,
                        x**3,x**2*y,x*y**2,y**3,
                        x**4,x**3*y,x**2*y**2,x*y**3,y**4,
                        x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]
    
    return X

X = make_X(x,y,5)
print(np.shape(X[0]))
print(np.shape(X))

def calc_beta(X,y):
    #beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #X = X + 0.0000001
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

print(calc_beta(X,z.ravel()))