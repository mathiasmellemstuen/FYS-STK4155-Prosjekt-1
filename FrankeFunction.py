from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import sklearn.preprocessing as sk
from sklearn.model_selection import train_test_split

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
    noise = np.random.normal(0, max_noise, len(x)*len(x))
    noise = noise.reshape(len(x), len(x))
    return ff + noise


z = FrankeFunctionNoised(x,y, 0.01)

def make_X(x,y,n = 5):
    x = x.ravel()
    y = y.ravel()
    if n == 1:
        X = np.c_[np.ones(len(x)),		  
                            x,y]
    if n == 2:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2]
    if n == 3:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3]
     
    if n == 4:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3,
                            x**4,x**3*y,x**2*y**2,x*y**3,y**4]

    if n == 5:
        X = np.c_[np.ones(len(x)),		  
                            x,y,
                            x**2,x*y,y**2,
                            x**3,x**2*y,x*y**2,y**3,
                            x**4,x**3*y,x**2*y**2,x*y**3,y**4,
                            x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5]
        
    return X

X = make_X(x,y,5)

#trying automated make X
#from sklearn.preprocessing import PolynomialFeatures
#n = 5
#poly = PolynomialFeatures(n)
#X = poly.fit_transform(x.ravel(), y.ravel())
#print("X")
#print(np.shape(X))

def calc_beta(X,y):
    #beta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    #X = X + 0.0000001
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

beta = calc_beta(X,z.ravel())
y_tilde = X @ beta


#z_noise = FrankeFunctionNoised(x,y, 0.001)

def MSE(y,y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n

# Plot the surface.
y_tilde2D = np.reshape(y_tilde, (20,20))
surf = ax.plot_surface(x, y, y_tilde2D,
            linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
# Customize the z axis.
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

def R2score(y,y_tilde):

    y_mean = np.mean(y)

    divisor = 0
    divident = 0

    for i in range(len(y)):
        divisor += (y[i] - y_tilde[i]) ** 2
        divident += (y[i] - y_mean) ** 2

    return 1 - (divisor / divident)


def plot_MSE_R2_beta(x,y):
    mse_arr = np.zeros(5)
    R2_arr = np.zeros(5)
    for n in range(1,6):
        X = make_X(x,y,n)
        X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
        beta = calc_beta(X_train, y_train)
        y_tilde = X_test @ beta
        mse_arr[n-1] = MSE(y_test, y_tilde)
        R2_arr[n-1] = R2score(y_test, y_tilde)

    n_arr = np.linspace(1,5,5)
    fig, axs = plt.subplots(2)
    axs[0].plot(n_arr, mse_arr, label= "MSE")
    axs[1].plot(n_arr, R2_arr, label= "R2")
    plt.legend()
    plt.show()

plot_MSE_R2_beta(x,y)