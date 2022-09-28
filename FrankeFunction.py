from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def FrankeFunctionNoised(x, y, max_noise): 

    ff = FrankeFunction(x, y)
    noise = np.random.normal(0, max_noise, len(x)*len(x))
    noise = noise.reshape(len(x), len(x))
    return ff + noise

# Plot the surface.
def plot_surface(x, y, z, z_tilde):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    z_tilde2D = np.reshape(z_tilde, (20,20))
    surf = ax.plot_surface(x, y, z_tilde2D,
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