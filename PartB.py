from FrankeFunction import FrankeFunction, FrankeFunctionNoised
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Creating random data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

# Creating meshgrid
x, y = np.meshgrid(x, y)
z = FrankeFunctionNoised(x, y, 0.05)

# Maximum number of polynominals, will also determine size of design matrix
max_polynominal = 5

def create_design_matrix(x, y, max_polyniominal = 5):
    X = np.zeros((len(x), max_polyniominal + 1))

    # Assigning ones at the first row
    X[:, 0] = np.ones(len(X[:,0]))
    
    for i in range(len(x)):
        for j in range(max_polyniominal):
            X[i, j] = (x ** i) * (y ** j)

    for i in range(max_polynominal + 1): 
        for j in range(max_polynominal - i, 0, -1): 

            if i == 0 and j == 0: 
                break
            
    
    print(X)

create_design_matrix(x, y)

figure = plt.figure()
ax = figure.gca(projection="3d")
surface = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

figure.colorbar(surface, shrink=0.5, aspect=5)

plt.show()