from FrankeFunction import FrankeFunction, FrankeFunctionNoised
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Creating random data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

# Maximum number of polynominals, will also determine size of design matrix
max_polynominal = 5
def create_design_matrix(x, y, max_polyniominal = 5):
    X = np.ones((len(x), max_polyniominal))

    for i in range(len(x)):
        for j in range(1, max_polyniominal):

            poly_type = (j - 1) % 3
            current_polynominal = 1 + np.floor(j / 4)

            if poly_type == 0:
                X[i, j] = (x[i] ** current_polynominal) 
            elif poly_type == 1:
                X[i, j] = (y[i] ** current_polynominal) 
            elif poly_type == 2:
                X[i, j] = (x[i] ** current_polynominal) * (y[i] ** current_polynominal) 
    return X

X = create_design_matrix(x, y, 6)
print(X)

# Creating meshgrid
x, y = np.meshgrid(x, y)
z = FrankeFunctionNoised(x, y, 0.05)

figure = plt.figure()
ax = figure.gca(projection="3d")
surface = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

figure.colorbar(surface, shrink=0.5, aspect=5)

plt.show()