from sklearn.linear_model import Lasso
from design_matrix import create_design_matrix
from FrankeFunction import FrankeFunctionNoised
import numpy as np

x = np.arange(0, 1, 0.075)
y = np.arange(0, 1, 0.075)
x, y = np.meshgrid(x,y)
z = FrankeFunctionNoised(x, y, 0.01)
x = x.ravel()
y = y.ravel()
z = z.ravel()

alpha = 0.1
lasso = Lasso(alpha=alpha)

X = create_design_matrix(x, y, 5)
max_polynomial = 11

lasso.fit(X=X)

for current_polynomial in range(0, max_polynomial):
    pass