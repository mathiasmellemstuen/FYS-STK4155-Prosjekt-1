import numpy as np

def calc_beta(X,y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y