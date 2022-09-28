import numpy as np

def ridge_beta(X, y, lam):
    I = np.eye(len(X[0]), len(X[0]))
    return np.linalg.pinv(X.T @ X + lam*I) @ X.T @ y

