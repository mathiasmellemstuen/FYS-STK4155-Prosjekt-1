import numpy as np
from enum import Enum
from sklearn.linear_model import Lasso

class LinearModelType(Enum):
    OLS = 1
    RIDGE = 2
    LASSO = 3

class LinearModel: 
    def __init__(self, linear_model_type: LinearModelType):
        self.current_linear_model_type = linear_model_type
        self.lmda = 1
        self.lasso_model = Lasso(alpha = self.lmda, fit_intercept=False, max_iter=1000, tol=0.1, normalize=True)
        self.betas = None

    def fit(self, X, y):
        if self.current_linear_model_type == LinearModelType.OLS: 
            self.ordinary_least_squares_fit(X, y)

        elif self.current_linear_model_type == LinearModelType.RIDGE: 
            self.ridge_fit(X, y)

        elif self.current_linear_model_type == LinearModelType.LASSO: 
            self.lasso_model.fit(X, y)

    def predict(self, pred_data):
        if self.current_linear_model_type == LinearModelType.OLS: 
            return pred_data @ self.betas

        elif self.current_linear_model_type == LinearModelType.RIDGE: 
            return pred_data @ self.betas

        elif self.current_linear_model_type == LinearModelType.LASSO:
            return self.lasso_model.predict(pred_data)

    def ordinary_least_squares_fit(self, X, y): 
        self.betas =  np.linalg.pinv(X.T @ X) @ X.T @ y

    def ridge_fit(self, X, y): 
        I = np.eye(len(X[0]), len(X[0]))
        self.betas = np.linalg.pinv(X.T @ X + self.lmda * I) @ X.T @ y

    def set_lambda(self, lmda):
        self.lmda = lmda
        self.lasso_model = Lasso(alpha = self.lmda, fit_intercept=False, max_iter=1000, tol=0.01, normalize=True)

    def set_linear_model_type(self, type: LinearModelType): 
        self.current_linear_model_type = type