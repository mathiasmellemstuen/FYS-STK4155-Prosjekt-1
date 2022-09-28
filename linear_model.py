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
        self.hyperparameter_lambda = 1
        self.lasso_model = Lasso(alpha = self.hyperparameter_lambda)
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
        self.betas =  np.linalg.pinv(X.T @ X + (self.hyperparameter_lambda * np.identity(X.shape))) @ X.T @ y

    def set_lambda(self, lmda):
        self.hyperparameter_lambda = lmda
        self.lasso_model = Lasso(alpha = self.hyperparameter_lambda)

    def set_linear_model_type(self, type: LinearModelType): 
        self.current_linear_model_type = type