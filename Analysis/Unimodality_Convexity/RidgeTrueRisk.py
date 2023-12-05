import numpy as np
import time
from scipy.linalg import svd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd 
    
    
class RidgeTrueRisk:

    def __init__(self, alphas=np.logspace(-10, 10, 11, endpoint=True, base=10), fit_intercept=True, normalize=True):
        self.alphas=alphas
        self.fit_intercept=fit_intercept
        self.normalize=normalize

    @staticmethod
    def alpha_range_GMLNET(x, y):
        n, p = x.shape
        # x_mu = x.mean(axis=0)
        # x_star = ((x - x_mu)/(1/n**0.5*np.sum((x - x_mu)**2, axis=0)))
        alpha_max = 1/((0.001)*n) * np.max(np.abs(x.T.dot(y)))
        alpha_min = 0.0001*alpha_max if n >= p else 0.01*alpha_max
        return alpha_min, alpha_max

    @staticmethod
    def alpha_log_grid(alpha_min, alpha_max, l=100, base=10.0):
        log_min = np.log(alpha_min) / np.log(base)
        log_max = np.log(alpha_max) / np.log(base)
        return np.logspace(log_min, log_max, l, endpoint=True)

    def fit(self, x, y, x_test, y_test):
        
        n, p = x.shape

        a_x, a_y = (x.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x, b_y = (x.std(axis=0), y.std()) if self.normalize else (np.ones(p), 1.0)

        x = (x - a_x)/b_x
        y = (y - a_y)/b_y
        
        a_x_test, a_y_test = (x_test.mean(axis=0), y_test.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x_test, b_y_test = (x_test.std(axis=0), y_test.std()) if self.normalize else (np.ones(p), 1.0)

        x_test = (x_test - a_x_test)/b_x_test
        y_test = (y_test - a_y_test)/b_y_test
        
        if np.isscalar(self.alphas):
            alpha_min, alpha_max = self.alpha_range_GMLNET(x, y)
            self.alphas_ = self.alpha_log_grid(alpha_min, alpha_max, self.alphas)
        else:
            self.alphas_ = self.alphas
        
        LR = LinearRegression().fit(x, y)
        self.lr_coef = LR.coef_
        
        self.true_risk = np.zeros_like(self.alphas_)
        self.coefs = []
    
        for i in range(len(self.alphas_)):
            rr = Ridge(alpha=self.alphas_[i], fit_intercept = self.fit_intercept)
            rr.fit(x, y)
            pred_test_rr= rr.predict(x_test)
            self.true_risk[i] = (mean_squared_error(y_test,pred_test_rr))
            self.coefs.append(rr.coef_)
            
        return self
        