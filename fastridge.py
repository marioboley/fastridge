import numpy as np
import time
from scipy.linalg import svd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd 

class RidgeEM:

    def __init__(self, epsilon=0.00000001, fit_intercept=True, normalize=True, closed_form_m_step=True, trace=False, verbose=False, t2 = True):
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.trace = trace
        self.verbose = verbose
        self.closed_form_m_step = closed_form_m_step
        self.t2 = t2  #parameterization - if t2, tau2 follows beta prime & we maximizing in terms of tau2, else tau follows half Cauchy & we maximize for tau

    def __repr__(self):
        return f'RidgeEM(eps={self.epsilon})'

    @staticmethod
    def neg_q_function(theta, w, z, n, p):
        tau_square = theta[0]
        sigma_square = theta[1]
        neg_log_prior = np.log(1 + tau_square) + np.log(tau_square)/2
        q = (n + p + 2)/2 * np.log(sigma_square) + z/(2*sigma_square) + p*np.log(tau_square)/2 + w/(2*sigma_square*tau_square) + neg_log_prior
        return -q

    def fit(self, x, y):
        n, p = x.shape

        a_x, a_y = (x.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x, b_y = (x.std(axis=0), y.std()) if self.normalize else (np.ones(p), 1.0)

        x = (x - a_x)/b_x
        y = (y - a_y)/b_y
        
        svd_start_time = time.time()
        u, s, v_trans = svd(x, full_matrices=False)
        self.svdTime = time.time() - svd_start_time
        
        y_sqnorm = y.dot(y)
        c = u.T.dot(y) * s
        beta = c/s**2
        tau_square = 1
        sigma_square = y.var()
        RSS = 1e10
        self.iterations_ = 0

        if self.trace:
            self.coefs_ = [v_trans.T.dot(beta)]
            self.sigma_squares_ = [sigma_square]
            self.tau_squares_ = [tau_square]

        while True:
            RSS_old = RSS
            beta_old = beta
            beta = c / (s*s + 1/tau_square)

            w = beta.dot(beta) + sigma_square*((1/(s*s+1/tau_square)).sum()+tau_square*max(p-n, 0))
            
            RSS = y_sqnorm - 2*beta.dot(c)+(beta*beta).dot(s*s)
            z = RSS + sigma_square*(s*s/(s*s + 1/tau_square)).sum()

            if self.closed_form_m_step:
                
                if self.t2:
                    
                    #tau_square = ((((w**2)*(n**2)) + ((z**2)*(p**2)) + 2*w*z*(8 + 4*n + 4*p + n*p))**0.5 + w*n -z*p)/(2*z*(2+p)) ##half cauchy
                    tau_square = (w*(-1+n) - z*(1+p) + (4*w*(n+1)*z*(3+p)+(w+z*(p+1)-w*n)**2)**0.5) / (2*z*(3+p))  ##beta prime         
                    sigma_square = (z*tau_square + w) / ((n+p+2)*tau_square)
                    
                else:
                    
                    tau_square = (w*(-1+n) - z*p + (4*w*(n+1)*z*(2+p)+(w+z*p-w*n)**2)**0.5) / (2*z*(2+p))      
                    sigma_square = (z*tau_square + w) / ((n+p+1)*tau_square)
                    
                    
            else:
                theta_init = np.array([tau_square, sigma_square])
                opt_res = minimize(self.neg_q_function, x0=theta_init, args=(w, z, n, p), method='BFGS')
                theta = opt_res.x
                tau_square, sigma_square = theta[0], theta[1]
            
            #delta = abs(beta_old - beta).sum() / (1 + abs(beta).sum())
            delta = abs(RSS_old - RSS).sum() / (1 + abs(RSS).sum())

            if self.verbose or self.trace:
                coef = v_trans.T.dot(beta)
                if self.verbose:
                    print(tau_square, sigma_square, coef)
                if self.trace:
                    self.coefs_.append(coef)
                    self.sigma_squares_.append(sigma_square)
                    self.tau_squares_.append(tau_square)

            self.iterations_ += 1
            if  delta < self.epsilon:
                break

        beta = c / (s*s + 1/tau_square)
        beta = v_trans.T.dot(beta)
        
        self.coef_ = beta * b_y / b_x
        self.intercept_ = a_y - self.coef_.dot(a_x)
        self.sigma_square_ = sigma_square * b_y**2
        self.tau_square_ = tau_square
        self.alpha_ = 1/tau_square
        return self

    def predict(self, x):
        return x.dot(self.coef_) + self.intercept_


class RidgeLOOCV:

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

    def fit(self, x, y):
        n, p = x.shape

        a_x, a_y = (x.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x, b_y = (x.std(axis=0), y.std()) if self.normalize else (np.ones(p), 1.0)

        x = (x - a_x)/b_x
        y = (y - a_y)/b_y

        if np.isscalar(self.alphas):
            alpha_min, alpha_max = self.alpha_range_GMLNET(x, y)
            self.alphas_ = self.alpha_log_grid(alpha_min, alpha_max, self.alphas)
        else:
            self.alphas_ = self.alphas

        u, s, v_trans = svd(x, full_matrices=False)
        c = u.T.dot(y) * s
        r = u*s

        self.loo_mse_ = np.zeros_like(self.alphas_)
        for i in range(len(self.alphas_)):
            # hat = u.dot(np.diag(s**2/(s**2 + self.alphas[i]))).dot(u.T)
            # err = y - hat.dot(y)
            # loo_mse[i] = np.mean((err / (1 - np.diagonal(hat)))**2)
            z = u*(s**2/(s**2 + self.alphas_[i]))
            h = (z*u).sum(axis=1)
            # print('h', h.shape)
            beta = c/(s**2 + self.alphas_[i])
            err = y - r.dot(beta)
            self.loo_mse_[i] = np.mean((err / (1 - h))**2)

        i_star = np.argmin(self.loo_mse_)
        self.alpha_ = self.alphas_[i_star]

        beta = c / (s**2 + self.alpha_)
        beta = v_trans.T.dot(beta)
        self.sigma_square_ = self.loo_mse_[i_star] * b_y**2
        self.coef_ = beta * b_y / b_x
        self.intercept_ = a_y - self.coef_.dot(a_x)

        return self

    def predict(self, x):
        return x.dot(self.coef_) + self.intercept_ 