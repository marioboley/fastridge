import time
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar 
from scipy.stats import norm

class LinearProblem:

    def __init__(self, beta, sigma, x_dist):
        self.beta = beta
        self.sigma = sigma
        self.x_dist = x_dist

    def rvs(self, number=100):
        x = self.x_dist.rvs(size=number)
        y = x.dot(self.beta) + norm.rvs(0, self.sigma, size=number)
        return x, y

class ParameterMeanSquaredError:

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.coef_ - prob.beta)**2).mean()

    @staticmethod
    def __str__():
        return 'parameter_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{\beta}-\beta\|^2/p$'

class PredictionMeanSquaredError:

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.predict(x) - y)**2).mean()

    @staticmethod
    def __str__():
        return 'prediction_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{y}-y\|^2/m$'    

class RegularizationParameter:

    @staticmethod
    def __call__(est, prob, x, y):
        return est.alpha_

    @staticmethod
    def __str__():
        return 'lambda'

    @staticmethod
    def symbol():
        return r'$\lambda$'

class NumberOfIterations:

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'iterations_'):
            return est.iterations_
        elif hasattr(est, 'alphas'):
            return len(est.alphas)
        else:
            return float('nan')

    @staticmethod
    def __str__():
        return 'number_of_iterations'

    @staticmethod
    def symbol():
        return '$k$'

class VarianceAbsoluteError:

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'sigma_square_'):
            return abs(prob.sigma**2 - est.sigma_square_)
        else:
            return float('nan')

    @staticmethod
    def __str__():
        return 'variance_abs_error'

    @staticmethod
    def symbol():
        return r'$|\hat{\sigma}^2-\sigma^2|$'

class FittingTime:

    @staticmethod
    def __call__(est, prob, x, y):
        return est.fitting_time_

    @staticmethod
    def __str__():
        return 'fitting_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{fit}$ [s]'

parameter_mean_squared_error = ParameterMeanSquaredError()
prediction_mean_squared_error = PredictionMeanSquaredError()
regularization_parameter = RegularizationParameter()
number_of_iterations = NumberOfIterations()
variance_abs_error = VarianceAbsoluteError()
fitting_time = FittingTime()

default_stats = [parameter_mean_squared_error, prediction_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time]

class Experiment:

    def __init__(self, problems, estimators, ns, reps, est_names=None, stats=default_stats, keep_fits=True, verbose=0):
        self.problems = problems
        self.estimators = estimators
        self.ns = np.atleast_2d(ns)
        self.ns = self.ns if len(self.ns)==len(self.problems) else self.ns.repeat(len(problems), axis=0)
        self.reps = reps
        self.verbose = verbose
        self.est_names = [str(est) for est in estimators] if est_names is None else est_names 
        self.stats = stats
        self.keep_fits = keep_fits
        self.test_size = 10000

    def run(self):
        if self.keep_fits:
            self.fits = {}
        for stat in self.stats:
            self.__dict__[str(stat)+'_'] = np.zeros(shape=(self.reps, len(self.problems), len(self.ns[0]), len(self.estimators)))
        for r in progress_bar(range(self.reps)):
            # if self.verbose:
            #     print('.', end='', flush=True)
            for i in range(len(self.problems)):
                x_test, y_test = self.problems[i].rvs(self.test_size)
                for n_idx, n in enumerate(self.ns[i]):
                    for j, est in enumerate(self.estimators):
                        x, y = self.problems[i].rvs(n)
                        _est = clone(est, safe=False)
                        fit_start_time = time.time()
                        _est.fit(x, y)
                        _est.fitting_time_ = time.time() - fit_start_time
                        if self.keep_fits:
                            self.fits[(r, i, n, j)] = _est
                        for stat in self.stats:
                            self.__dict__[str(stat)+'_'][r, i, n_idx, j] = stat(_est, self.problems[i], x_test, y_test)
        # if self.verbose:
        #     print()
        return self