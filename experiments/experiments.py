import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from fastprogress.fastprogress import progress_bar


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
        elif hasattr(est, 'alphas_'):
            return len(est.alphas_)
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

default_stats = [parameter_mean_squared_error, prediction_mean_squared_error,
                 regularization_parameter, number_of_iterations, fitting_time]


class Experiment:

    def __init__(self, problems, estimators, ns, reps, est_names=None, stats=default_stats,
                 keep_fits=True, verbose=0, seed=None):
        self.problems = problems
        self.estimators = estimators
        self.ns = np.atleast_2d(ns)
        self.ns = self.ns if len(self.ns) == len(self.problems) else self.ns.repeat(len(problems), axis=0)
        self.reps = reps
        self.verbose = verbose
        self.est_names = [str(est) for est in estimators] if est_names is None else est_names
        self.stats = stats
        self.keep_fits = keep_fits
        self.test_size = 10000
        self.rng = np.random.default_rng(seed)

    def run(self):
        if self.keep_fits:
            self.fits = {}
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.zeros(
                shape=(self.reps, len(self.problems), len(self.ns[0]), len(self.estimators)))
        for r in progress_bar(range(self.reps)):
            for i in range(len(self.problems)):
                x_test, y_test = self.problems[i].rvs(self.test_size, rng=self.rng)
                for n_idx, n in enumerate(self.ns[i]):
                    for j, est in enumerate(self.estimators):
                        x, y = self.problems[i].rvs(n, rng=self.rng)
                        _est = clone(est, safe=False)
                        fit_start_time = time.time()
                        _est.fit(x, y)
                        _est.fitting_time_ = time.time() - fit_start_time
                        if self.keep_fits:
                            self.fits[(r, i, n, j)] = _est
                        for stat in self.stats:
                            self.__dict__[str(stat) + '_'][r, i, n_idx, j] = stat(
                                _est, self.problems[i], x_test, y_test)
        return self


class RidgePathExperiment:

    def __init__(self, x_train, y_train, x_test, y_test, alphas,
                 fit_intercept=True, normalize=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def run(self):
        n, p = self.x_train.shape

        a_x = self.x_train.mean(axis=0) if self.fit_intercept else np.zeros(p)
        a_y = self.y_train.mean() if self.fit_intercept else 0.0
        b_x = self.x_train.std(axis=0) if self.normalize else np.ones(p)
        b_y = self.y_train.std() if self.normalize else 1.0

        x_tr = (self.x_train - a_x) / b_x
        y_tr = (self.y_train - a_y) / b_y
        a_x_te, a_y_te = (self.x_test.mean(axis=0), self.y_test.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x_te, b_y_te = (self.x_test.std(axis=0), self.y_test.std()) if self.normalize else (np.ones(p), 1.0)
        x_te = (self.x_test - a_x_te) / b_x_te
        y_te = (self.y_test - a_y_te) / b_y_te

        self.alphas_ = np.asarray(self.alphas)
        self.coef_path_ = np.zeros((p, len(self.alphas_)))
        self.true_risk_ = np.zeros(len(self.alphas_))

        for i, alpha in enumerate(self.alphas_):
            rr = Ridge(alpha=alpha, fit_intercept=False)
            rr.fit(x_tr, y_tr)
            self.coef_path_[:, i] = rr.coef_
            self.true_risk_[i] = mean_squared_error(y_te, rr.predict(x_te))

        lr = LinearRegression(fit_intercept=False)
        lr.fit(x_tr, y_tr)
        self.ols_coef_ = lr.coef_

        return self


def run_real_data_experiments(problems, estimators={}, n_iterations=100,
                              test_prop=0.3, seed=None, polynomial=None,
                              classification=False, verbose=True):
    """Run repeated train/test experiments on a list of EmpiricalDataProblem instances.

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
        Each problem specifies dataset, target column, and columns to drop.
    estimators : dict mapping str to estimator
    n_iterations : int
    test_prop : float
    seed : int or None
    polynomial : int or None
    classification : bool
    verbose : bool

    Returns
    -------
    list of dict
        One result dict per problem, parallel to the input list. Each dict maps
        estimator name to a dict of aggregated metrics.
    """
    results = []
    for problem in problems:
        X, y = problem.get_X_y()

        if verbose:
            print(problem.dataset, end=' ')

        categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        if categorical_cols:
            encoded_X = encoder.fit_transform(X[categorical_cols])
            X = pd.concat([
                X.drop(categorical_cols, axis=1),
                pd.DataFrame(encoded_X, columns=encoder.get_feature_names_out(categorical_cols))
            ], axis=1)

        if polynomial is not None:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
            npoly, ppoly = X_poly.shape
            if npoly * ppoly > 35000000:
                X_poly = X_poly.drop(X.columns, axis=1)
                pnew = int(np.ceil(35000000 / npoly))
                X_poly = X_poly.iloc[:, np.random.choice(X_poly.shape[1], size=pnew, replace=False)]
                X = pd.concat([X, X_poly], axis=1)
            else:
                X = X_poly

        if verbose:
            print(f'(n={X.shape[0]}, p={X.shape[1]})', end='')

        estimator_results = {
            est_name: {'mse': [], 'r2': [], 'time': [], 'p': [], 'lambda': [], 'iter': [], 'CA': [], 'q': []}
            for est_name in estimators
        }

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_iterations):
            if verbose:
                print('.', end='')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
            std = X_train.std()
            non_zero_std_cols = std[std != 0].index
            X_train = X_train[non_zero_std_cols]
            X_test = X_test[non_zero_std_cols]

            for est_name, estimator in estimators.items():
                t0 = time.time()
                estimator.fit(X_train, y_train)
                elapsed = time.time() - t0

                if classification:
                    estimator_results[est_name]['CA'].append(estimator.score(X_test, y_test))
                    estimator_results[est_name]['p'].append(X_train.shape[1])
                    estimator_results[est_name]['q'].append(len(estimator.classes_))
                else:
                    y_pred = estimator.predict(X_test)
                    estimator_results[est_name]['mse'].append(mean_squared_error(y_test, y_pred))
                    estimator_results[est_name]['r2'].append(r2_score(y_test, y_pred))
                    estimator_results[est_name]['p'].append(len(estimator.coef_))
                    estimator_results[est_name]['lambda'].append(estimator.alpha_)

                estimator_results[est_name]['time'].append(elapsed)
                if est_name == 'EM':
                    estimator_results[est_name]['iter'].append(estimator.iterations_)

        data_results = {}
        for est_name, er in estimator_results.items():
            data_results[est_name] = {
                'mse':     np.mean(er['mse']) if er['mse'] else float('nan'),
                'r2':      np.mean(er['r2']) if er['r2'] else float('nan'),
                'time':    np.mean(er['time']),
                'p':       np.mean(er['p']),
                'n_train': int(X_train.shape[0]),
                'lambda':  np.mean(er['lambda']) if er['lambda'] else float('nan'),
                'iter':    np.mean(er['iter']) if er['iter'] else 100,
                'CA':      np.mean(er['CA']) if er['CA'] else float('nan'),
                'q':       np.mean(er['q']) if er['q'] else float('nan'),
            }
        results.append(data_results)
        if verbose:
            print()

    return results
