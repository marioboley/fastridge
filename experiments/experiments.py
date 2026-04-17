import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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


class PredictionR2:
    """Computes R² between predictions and test targets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> round(float(prediction_r2(est, None, X, y)), 4)
    1.0
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return r2_score(y, est.predict(x))

    @staticmethod
    def __str__():
        return 'prediction_r2'

    @staticmethod
    def symbol():
        return r'$R^2$'


class NumberOfFeatures:
    """Returns the number of features used by the estimator (len of coef_).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> number_of_features(est, None, X, y)
    1
    """

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'coef_'):
            return len(est.coef_)
        return float('nan')

    @staticmethod
    def __str__():
        return 'number_of_features'

    @staticmethod
    def symbol():
        return r'$p$'


parameter_mean_squared_error = ParameterMeanSquaredError()
prediction_mean_squared_error = PredictionMeanSquaredError()
regularization_parameter = RegularizationParameter()
number_of_iterations = NumberOfIterations()
variance_abs_error = VarianceAbsoluteError()
fitting_time = FittingTime()
prediction_r2 = PredictionR2()
number_of_features = NumberOfFeatures()

default_stats = [parameter_mean_squared_error, prediction_mean_squared_error,
                 regularization_parameter, number_of_iterations, fitting_time]

empirical_default_stats = [
    prediction_mean_squared_error,
    prediction_r2,
    regularization_parameter,
    number_of_iterations,
    fitting_time,
    number_of_features,
]


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


class EmpiricalDataExperiment:
    """Run repeated train/test experiments on a list of EmpiricalDataProblem instances.

    Stores per-run results as numpy arrays of shape
    ``(n_iterations, n_problems, 1, n_estimators)`` per metric, matching the
    array layout of ``Experiment``. Failed runs (exception during fit) are
    recorded as NaN and trigger a ``warnings.warn``. The seed is reset before
    each problem's iteration loop so that each problem gets the same
    deterministic split sequence regardless of list ordering.

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    n_iterations : int
    test_prop : float, default 0.3
    seed : int or None
    stats : list of metric callables or None
        Each callable has signature ``(est, prob, x, y)``. Defaults to
        ``empirical_default_stats``.
    est_names : list of str or None
        Defaults to ``[str(e) for e in estimators]``.
    verbose : bool, default True

    Examples
    --------
    >>> from fastridge import RidgeEM
    >>> from problems import EmpiricalDataProblem
    >>> prob = EmpiricalDataProblem('diabetes', 'target')
    >>> exp = EmpiricalDataExperiment(
    ...     [prob], [RidgeEM()], n_iterations=2, seed=1, verbose=False)
    >>> exp.run().prediction_r2_.shape
    (2, 1, 1, 1)
    >>> exp.ns.shape
    (1, 1)
    >>> int(exp.ns[0, 0]) > 0
    True
    """

    def __init__(self, problems, estimators, n_iterations, test_prop=0.3,
                 seed=None, stats=None, est_names=None,
                 verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.n_iterations = n_iterations
        self.test_prop = test_prop
        self.seed = seed
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def run(self):
        n_problems = len(self.problems)
        n_estimators = len(self.estimators)

        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.n_iterations, n_problems, 1, n_estimators), np.nan)
        self.ns = np.zeros((n_problems, 1), dtype=int)

        for prob_idx, problem in enumerate(self.problems):
            X, y = problem.get_X_y()

            if self.verbose:
                print(problem.dataset, end=' ')

            self.ns[prob_idx, 0] = int(X.shape[0] * (1 - self.test_prop))

            if self.verbose:
                print(f'(n={X.shape[0]}, p={X.shape[1]})', end='')

            if self.seed is not None:
                np.random.seed(self.seed)

            for iter_idx in range(self.n_iterations):
                if self.verbose:
                    print('.', end='')

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_prop)

                std = X_train.std()
                non_zero = std[std != 0].index
                X_train = X_train[non_zero]
                X_test = X_test[non_zero]

                for est_idx, est in enumerate(self.estimators):
                    _est = clone(est, safe=False)
                    try:
                        t0 = time.time()
                        _est.fit(X_train, y_train)
                        _est.fitting_time_ = time.time() - t0
                    except Exception as e:
                        warnings.warn(
                            f"Run {iter_idx} failed for '{self.est_names[est_idx]}'"
                            f" on '{problem.dataset}': {e}")
                        continue

                    for stat in self.stats:
                        self.__dict__[str(stat) + '_'][
                            iter_idx, prob_idx, 0, est_idx] = stat(
                                _est, problem, X_test, y_test)

            if self.verbose:
                print()

        return self
