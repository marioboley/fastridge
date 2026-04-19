import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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
    ``(reps, n_problems, n_sizes, n_estimators)`` per metric, matching the
    array layout of ``Experiment``. Failed runs (exception during fit) are
    recorded as NaN and trigger a ``warnings.warn``.

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like
        Training set sizes. Broadcast to shape (n_problems, n_sizes).
        Use n_train_from_proportion() to derive from the dataset registry.
    seed : int or None
    generator : {'PCG64', 'MT19937'}, default 'PCG64'
        'MT19937' uses np.random.RandomState for legacy numerical equivalence.
    seed_scope : {'series', 'trial', 'experiment'}, default 'series'
        When the seed is reset: per-problem, per-rep, or once for the run.
    seed_progression : {'fixed', 'sequential'}, default 'fixed'
        'fixed' reuses seed; 'sequential' uses seed + unit_idx.
    stats : list of metric callables or None
    est_names : list of str or None
    verbose : bool, default True

    Examples
    --------
    >>> from fastridge import RidgeEM
    >>> from problems import EmpiricalDataProblem, n_train_from_proportion
    >>> prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    >>> ns = n_train_from_proportion([prob])
    >>> exp = EmpiricalDataExperiment(
    ...     [prob], [RidgeEM()], reps=2, ns=ns,
    ...     seed=1, generator='MT19937', verbose=False)
    >>> exp.run().prediction_r2_.shape
    (2, 1, 1, 1)
    >>> exp.ns.shape
    (1, 1)
    >>> int(exp.ns[0, 0]) > 0
    True
    """

    _rng_factories = {
        'PCG64':   lambda seed: np.random.Generator(np.random.PCG64(seed)),
        'MT19937': lambda seed: np.random.RandomState(seed),
    }

    def __init__(self, problems, estimators, reps, ns,
                 seed=None,
                 generator='PCG64',
                 seed_scope='series',
                 seed_progression='fixed',
                 stats=None, est_names=None, verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.reps = reps
        self.ns = np.atleast_2d(ns)
        if len(self.ns) != len(self.problems):
            self.ns = self.ns.repeat(len(self.problems), axis=0)
        self.seed = seed
        self.generator = generator
        self.seed_scope = seed_scope
        self.seed_progression = seed_progression
        self._rng_factory = self._rng_factories[generator]
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def _make_rng(self, unit_idx):
        seed_val = None if self.seed is None else (
            self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx
        )
        return self._rng_factory(seed_val)

    def run(self):
        n_problems = len(self.problems)
        n_estimators = len(self.estimators)
        n_sizes = len(self.ns[0])

        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)

        if self.seed_scope == 'experiment':
            self.rng = self._make_rng(unit_idx=0)

        for prob_idx, problem in enumerate(self.problems):
            if self.verbose:
                print(problem.dataset, end=' ')

            for n_idx, n_train in enumerate(self.ns[prob_idx]):
                if self.seed_scope == 'series':
                    self.rng = self._make_rng(unit_idx=prob_idx)

                for iter_idx in range(self.reps):
                    if self.verbose:
                        print('.', end='')

                    if self.seed_scope == 'trial':
                        self.rng = self._make_rng(unit_idx=iter_idx)

                    X_train, X_test, y_train, y_test = problem.get_X_y(
                        n_train, rng=self.rng)

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
                                iter_idx, prob_idx, n_idx, est_idx] = stat(
                                    _est, problem, X_test, y_test)

            if self.verbose:
                print()

        return self
