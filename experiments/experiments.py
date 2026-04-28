import time
import warnings
import os
import datetime
import random
import string
import sys
import platform
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastprogress.fastprogress import progress_bar

from util import to_json, from_json, save_json, load_json, environment


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

RUN_FILE_STATE = [
    'run_id_', 'timestamp_start_', 'timestamp_end_', 'environment_',
    'problem_keys_', 'estimator_keys_', 'trials_retrieved_', 'trials_computed_'
]


def cache_key(obj, slug=''):
    """Return a filesystem-safe cache key for obj.

    The key is ``ClassName[_slug]__<joblib_hash>``. The optional slug is
    included verbatim between the class name and the hash to make cache
    directories human-browsable; callers are responsible for supplying a
    meaningful, filesystem-safe value (e.g. a dataset name).

    >>> cache_key(object(), slug='')[:len('object__')]
    'object__'
    >>> key = cache_key(object(), slug='iris')
    >>> key.startswith('object_iris__')
    True
    """
    sep = '_' if slug else ''
    return f'{type(obj).__name__}{sep}{slug}__{joblib.hash(obj)}'



def make_run_id(class_name):
    """Return a unique run identifier string of the form ``ClassName__YYYYMMDD-HHMMSS-xxxx``.

    The timestamp component is the current local time; the four-character
    suffix is a random alphanumeric string that disambiguates runs started
    within the same second.
    """
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'{class_name}__{ts}-{suffix}'


class Metric:

    def warn_recompute(self, existing, new_value):
        mean = np.array(existing).mean(axis=0)
        if not np.allclose(np.asarray(new_value), mean):
            return f'{type(self).__name__}: recomputed value differs from existing mean'
        return None

    def warn_retrieval(self, computations):
        values = np.array([np.asarray(c['value']) for c in computations])
        if not np.allclose(values, values.mean(axis=0)):
            return f'{type(self).__name__}: stored computations have non-negligible variation'
        return None


class ParameterMeanSquaredError(Metric):
    """Mean squared error between estimated and true coefficients, averaged over all elements.

    Applicable only to synthetic experiments where prob.beta is defined.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     coef_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> class _P:
    ...     beta = np.array([[1.1, 1.9], [2.9, 4.1]])
    >>> round(parameter_mean_squared_error(_E(), _P(), None, None), 2)
    0.01
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return float(((est.coef_ - prob.beta)**2).mean())

    def __str__(self):
        return 'parameter_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{\beta}-\beta\|^2/p$'


class PredictionMeanSquaredError(Metric):
    """Mean squared error between predictions and test targets, averaged over all elements.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X2 = np.eye(3)
    >>> Y2 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
    >>> est2 = Ridge(alpha=0.0001).fit(X2, Y2)
    >>> prediction_mean_squared_error(est2, None, X2, Y2) < 1e-6
    True
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return float(((est.predict(x) - np.asarray(y))**2).mean())

    def __str__(self):
        return 'prediction_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{y}-y\|^2/m$'


class RegularizationParameter(Metric):
    """Mean regularization parameter alpha_ across targets.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     alpha_ = np.array([1.0, 3.0])
    >>> regularization_parameter(_E(), None, None, None)
    2.0
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return float(np.mean(est.alpha_))

    def __str__(self):
        return 'lambda'

    @staticmethod
    def symbol():
        return r'$\lambda$'


class NumberOfIterations(Metric):
    """Total EM steps or total LOO alpha evaluations across all targets.

    Examples
    --------
    >>> import numpy as np
    >>> from fastridge import RidgeEM, RidgeLOOCV
    >>> from sklearn.linear_model import RidgeCV
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 3))
    >>> Y = rng.standard_normal((20, 2))
    >>> em = RidgeEM().fit(X, Y)
    >>> number_of_iterations(em, None, None, Y) == int(sum(em.iterations_))
    True
    >>> cv = RidgeLOOCV(alphas=10).fit(X, Y)
    >>> number_of_iterations(cv, None, None, Y)
    20
    >>> cv_sk = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, Y[:, 0])
    >>> number_of_iterations(cv_sk, None, None, Y[:, 0])
    3
    """

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'iterations_'):
            return int(np.sum(est.iterations_))
        elif hasattr(est, 'alphas_'):
            return len(est.alphas_) * (y.shape[1] if np.ndim(y) > 1 else 1)
        elif hasattr(est, 'alphas'):
            return len(est.alphas)
        else:
            return float('nan')

    def __str__(self):
        return 'number_of_iterations'

    @staticmethod
    def symbol():
        return '$k$'


class VarianceAbsoluteError(Metric):
    """Mean absolute error between estimated and true noise variance, averaged across targets.

    Returns NaN when the estimator has no sigma_square_ attribute.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     sigma_square_ = np.array([0.25, 0.36])
    >>> class _P:
    ...     sigma = 0.5
    >>> round(variance_abs_error(_E(), _P(), None, None), 3)
    0.055
    """

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'sigma_square_'):
            return float(np.mean(np.abs(prob.sigma**2 - est.sigma_square_)))
        else:
            return float('nan')

    def __str__(self):
        return 'variance_abs_error'

    @staticmethod
    def symbol():
        return r'$|\hat{\sigma}^2-\sigma^2|$'


class FittingTime(Metric):

    def warn_recompute(self, existing, new_value):
        arr = np.array(existing)
        mean = arr.mean(axis=0)
        se = (arr.std(axis=0, ddof=1) / np.sqrt(len(existing))
              if len(existing) >= 2 else np.zeros_like(mean))
        if np.any(np.abs(np.asarray(new_value) - mean) > 1.96 * se):
            return 'FittingTime: new value outside 95% CI of existing computation(s)'
        return None

    def warn_retrieval(self, computations):
        if len(computations) == 1:
            return ('FittingTime: only one computation stored; reliability unknown. '
                    'Re-run with force_recompute=True to improve estimate.')
        means = np.array([np.mean(c['value']) for c in computations])
        se = means.std(ddof=1) / np.sqrt(len(computations))
        if 1.96 * se > 1.0:
            return (f'FittingTime: cached mean unreliable '
                    f'(95% CI width {2 * 1.96 * se:.1f}s). '
                    'Re-run with force_recompute=True.')
        return None

    @staticmethod
    def __call__(est, prob, x, y):
        return est.fitting_time_

    def __str__(self):
        return 'fitting_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{fit}$ [s]'


class PredictionR2(Metric):
    """R^2 between predictions and test targets; uniform average across targets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> round(prediction_r2(est, None, X, y), 4)
    1.0
    >>> X3 = np.eye(3)
    >>> Y3 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
    >>> est3 = Ridge(alpha=0.0001).fit(X3, Y3)
    >>> prediction_r2(est3, None, X3, Y3) > 0.99
    True
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return float(r2_score(y, est.predict(x)))

    def __str__(self):
        return 'prediction_r2'

    @staticmethod
    def symbol():
        return r'$R^2$'


class NumberOfFeatures(Metric):
    """Returns the number of features used by the estimator (last dim of coef_.shape).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> number_of_features(est, None, X, y)
    1
    >>> X4 = np.arange(20).reshape(10, 2).astype(float)
    >>> Y4 = np.column_stack([X4[:, 0], X4[:, 1]])
    >>> est4 = Ridge(alpha=0.0001).fit(X4, Y4)
    >>> number_of_features(est4, None, None, None)
    2
    """

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'coef_'):
            return est.coef_.shape[-1]
        return float('nan')

    def __str__(self):
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
    """Run repeated train/test experiments on EmpiricalDataProblem instances.

    Uses per-trial PCG64 seeding: trial seed = seed + rep_idx. Results are
    cached per (problem, n_train, estimator, trial_seed).

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like of shape (n_problems, n_sizes)
        A 1-D input is broadcast to all problems.
    seed : int, default 0
    stats : list of Metric or None
    est_names : list of str or None
    verbose : bool, default True
    """

    def __init__(self, problems, estimators, reps, ns,
                 seed=0, stats=None, est_names=None, verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.reps = reps
        self.ns = np.atleast_2d(ns)
        if len(self.ns) != len(self.problems):
            self.ns = self.ns.repeat(len(self.problems), axis=0)
        self.seed = seed
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def _trial_cache_dir(self, prob_idx, n_idx, est_idx, rep_idx):
        return os.path.join(
            CACHE_DIR, 'trial',
            self.problem_keys_[prob_idx],
            # cache_key(self.problems[prob_idx]),
            str(int(self.ns[prob_idx][n_idx])),
            # cache_key(self.estimators[est_idx]),
            self.estimator_keys_[est_idx],
            str(self.seed + rep_idx),
        )

    def _all_stats_in_trial_cache(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        return all(
            load_json(os.path.join(d, str(stat) + '.json'),
                      default={'computations': [], 'retrievals': []})['computations']
            for stat in self.stats
        )

    def _retrieve_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            mean_val = float(np.mean([c['value'] for c in data['computations']]))
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = mean_val
            data['retrievals'].append({'value': mean_val, 'run_id': self.run_id_})
            save_json(path, data, indent=None)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(msg)

    def _run_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.Generator(np.random.PCG64(self.seed + rep_idx))
        X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
        _est = clone(self.estimators[est_idx], safe=False)
        try:
            t0 = time.time()
            _est.fit(X_train, y_train)
            _est.fitting_time_ = time.time() - t0
        except Exception as e:
            warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                          f" on '{problem.dataset}': {e}")
            return
        for stat in self.stats:
            val = stat(_est, problem, X_test, y_test)
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val

    def _write_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            new_val = self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx]
            if data['computations']:
                msg = stat.warn_recompute(
                    [c['value'] for c in data['computations']], new_val)
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': to_json(new_val), 'run_id': self.run_id_})
            save_json(path, data, indent=None)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = make_run_id(type(self).__name__)
        self.timestamp_start_ = datetime.datetime.now().isoformat()
        self.trials_computed_ = self.trials_retrieved_ = 0
        self.environment_ = environment()
        self.estimator_keys_ = [cache_key(est) for est in self.estimators]
        self.problem_keys_ = [cache_key(prob) for prob in self.problems]
        run_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json')
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
    
        for prob_idx in range(n_problems):
            if self.verbose:
                print(self.problems[prob_idx].dataset, end=' ')
            for n_idx in range(n_sizes):
                for est_idx in range(n_estimators):
                    for rep_idx in range(self.reps):
                        if (not force_recompute and not ignore_cache
                                and self._all_stats_in_trial_cache(
                                    prob_idx, n_idx, est_idx, rep_idx)):
                            self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_retrieved_ += 1
                        else:
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_computed_ += 1
                        if self.verbose:
                            print('.', end='', flush=True)
            if self.verbose:
                print()
        self.timestamp_end_ = datetime.datetime.now().isoformat()
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
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
    ns : array-like of shape (n_problems, n_sizes)
        Provides n_sizes training sizes per problem for which to assess
        estimator performance. A 1-D input is treated as a single row and
        broadcast to all problems; to assign a single varying size per problem
        pass a list of single-element lists ``[[n1], [n2], ...]`` or use
        n_train_from_proportion().
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

    def _seed_val(self, unit_idx):
        if self.seed is None:
            return None
        return self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx

    def _make_rng(self, unit_idx):
        return self._rng_factory(self._seed_val(unit_idx))

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
