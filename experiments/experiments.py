import time
import warnings
import json
import os
import tempfile
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


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def _cache_key(obj, slug=''):
    """Return a filesystem-safe cache key for obj.

    The key is ``ClassName[_slug]__<joblib_hash>``. The optional slug is
    included verbatim between the class name and the hash to make cache
    directories human-browsable; callers are responsible for supplying a
    meaningful, filesystem-safe value (e.g. a dataset name).

    >>> _cache_key(object(), slug='')[:len('object__')]
    'object__'
    >>> key = _cache_key(object(), slug='iris')
    >>> key.startswith('object_iris__')
    True
    """
    sep = '_' if slug else ''
    return f'{type(obj).__name__}{sep}{slug}__{joblib.hash(obj)}'



def _load_metric_file(path):
    if not os.path.exists(path):
        return {'computations': [], 'retrievals': []}
    with open(path) as f:
        return json.load(f)


def _save_metric_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception as e:
        os.unlink(tmp)
        warnings.warn(f'Cache write failed for {path}: {e}')


def _make_run_id(class_name):
    """Return a unique run identifier string of the form ``ClassName__YYYYMMDD-HHMMSS-xxxx``.

    The timestamp component is the current local time; the four-character
    suffix is a random alphanumeric string that disambiguates runs started
    within the same second.
    """
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'{class_name}__{ts}-{suffix}'


def _write_run_file(run_id, exp_spec, summary):
    data = {
        'run_id': run_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'experiment_spec': exp_spec,
        'summary': summary,
    }
    path = os.path.join(CACHE_DIR, 'runs', f'{run_id}.json')
    _save_metric_file(path, data)


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

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.coef_ - prob.beta)**2).mean()

    @staticmethod
    def __str__():
        return 'parameter_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{\beta}-\beta\|^2/p$'


class PredictionMeanSquaredError(Metric):

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.predict(x) - y)**2).mean()

    @staticmethod
    def __str__():
        return 'prediction_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{y}-y\|^2/m$'


class RegularizationParameter(Metric):

    @staticmethod
    def __call__(est, prob, x, y):
        return est.alpha_

    @staticmethod
    def __str__():
        return 'lambda'

    @staticmethod
    def symbol():
        return r'$\lambda$'


class NumberOfIterations(Metric):

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


class VarianceAbsoluteError(Metric):

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

    @staticmethod
    def __str__():
        return 'fitting_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{fit}$ [s]'


class PredictionR2(Metric):
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


class NumberOfFeatures(Metric):
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
            _cache_key(self.problems[prob_idx]),
            str(int(self.ns[prob_idx][n_idx])),
            _cache_key(self.estimators[est_idx]),
            str(self.seed + rep_idx),
        )

    def _all_stats_in_trial_cache(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        return all(
            _load_metric_file(os.path.join(d, str(stat) + '.json'))['computations']
            for stat in self.stats
        )

    def _retrieve_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            mean_val = float(np.mean([c['value'] for c in data['computations']]))
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = mean_val
            data['retrievals'].append({'value': mean_val, 'run_id': self.run_id_})
            _save_metric_file(path, data)
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
            self._last_trial_values = None
            return
        self._last_trial_values = {}
        for stat in self.stats:
            val = stat(_est, problem, X_test, y_test)
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
            self._last_trial_values[str(stat)] = (
                val.tolist() if hasattr(val, 'tolist') else float(val))

    def _write_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        if self._last_trial_values is None:
            return
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            new_val = self._last_trial_values[str(stat)]
            if data['computations']:
                msg = stat.warn_recompute(
                    [c['value'] for c in data['computations']], new_val)
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': new_val, 'run_id': self.run_id_})
            _save_metric_file(path, data)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = _make_run_id(type(self).__name__)
        trials_computed = trials_retrieved = 0
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
                            trials_retrieved += 1
                        else:
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            trials_computed += 1
                        if self.verbose:
                            print('.', end='', flush=True)
            if self.verbose:
                print()
        if not ignore_cache:
            _write_run_file(self.run_id_, {
                'problems': [_cache_key(p) for p in self.problems],
                'estimators': [_cache_key(e) for e in self.estimators],
                'ns': self.ns.tolist(),
                'reps': self.reps,
                'seed': self.seed,
            }, {'trials_computed': trials_computed, 'trials_retrieved': trials_retrieved})
        return self


class ExperimentWithPerSeriesSeeding:
    """Run repeated train/test experiments with per-series MT19937 seeding.

    Numerically identical to EmpiricalDataExperiment(generator='MT19937',
    seed_scope='series', seed_progression='fixed') at the same seed. Results
    are cached per (problem, n_train, estimator, reps, seed).

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like of shape (n_problems, n_sizes)
        A 1-D input is broadcast to all problems.
    seed : int or None
    stats : list of Metric or None
    est_names : list of str or None
    verbose : bool, default True
    """

    def __init__(self, problems, estimators, reps, ns,
                 seed=None, stats=None, est_names=None, verbose=True):
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

    def _series_cache_dir(self, prob_idx, n_idx, est_idx):
        return os.path.join(
            CACHE_DIR, 'series',
            _cache_key(self.problems[prob_idx]),
            str(int(self.ns[prob_idx][n_idx])),
            _cache_key(self.estimators[est_idx]),
            str(self.reps),
            str(self.seed),
        )

    def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        return all(
            _load_metric_file(os.path.join(d, str(stat) + '.json'))['computations']
            for stat in self.stats
        )

    def _retrieve_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            values = [np.asarray(c['value']) for c in data['computations']]
            mean_val = np.mean(values, axis=0)
            self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
            serialisable = mean_val.tolist() if hasattr(mean_val, 'tolist') else float(mean_val)
            data['retrievals'].append({'value': serialisable, 'run_id': self.run_id_})
            _save_metric_file(path, data)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(msg)
        if self.verbose:
            for _ in range(self.reps):
                print('.', end='', flush=True)

    def _run_series(self, prob_idx, n_idx, est_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.RandomState(self.seed)
        self._last_series_values = {str(stat): [] for stat in self.stats}
        for rep_idx in range(self.reps):
            X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
            _est = clone(self.estimators[est_idx], safe=False)
            try:
                t0 = time.time()
                _est.fit(X_train, y_train)
                _est.fitting_time_ = time.time() - t0
            except Exception as e:
                warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                              f" on '{problem.dataset}': {e}")
                for stat in self.stats:
                    self._last_series_values[str(stat)].append(float('nan'))
            else:
                for stat in self.stats:
                    val = stat(_est, problem, X_test, y_test)
                    self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
                    self._last_series_values[str(stat)].append(
                        val.tolist() if hasattr(val, 'tolist') else float(val))
            if self.verbose:
                print('.', end='', flush=True)

    def _write_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            new_values = self._last_series_values[str(stat)]
            if data['computations']:
                existing = [np.asarray(c['value']) for c in data['computations']]
                msg = stat.warn_recompute(existing, np.asarray(new_values))
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': new_values, 'run_id': self.run_id_})
            _save_metric_file(path, data)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = _make_run_id(type(self).__name__)
        trials_computed = trials_retrieved = 0
        for prob_idx in range(n_problems):
            if self.verbose:
                print(self.problems[prob_idx].dataset, end=' ')
            for n_idx in range(n_sizes):
                for est_idx in range(n_estimators):
                    if (not force_recompute and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        trials_retrieved += self.reps
                    else:
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        trials_computed += self.reps
            if self.verbose:
                print()
        if not ignore_cache:
            _write_run_file(self.run_id_, {
                'problems': [_cache_key(p) for p in self.problems],
                'estimators': [_cache_key(e) for e in self.estimators],
                'ns': self.ns.tolist(),
                'reps': self.reps,
                'seed': self.seed,
            }, {'trials_computed': trials_computed, 'trials_retrieved': trials_retrieved})
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
