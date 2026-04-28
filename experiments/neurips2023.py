import os
import time
import warnings
import datetime
import dataclasses
import shutil
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar
from fastridge import RidgeEM, RidgeLOOCV
from experiments import (default_stats, CACHE_DIR, cache_key, make_run_id,
                          RUN_FILE_STATE, empirical_default_stats)
from problems import EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric
from data import DATASETS
from util import save_json, load_json, to_json, environment


class SyntheticDataExperiment:

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


NEURIPS2023 = frozenset({
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric,
                         y_transforms=(np.log,),
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('blog',             'V281',
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'),
                         nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices',        'reference',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric,
                         y_transforms=(np.log1p,),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=('GT_turbine_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=('GT_compressor_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=('total_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=('motor_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ribo',             'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          'G3',
                         drop=('G1', 'G2'),
                         x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',           'V97',
                         zero_variance_filter=True),
    EmpiricalDataProblem('twitter',          'V78',
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         y_transforms=(np.log,),
                         zero_variance_filter=True),
})


NEURIPS2023_D2 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(2),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(3),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)

# Training set sizes (n_train = floor(0.7 * n_actual)) derived from actual
# post-preprocessing row counts. Keyed by dataset name and shared across
# NEURIPS2023, NEURIPS2023_D2, and NEURIPS2023_D3 (polynomial expansion does
# not change row count; no dataset in these sets has differing n across targets).
#
# Deviations from floor(0.7 * DATASETS[dataset]['n']):
#   automobile: actual n=159 after drop_rows (registry n=205) -> n_train=111 vs 143
#   autompg:    actual n=392 after drop_rows (registry n=398) -> n_train=274 vs 278
#   facebook:   actual n=499 after drop_rows (registry n=500) -> n_train=349 vs 350
NEURIPS2023_TRAIN_SIZES = {
    'abalone':          2923,
    'airfoil':          1052,
    'automobile':        111,
    'autompg':           274,
    'blog':            36677,
    'boston':            354,
    'concrete':          721,
    'crime':            1395,
    'ct_slices':       37450,
    'diabetes':          309,
    'eye':                84,
    'facebook':          349,
    'forest':            361,
    'naval_propulsion': 8353,
    'parkinsons':       4112,
    'real_estate':       289,
    'ribo':               49,
    'student':           454,
    'tomshw':          19725,
    'twitter':        408275,
    'yacht':             215,
}

NEURIPS2023_ESTIMATORS = [
    RidgeEM(t2=False),
    RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
    RidgeLOOCV(alphas=100),
]

NEURIPS2023_EST_NAMES = ['EM', 'CV_fix', 'CV_glm']


class ExperimentWithPerSeriesSeeding:
    """Run repeated train/test experiments with per-series MT19937 seeding.

    Each (problem, n_train, estimator) combination uses a fresh MT19937 RNG
    seeded at seed, running reps trials. Results are cached per
    (problem, n_train, estimator, reps, seed).

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
            self.problem_keys_[prob_idx],
            str(int(self.ns[prob_idx][n_idx])),
            self.estimator_keys_[est_idx],
            str(self.reps),
            str(self.seed),
        )

    def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        return all(
            load_json(os.path.join(d, str(stat) + '.json'),
                      default={'computations': [], 'retrievals': []})['computations']
            for stat in self.stats
        )

    def _retrieve_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            values = [np.asarray(c['value']) for c in data['computations']]
            mean_val = np.mean(values, axis=0)
            self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
            data['retrievals'].append({'value': to_json(mean_val), 'run_id': self.run_id_})
            save_json(path, data, indent=None)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(f"[problem={self.problems[prob_idx].dataset}, "
                              f"n={int(self.ns[prob_idx][n_idx])}, "
                              f"est={self.est_names[est_idx]}] {msg}")
        if self.verbose:
            for _ in range(self.reps):
                print('.', end='', flush=True)

    def _run_series(self, prob_idx, n_idx, est_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.RandomState(self.seed)
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
            else:
                for stat in self.stats:
                    val = stat(_est, problem, X_test, y_test)
                    self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
            if self.verbose:
                print('.', end='', flush=True)

    def _write_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            new_values = self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx]
            if data['computations']:
                msg = stat.warn_recompute(
                    [c['value'] for c in data['computations']], new_values)
                if msg:
                    warnings.warn(
                        f"[problem={self.problems[prob_idx].dataset}, "
                        f"n={int(self.ns[prob_idx][n_idx])}, "
                        f"est={self.est_names[est_idx]}] {msg}")
            data['computations'].append({'value': to_json(new_values), 'run_id': self.run_id_})
            save_json(path, data, indent=None)

    def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = make_run_id(type(self).__name__)
        self.environment_ = environment()
        self.timestamp_start_ = datetime.datetime.now().isoformat()
        self.trials_computed_ = self.trials_retrieved_ = 0
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
                    if (not force_recompute and not overwrite_cache and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        self.trials_retrieved_ += self.reps
                    else:
                        if overwrite_cache and not ignore_cache:
                            shutil.rmtree(
                                self._series_cache_dir(prob_idx, n_idx, est_idx),
                                ignore_errors=True)
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        self.trials_computed_ += self.reps
            if self.verbose:
                print()
        self.timestamp_end_ = datetime.datetime.now().isoformat()
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        return self


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Re-run all NeurIPS 2023 empirical experiments.')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--ignore_cache',    action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    args = parser.parse_args()

    for problems in [NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3]:
        problems_sorted = sorted(problems, key=lambda p: DATASETS[p.dataset]['n'])
        ExperimentWithPerSeriesSeeding(
            problems=problems_sorted,
            estimators=NEURIPS2023_ESTIMATORS,
            reps=100,
            ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_sorted],
            seed=123,
            est_names=NEURIPS2023_EST_NAMES,
        ).run(
            force_recompute=args.force_recompute,
            ignore_cache=args.ignore_cache,
            overwrite_cache=args.overwrite_cache,
        )
