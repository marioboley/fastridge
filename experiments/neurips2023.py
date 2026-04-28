import time
import dataclasses
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar
from fastridge import RidgeEM, RidgeLOOCV
from experiments import default_stats
from problems import EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric
from data import DATASETS


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
