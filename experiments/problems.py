"""
Problem classes for simulated and empirical data experiments.
"""
import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wishart, multivariate_normal, uniform
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from data import get_dataset, DATASETS


class EmpiricalDataProblem:
    """A prediction problem defined by a dataset and a target variable.

    Optionally, can also define pre-processing steps of column dropping,
    missing value handling, and feature/target transformations (applied in
    this order).

    Parameters
    ----------
    dataset : str
        Name of the dataset as registered in data.DATASETS.
    target : str
        Name of the target column.
    drop : list of str, optional
        Column names to drop before returning X. Columns absent from the
        dataset are skipped with a warning.
    nan_policy : {'drop_rows', 'drop_cols'} or None, optional
        How to handle remaining NaN values after dropping rows where the
        target is NaN. 'drop_rows' drops any row with a NaN; 'drop_cols'
        drops any column with a NaN. None (default) leaves NaNs in place.
    x_transforms : list of callable, optional
        Ordered sequence of callables with signature ``(X, rng)`` applied to
        X after the X/y split. ``rng`` is always a ``Generator`` or
        ``RandomState``; deterministic transforms may ignore it.
        ``OneHotEncodeCategories`` and ``PolynomialExpansion`` satisfy this
        contract.
    y_transforms : list of callable, optional
        Ordered sequence of ``Series -> Series`` transforms applied to y
        after the X/y split. Numpy ufuncs (``np.log``, ``np.log1p``) satisfy
        this contract directly.

    Examples
    --------
    Basic usage — returns (X_train, X_test, y_train, y_test):

    >>> import numpy as np
    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X_train, X_test, y_train, y_test = diabetes.get_X_y(300)
    >>> X_train.shape
    (300, 10)

    Dropping columns:

    >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...                                        drop=['Froude_number'])
    >>> X_train, _, _, _ = yacht_no_froude.get_X_y(200)
    >>> X_train.shape[1]
    5

    NaN handling:

    >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
    >>> X_train, _, _, _ = auto.get_X_y(100)
    >>> X_train.shape[0]
    100

    y_transforms:

    >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
    ...                                     y_transforms=[np.log])
    >>> _, _, y_log, _ = diabetes_log.get_X_y(300, rng=0)
    >>> _, _, y_base, _ = diabetes.get_X_y(300, rng=0)
    >>> np.allclose(y_log.values, np.log(y_base.values))
    True

    x_transforms:

    >>> ohe = EmpiricalDataProblem('automobile', 'price',
    ...                            nan_policy='drop_rows',
    ...                            x_transforms=[OneHotEncodeCategories()])
    >>> X_train_ohe, _, _, _ = ohe.get_X_y(100)
    >>> 'fuel-type_gas' in X_train_ohe.columns
    True

    zero_variance_filter drops constant train columns from both splits:

    >>> naval = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
    ...     drop=['GT_turbine_decay'])
    >>> naval_filt = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
    ...     drop=['GT_turbine_decay'], zero_variance_filter=True)
    >>> Xtr, _, _, _ = naval.get_X_y(50, rng=0)
    >>> std = Xtr.std()
    >>> zero_var = std[std == 0].index.tolist()
    >>> zero_var
    ['T1', 'P1']
    >>> Xtr_f, Xte_f, _, _ = naval_filt.get_X_y(50, rng=0)
    >>> list(Xtr_f.columns) == [c for c in Xtr.columns if c not in zero_var]
    True
    >>> list(Xte_f.columns) == list(Xtr_f.columns)
    True

    Value-object identity (repr-based hash and equality):

    >>> p1 = EmpiricalDataProblem('diabetes', 'target')
    >>> p2 = EmpiricalDataProblem('diabetes', 'target')
    >>> p1 == p2
    True
    >>> p1 is p2
    False
    >>> len(frozenset({p1, p2}))
    1
    >>> p3 = EmpiricalDataProblem('yacht', 'Residuary_resistance')
    >>> len(frozenset({p1, p2, p3}))
    2
    >>> EmpiricalDataProblem('diabetes', 'target').drop
    ()
    >>> EmpiricalDataProblem('diabetes', 'target', drop=['bmi']).drop
    ('bmi',)
    >>> repr(EmpiricalDataProblem('diabetes', 'target'))
    "EmpiricalDataProblem('diabetes', 'target')"
    >>> repr(EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...     drop=['Froude_number'], nan_policy='drop_rows',
    ...     y_transforms=[np.log]))
    "EmpiricalDataProblem('yacht', 'Residuary_resistance', drop=['Froude_number'], nan_policy='drop_rows', y_transforms=[<ufunc 'log'>])"
    """

    def __init__(self, dataset, target, drop=None, nan_policy=None,
                 x_transforms=None, y_transforms=None, zero_variance_filter=False):
        self.dataset = dataset
        self.target = target
        self.drop = tuple(drop or [])
        self.nan_policy = nan_policy
        self.x_transforms = tuple(x_transforms or [])
        self.y_transforms = tuple(y_transforms or [])
        self.zero_variance_filter = zero_variance_filter
        self._repr = (
            f'EmpiricalDataProblem({self.dataset!r}, {self.target!r}'
            + (f', drop={list(self.drop)!r}' if self.drop else '')
            + (f', nan_policy={self.nan_policy!r}' if self.nan_policy else '')
            + (f', x_transforms={list(self.x_transforms)!r}' if self.x_transforms else '')
            + (f', y_transforms={list(self.y_transforms)!r}' if self.y_transforms else '')
            + (', zero_variance_filter=True' if self.zero_variance_filter else '')
            + ')'
        )

    def replace(self, **kwargs):
        return EmpiricalDataProblem(
            dataset=kwargs.get('dataset', self.dataset),
            target=kwargs.get('target', self.target),
            drop=kwargs.get('drop', list(self.drop)),
            nan_policy=kwargs.get('nan_policy', self.nan_policy),
            x_transforms=kwargs.get('x_transforms', list(self.x_transforms)),
            y_transforms=kwargs.get('y_transforms', list(self.y_transforms)),
            zero_variance_filter=kwargs.get('zero_variance_filter', self.zero_variance_filter),
        )

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, EmpiricalDataProblem):
            return NotImplemented
        return self._repr == other._repr

    def __hash__(self):
        return hash(self._repr)

    def get_X_y(self, n_train, rng=None):
        if not isinstance(rng, np.random.RandomState):
            rng = np.random.default_rng(rng)
        df = get_dataset(self.dataset)
        missing = [c for c in self.drop if c not in df.columns]
        if missing:
            warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        df = df.dropna(subset=[self.target])
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
        elif self.nan_policy == 'drop_cols':
            df = df.dropna(axis=1)
        df = df.reset_index(drop=True)
        X = df.drop(columns=[self.target])
        y = df[self.target]
        for fn in self.y_transforms:
            y = fn(y)
        for fn in self.x_transforms:
            X = fn(X, rng)
        n_test = len(X) - n_train
        if n_test < 1:
            raise ValueError(
                f"n_train={n_train} leaves no test rows (dataset has {len(X)} rows "
                f"after preprocessing).")
        indices = rng.permutation(len(X))
        X_train = X.iloc[indices[n_test:]]
        X_test  = X.iloc[indices[:n_test]]
        y_train = y.iloc[indices[n_test:]]
        y_test  = y.iloc[indices[:n_test]]
        if self.zero_variance_filter:
            std = X_train.std()
            non_zero = std[std != 0].index
            X_train = X_train[non_zero]
            X_test = X_test[non_zero]
        return X_train, X_test, y_train, y_test


def n_train_from_proportion(problems, prop=0.7):
    """Return per-problem n_train ints derived from a proportion of dataset size.

    Uses the 'n' entry in DATASETS for each problem. Raises KeyError if 'n' is
    absent. The registry count may differ from the actual loaded row count when
    nan_policy drops rows.

    Examples
    --------
    >>> probs = [EmpiricalDataProblem('diabetes', 'target')]
    >>> int(n_train_from_proportion(probs)[0])
    309
    >>> int(n_train_from_proportion(probs, prop=0.8)[0])
    353
    """
    return np.array([int(DATASETS[p.dataset]['n'] * prop) for p in problems])


@dataclass(frozen=True)
class PolynomialExpansion:
    """Callable value object that applies polynomial feature expansion.

    Parameters
    ----------
    degree : int
        Polynomial degree passed to PolynomialFeatures.
    max_entries : int, optional
        Maximum total entries (n * p_expanded) before interaction columns are
        subsampled; linear columns are always kept. Default 50_000_000.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    >>> pe = PolynomialExpansion(2)
    >>> rng = np.random.default_rng(0)
    >>> list(pe(X, rng).columns)
    ['a', 'b', 'a^2', 'a b', 'b^2']
    >>> pe(X, rng).shape
    (3, 5)
    >>> PolynomialExpansion(2) == PolynomialExpansion(2)
    True
    >>> PolynomialExpansion(2) == PolynomialExpansion(3)
    False
    >>> len({PolynomialExpansion(2), PolynomialExpansion(2)})
    1

    With subsampling: total columns = ceil(max_entries / n).
    >>> small = PolynomialExpansion(2, max_entries=9)
    >>> result = small(X, np.random.default_rng(0))
    >>> result.shape[1]
    3
    >>> 'a' in result.columns and 'b' in result.columns
    True
    """
    degree: int
    max_entries: int = 50_000_000

    def __call__(self, X, rng):
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = pd.DataFrame(
            poly.fit_transform(X),
            columns=poly.get_feature_names_out(X.columns),
            index=X.index
        )
        n, p = X_poly.shape
        if n * p > self.max_entries:
            linear_cols = list(X.columns)
            interaction_cols = [c for c in X_poly.columns if c not in linear_cols]
            p_budget = int(np.ceil(self.max_entries / n))
            pnew = max(0, min(len(interaction_cols), p_budget - len(linear_cols)))
            sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
            return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
        return X_poly


class OneHotEncodeCategories:
    """Callable value object that one-hot encodes all categorical columns.

    Detects non-numeric columns via pd.api.types.is_numeric_dtype. Encodes
    them with OneHotEncoder(drop='first'), reconstructs a DataFrame. No-op
    when all columns are numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> enc = OneHotEncodeCategories()
    >>> rng = np.random.default_rng(0)
    >>> X_num = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    >>> enc(X_num, rng).equals(X_num)
    True
    >>> X_cat = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': [1.0, 2.0, 3.0]})
    >>> result = enc(X_cat, rng)
    >>> 'color' in result.columns
    False
    >>> 'size' in result.columns
    True
    >>> result.shape
    (3, 2)
    >>> OneHotEncodeCategories() == OneHotEncodeCategories()
    True
    >>> len({OneHotEncodeCategories(), OneHotEncodeCategories()})
    1
    >>> repr(OneHotEncodeCategories())
    'OneHotEncodeCategories()'
    """

    def __call__(self, X, rng):
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if not cat_cols:
            return X
        enc = OneHotEncoder(drop='first', sparse_output=False)
        encoded = enc.fit_transform(X[cat_cols])
        return pd.concat([
            X.drop(columns=cat_cols),
            pd.DataFrame(encoded,
                         columns=enc.get_feature_names_out(cat_cols),
                         index=X.index)
        ], axis=1)

    def __eq__(self, other):
        return isinstance(other, OneHotEncodeCategories)

    def __hash__(self):
        return hash(type(self).__name__)

    def __repr__(self):
        return 'OneHotEncodeCategories()'


class linear_problem:

    def __init__(self, beta, sigma, x_dist):
        self.beta = beta
        self.sigma = sigma
        self.x_dist = x_dist

    def rvs(self, number=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        x = self.x_dist.rvs(size=number, random_state=rng)
        y = x.dot(self.beta) + rng.normal(0, self.sigma, size=number)
        return x, y


class eye_covariates:

    def __init__(self, p):
        self.p = p

    def rvs(self, n, random_state=None):
        I = np.eye(self.p)
        rnd_idx = np.random.choice(self.p, size=n % self.p)
        return np.row_stack((n // self.p * (I,) + (I[rnd_idx],)))


class multivariate_bernoulli:

    def __init__(self, probs):
        self.probs = probs

    def rvs(self, size=1, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()
        res = random_state.uniform(size=(size, len(self.probs)))
        return (res <= self.probs).astype(float)


def random_sparse_vector(p, r, std=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    beta_ = rng.multivariate_normal(np.zeros(r), np.diag(r * [std]))
    idx = rng.choice(p, r, replace=False)
    beta = np.zeros(p)
    beta[idx] = beta_
    return beta


def random_sparse_factor_problem(p=100, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = p if r is None else r
    x_dist = multivariate_bernoulli(np.array([1 / p] * p))
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, x_dist)


def random_multiple_means_problem(p=100, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = p if r is None else r
    x_dist = eye_covariates(p)
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, x_dist)


def random_problem(p, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = p if r is None else r
    x_cov = wishart.rvs(p, np.eye(p), random_state=rng)
    x_mu = rng.multivariate_normal(np.zeros(p), np.eye(p))
    x_dist = multivariate_normal(x_mu, x_cov)
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, x_dist)


_OHE = [OneHotEncodeCategories()]

NEURIPS2023 = frozenset({
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=_OHE, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=_OHE,
                         y_transforms=[np.log],
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('blog',             'V281',
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices',        'reference',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows',
                         x_transforms=_OHE,
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=_OHE,
                         y_transforms=[np.log1p],
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=['GT_turbine_decay'],
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=['GT_compressor_decay'],
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=['total_UPDRS'],
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=['motor_UPDRS'],
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ribo',             'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          'G3',
                         drop=['G1', 'G2'],
                         x_transforms=_OHE,
                         zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',           'V97',
                         zero_variance_filter=True),
    EmpiricalDataProblem('twitter',          'V78',
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         y_transforms=[np.log],
                         zero_variance_filter=True),
})


NEURIPS2023_D2 = frozenset(
    p.replace(x_transforms=p.x_transforms + (PolynomialExpansion(2),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    p.replace(x_transforms=p.x_transforms + (PolynomialExpansion(3),))
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
