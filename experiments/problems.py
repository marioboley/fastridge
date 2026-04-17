"""
Problem classes for simulated and empirical data experiments.
"""
import warnings

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
        Ordered sequence of ``DataFrame -> DataFrame`` transforms applied to
        X after the X/y split. ``OneHotEncodeCategories`` and
        ``PolynomialExpansion`` satisfy this contract.
    y_transforms : list of callable, optional
        Ordered sequence of ``Series -> Series`` transforms applied to y
        after the X/y split. Numpy ufuncs (``np.log``, ``np.log1p``) satisfy
        this contract directly.

    Examples
    --------
    Basic usage:

    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X, y = diabetes.get_X_y()
    >>> X.shape
    (442, 10)
    >>> yacht = EmpiricalDataProblem('yacht', 'Residuary_resistance')
    >>> X, y = yacht.get_X_y()
    >>> X.shape
    (308, 6)
    >>> automobile = EmpiricalDataProblem('automobile', 'price')
    >>> X, y = automobile.get_X_y()
    >>> X.shape[0]
    201
    >>> list(X.index) == list(y.index) == list(range(201))
    True

    Dropping columns:

    >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...                                        drop=['Froude_number'])
    >>> X, y = yacht_no_froude.get_X_y()
    >>> X.shape
    (308, 5)

    NaN handling — drop rows or columns (index is always reset and aligned):

    >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
    >>> X, y = auto.get_X_y()
    >>> X.shape[0]
    159
    >>> list(X.index) == list(y.index) == list(range(159))
    True
    >>> auto_cols = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_cols')
    >>> X, y = auto_cols.get_X_y()
    >>> X.shape
    (201, 19)
    >>> list(X.index) == list(y.index) == list(range(201))
    True

    y_transforms — apply callables to the target after the X/y split:

    >>> import numpy as np
    >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
    ...                                     y_transforms=[np.log])
    >>> X, y_log = diabetes_log.get_X_y()
    >>> X_base, y_base = diabetes.get_X_y()
    >>> np.allclose(y_log.values, np.log(y_base.values))
    True

    x_transforms — apply callables to X after the X/y split:

    >>> ohe = EmpiricalDataProblem('automobile', 'price',
    ...                            nan_policy='drop_rows',
    ...                            x_transforms=[OneHotEncodeCategories()])
    >>> X_ohe, y_ohe = ohe.get_X_y()
    >>> X_ohe.shape[0]
    159
    >>> 'fuel-type_gas' in X_ohe.columns
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
                 x_transforms=None, y_transforms=None):
        self.dataset = dataset
        self.target = target
        self.drop = tuple(drop or [])
        self.nan_policy = nan_policy
        self.x_transforms = tuple(x_transforms or [])
        self.y_transforms = tuple(y_transforms or [])
        self._repr = (
            f'EmpiricalDataProblem({self.dataset!r}, {self.target!r}'
            + (f', drop={list(self.drop)!r}' if self.drop else '')
            + (f', nan_policy={self.nan_policy!r}' if self.nan_policy else '')
            + (f', x_transforms={list(self.x_transforms)!r}' if self.x_transforms else '')
            + (f', y_transforms={list(self.y_transforms)!r}' if self.y_transforms else '')
            + ')'
        )

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, EmpiricalDataProblem):
            return NotImplemented
        return self._repr == other._repr

    def __hash__(self):
        return hash(self._repr)

    def get_X_y(self):
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
            X = fn(X)
        return X, y


class PolynomialExpansion:
    """Callable value object that applies polynomial feature expansion.

    Parameters
    ----------
    degree : int
        Polynomial degree passed to PolynomialFeatures.
    max_entries : int, optional
        Maximum total entries (n * p) before column subsampling is applied.
        Default 50_000_000 (= 35M / 0.7, the paper-correct budget for a 70/30
        train/test split). When exceeded, linear terms are always kept and
        interaction columns are subsampled deterministically.
    budget_target : {'total', 'interaction'}, optional
        How the column budget is applied. ``'total'`` (default): budget is the
        maximum number of *total* columns (linear + interaction); interaction
        columns kept = ``max(0, ceil(max_entries/n) - p_linear)``. This matches
        the NeurIPS 2023 paper. ``'interaction'``: budget is applied to
        interaction columns only (``ceil(max_entries/n)`` interaction columns
        kept, regardless of p_linear); this reproduces the legacy experiment
        runner behaviour, which was a bug relative to the paper.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    >>> pe = PolynomialExpansion(2)
    >>> list(pe(X).columns)
    ['a', 'b', 'a^2', 'a b', 'b^2']
    >>> pe(X).shape
    (3, 5)
    >>> PolynomialExpansion(2) == PolynomialExpansion(2)
    True
    >>> PolynomialExpansion(2) == PolynomialExpansion(3)
    False
    >>> len({PolynomialExpansion(2), PolynomialExpansion(2)})
    1
    >>> repr(PolynomialExpansion(2))
    'PolynomialExpansion(2)'
    >>> repr(PolynomialExpansion(2, max_entries=1000))
    'PolynomialExpansion(2, max_entries=1000)'
    >>> repr(PolynomialExpansion(2, budget_target='interaction'))
    "PolynomialExpansion(2, budget_target='interaction')"

    With subsampling (budget_target='total', default): budget bounds total cols.
    >>> small = PolynomialExpansion(2, max_entries=9)
    >>> result = small(X)
    >>> result.shape[1]
    3
    >>> 'a' in result.columns and 'b' in result.columns
    True

    With budget_target='interaction': budget bounds only interaction cols.
    >>> legacy = PolynomialExpansion(2, max_entries=9, budget_target='interaction')
    >>> legacy(X).shape[1]
    5
    """

    def __init__(self, degree, max_entries=50_000_000, budget_target='total'):
        self.degree = degree
        self.max_entries = max_entries
        self.budget_target = budget_target

    def __call__(self, X):
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
            if self.budget_target == 'total':
                pnew = max(0, min(len(interaction_cols), p_budget - len(linear_cols)))
            else:
                pnew = min(len(interaction_cols), p_budget)
            rng = np.random.default_rng(self.degree)
            sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
            return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
        return X_poly

    def __eq__(self, other):
        return isinstance(other, PolynomialExpansion) and \
               (self.degree, self.max_entries, self.budget_target) == \
               (other.degree, other.max_entries, other.budget_target)

    def __hash__(self):
        return hash((type(self).__name__, self.degree, self.max_entries, self.budget_target))

    def __repr__(self):
        parts = [str(self.degree)]
        if self.max_entries != 50_000_000:
            parts.append(f'max_entries={self.max_entries}')
        if self.budget_target != 'total':
            parts.append(f'budget_target={self.budget_target!r}')
        return f'PolynomialExpansion({", ".join(parts)})'


class OneHotEncodeCategories:
    """Callable value object that one-hot encodes all categorical columns.

    Detects non-numeric columns via pd.api.types.is_numeric_dtype. Encodes
    them with OneHotEncoder(drop='first'), reconstructs a DataFrame. No-op
    when all columns are numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> enc = OneHotEncodeCategories()
    >>> X_num = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    >>> enc(X_num).equals(X_num)
    True
    >>> X_cat = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': [1.0, 2.0, 3.0]})
    >>> result = enc(X_cat)
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

    def __call__(self, X):
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
                         x_transforms=_OHE),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=_OHE,
                         y_transforms=[np.log]),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows'),
    EmpiricalDataProblem('blog',             'V281'),
    EmpiricalDataProblem('boston',           'medv'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols'),
    EmpiricalDataProblem('ct_slices',        'reference'),
    EmpiricalDataProblem('diabetes',         'target'),
    EmpiricalDataProblem('eye',              'y'),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows',
                         x_transforms=_OHE),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=_OHE,
                         y_transforms=[np.log1p]),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=['GT_turbine_decay']),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=['GT_compressor_decay']),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=['total_UPDRS']),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=['motor_UPDRS']),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area'),
    EmpiricalDataProblem('ribo',             'y'),
    EmpiricalDataProblem('student',          'G3',
                         drop=['G1', 'G2'],
                         x_transforms=_OHE),
    EmpiricalDataProblem('tomshw',           'V97'),
    EmpiricalDataProblem('twitter',          'V78'),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         y_transforms=[np.log]),
})


def _with_polynomial(p, degree):
    return EmpiricalDataProblem(
        p.dataset, p.target, list(p.drop), p.nan_policy,
        x_transforms=list(p.x_transforms) + [PolynomialExpansion(degree)],
        y_transforms=list(p.y_transforms),
    )


NEURIPS2023_D2 = frozenset(
    _with_polynomial(p, 2)
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    _with_polynomial(p, 3)
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)
