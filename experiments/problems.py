"""
Problem classes for simulated and empirical data experiments.

>>> p = EmpiricalDataProblem('diabetes', 'target')
>>> X, y = p.get_X_y()
>>> X.shape
(442, 10)
>>> p_drop = EmpiricalDataProblem('yacht', 'Residuary_resistance', drop=[])
>>> X2, y2 = p_drop.get_X_y()
>>> X2.shape
(308, 6)
>>> p_auto = EmpiricalDataProblem('automobile', 'price')
>>> X_auto, y_auto = p_auto.get_X_y()
>>> X_auto.shape[0]
201
>>> list(X_auto.index) == list(y_auto.index) == list(range(201))
True
>>> p_auto_drop = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
>>> X_auto_drop, y_auto_drop = p_auto_drop.get_X_y()
>>> X_auto_drop.shape[0]
159
>>> list(X_auto_drop.index) == list(y_auto_drop.index) == list(range(159))
True
>>> p_auto_cols = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_cols')
>>> X_auto_cols, y_auto_cols = p_auto_cols.get_X_y()
>>> X_auto_cols.shape
(201, 19)
>>> list(X_auto_cols.index) == list(y_auto_cols.index) == list(range(201))
True
"""
import warnings

import numpy as np
from scipy.stats import wishart, multivariate_normal, uniform

from data import get_dataset


class EmpiricalDataProblem:
    """A prediction problem defined by a dataset and a target variable.

    Optionally, can also define pre-processing steps of column dropping,
    missing value handling, and column transformations (applied in this
    order).

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
    transforms : list of (str, callable) pairs, optional
        Ordered sequence of column transforms applied after NaN handling.
        Each pair ``(column_name, fn)`` applies ``fn`` to the named column
        in-place; ``fn`` must map a ``pd.Series`` to a ``pd.Series`` of the
        same length (numpy ufuncs satisfy this). Raises ``ValueError`` if a
        named column is absent from the DataFrame at transform time.

    >>> import numpy as np
    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X, y = diabetes.get_X_y()
    >>> X.shape
    (442, 10)
    >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
    ...                                     transforms=[('target', np.log)])
    >>> X_log, y_log = diabetes_log.get_X_y()
    >>> np.allclose(y_log.values, np.log(y.values))
    True
    >>> diabetes_bad = EmpiricalDataProblem('diabetes', 'target',
    ...                              transforms=[('nonexistent', np.log)])
    >>> diabetes_bad.get_X_y()
    Traceback (most recent call last):
        ...
    ValueError: Column 'nonexistent' not found in dataset 'diabetes' at transform time.
    """

    def __init__(self, dataset, target, drop=None, nan_policy=None, transforms=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []
        self.nan_policy = nan_policy
        self.transforms = transforms or []

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
        for col, fn in self.transforms:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in dataset '{self.dataset}' at transform time."
                )
            df[col] = fn(df[col])
        return df.drop(columns=[self.target]), df[self.target]


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
