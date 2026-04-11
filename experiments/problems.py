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
>>> p_none = EmpiricalDataProblem('automobile', 'price')
>>> X_none, y_none = p_none.get_X_y()
>>> X_none.shape[0]
201
>>> p_drop2 = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
>>> X_drop2, y_drop2 = p_drop2.get_X_y()
>>> X_drop2.shape[0]
159
"""
import warnings

import numpy as np
from scipy.stats import wishart, multivariate_normal, uniform

from data import get_dataset


class EmpiricalDataProblem:

    def __init__(self, dataset, target, drop=None, nan_policy=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []
        self.nan_policy = nan_policy

    def get_X_y(self):
        df = get_dataset(self.dataset)
        missing = [c for c in self.drop if c not in df.columns]
        if missing:
            # warn rather than error — drop list may include columns absent in some sources
            warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        df = df.dropna(subset=[self.target])
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
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
