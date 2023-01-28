import numpy as np
from numpy.random import choice
from scipy.stats import wishart, multivariate_normal, norm, uniform

class linear_problem:

    def __init__(self, beta, sigma, x_dist):
        self.beta = beta
        self.sigma = sigma
        self.x_dist = x_dist

    def rvs(self, number=100):
        x = self.x_dist.rvs(size=number)
        y = x.dot(self.beta) + norm.rvs(0, self.sigma, size=number)
        return x, y


class eye_covariates:

    def __init__(self, p):
        self.p = p

    def rvs(self, n):
        I = np.eye(self.p)
        rnd_idx = np.random.choice(self.p, size=n%self.p)
        np.row_stack((n//self.p*(I,)+(I[rnd_idx],)))


class multivariate_bernoulli:

    def __init__(self, probs):
        self.probs = probs

    def rvs(self, size=1):
        res = uniform.rvs(size=(size, len(self.probs)))
        return (res <= self.probs).astype(float)


def random_sparse_vector(p, r, std=1):
    beta_ = multivariate_normal.rvs(np.zeros(r), np.diag(r*[std]))
    idx = choice(p, r)
    beta = np.zeros(p)
    beta[idx] = beta_
    return beta

def random_sparse_factor_problem(p=100, r=None, sigma_beta=1.0, sigma_eps=0.5):
    r = p if r is None else r
    x_dist = multivariate_bernoulli(np.array([1/p]*p))
    beta = random_sparse_vector(p, r, sigma_beta)
    return linear_problem(beta, sigma_eps, x_dist)

def random_multiple_means_problem(p=100, r=None, sigma_beta=1.0, sigma_eps=0.5):
    r = p if r is None else r
    x_dist = eye_covariates(p)
    beta = random_sparse_vector(p, r, sigma_beta)
    return linear_problem(beta, sigma_eps, x_dist)    

def random_problem(p, r=None, sigma_beta=1.0, sigma_eps=0.5):
    r = p if r is None else r
    x_cov = wishart.rvs(p, np.eye(p))
    x_mu = multivariate_normal.rvs(np.zeros(p))
    x_dist = multivariate_normal(x_mu, x_cov)
    beta = random_sparse_vector(p, r, sigma_beta)
    return linear_problem(beta, sigma_eps, x_dist)
