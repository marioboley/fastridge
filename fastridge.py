"""
Fast and accurate ridge regression via Expectation Maximization.

Examples
--------
>>> import numpy as np
>>> from fastridge import RidgeEM, RidgeLOOCV
>>> rng = np.random.default_rng(0)
>>> n, p = 500, 5
>>> beta = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
>>> X = rng.standard_normal((n, p))
>>> y = X @ beta + 0.05 * rng.standard_normal(n)
>>> em = RidgeEM().fit(X, y)
>>> np.allclose(em.coef_, beta, atol=0.1)
True
>>> loocv = RidgeLOOCV().fit(X, y)
>>> np.allclose(loocv.coef_, beta, atol=0.1)
True
"""
import numpy as np
import time
from scipy.linalg import svd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin

from importlib.metadata import version                                                                                                                                                                                                                                                       
__version__ = version('fastridge') 

def neg_q_function(theta, w, z, n, p):
    """Negative Q-function for the numerical M-step in EM for Bayesian ridge.

    Minimized via BFGS when closed_form_m_step=False. theta[0] = tau_square,
    theta[1] = sigma_square; w and z are E-step sufficient statistics ESN and ESS.
    """
    tau_square, sigma_square = theta[0], theta[1]
    neg_log_prior = np.log(1 + tau_square) + np.log(tau_square) / 2
    q = ((n + p + 2) / 2 * np.log(sigma_square) + z / (2 * sigma_square)
         + p * np.log(tau_square) / 2 + w / (2 * sigma_square * tau_square)
         + neg_log_prior)
    return -q


def em_max_marginal_posterior_ridge(c, s, n, p, y_sqnorm, epsilon=1e-8, t2=True,
                                    closed_form_m_step=True, verbose=False, trace=False):
    """Find ridge hyperparameters via EM in SVD-projected space.

    All inputs and outputs are relative to the orthogonalised input; beta is
    the coefficient vector in that space (not rotated back to feature space).

    Parameters
    ----------
    c : ndarray, shape (r,)
        Projected observations U^T y * s, where r = min(n, p).
    s : ndarray, shape (r,)
        Singular values.
    n, p : int
        Original dataset dimensions.
    y_sqnorm : float
        Squared norm of the (preprocessed) target vector.
    epsilon : float
        Convergence threshold on relative RSS change.
    t2 : bool
        If True, Beta Prime prior on tau^2; if False, half-Cauchy prior on tau.
    closed_form_m_step : bool
        If True, use closed-form M-step; if False, use BFGS via neg_q_function.
    verbose : bool
        If True, print (tau_square, sigma_square) at each iteration.
    trace : bool
        If True, additionally return per-iteration histories.

    Returns
    -------
    sigma_square, tau_square, beta, n_iter
        Converged hyperparameters and projected coefficient vector.
        When trace=True, additionally returns sigma_hist, tau_hist, beta_hist
        (lists including initial state at index 0, then one entry per iteration).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> n, p = 500, 3
    >>> beta_true = np.array([1., -1., 0.5])
    >>> X = rng.standard_normal((n, p))
    >>> y = X @ beta_true + 0.1 * rng.standard_normal(n)
    >>> Xn = (X - X.mean(0)) / X.std(0); yn = (y - y.mean()) / y.std()
    >>> u, s_sv, vt = svd(Xn, full_matrices=False)
    >>> c = u.T @ yn * s_sv
    >>> sigma_sq, tau_sq, beta, n_iter = em_max_marginal_posterior_ridge(
    ...     c, s_sv, n, p, float(yn @ yn))
    >>> n_iter > 0
    True
    >>> coef = vt.T @ beta * y.std() / X.std(0)
    >>> np.allclose(coef, beta_true, atol=0.1)
    True
    >>> out = em_max_marginal_posterior_ridge(c, s_sv, n, p, float(yn @ yn), trace=True)
    >>> sigma_sq2, tau_sq2, beta2, n_iter2, sh, th, bh = out
    >>> len(sh) == n_iter2 + 1
    True
    >>> np.allclose(bh[-1], beta2)
    True
    """
    tau_square = 1.0
    sigma_square = y_sqnorm / n
    RSS = 1e10
    n_iter = 0
    beta_init = c / s ** 2
    if trace:
        sigma_hist = [sigma_square]
        tau_hist = [tau_square]
        beta_hist = [beta_init.copy()]

    while True:
        RSS_old = RSS
        beta = c / (s * s + 1.0 / tau_square)
        ESN = (beta.dot(beta)
               + sigma_square * ((1.0 / (s * s + 1.0 / tau_square)).sum()
                                 + tau_square * max(p - n, 0)))
        RSS = y_sqnorm - 2.0 * beta.dot(c) + (beta * beta).dot(s * s)
        ESS = RSS + sigma_square * (s * s / (s * s + 1.0 / tau_square)).sum()
        if closed_form_m_step:
            if t2:
                tau_square = ((ESN * (-1 + n) - ESS * (1 + p)
                               + (4 * ESN * (n + 1) * ESS * (3 + p)
                                  + (ESN + ESS * (p + 1) - ESN * n) ** 2) ** 0.5)
                              / (2 * ESS * (3 + p)))
                sigma_square = (ESS * tau_square + ESN) / ((n + p + 2) * tau_square)
            else:
                tau_square = ((ESN * (-1 + n) - ESS * p
                               + (4 * ESN * (n + 1) * ESS * (2 + p)
                                  + (ESN + ESS * p - ESN * n) ** 2) ** 0.5)
                              / (2 * ESS * (2 + p)))
                sigma_square = (ESS * tau_square + ESN) / ((n + p + 1) * tau_square)
        else:
            theta_init = np.array([tau_square, sigma_square])
            opt_res = minimize(neg_q_function, x0=theta_init,
                               args=(ESN, ESS, n, p), method='BFGS')
            tau_square, sigma_square = opt_res.x[0], opt_res.x[1]
        delta = abs(RSS_old - RSS) / (1.0 + abs(RSS))
        if verbose:
            print(tau_square, sigma_square)
        if trace:
            beta_t = c / (s * s + 1.0 / tau_square)
            sigma_hist.append(sigma_square)
            tau_hist.append(tau_square)
            beta_hist.append(beta_t)
        n_iter += 1
        if delta < epsilon:
            break

    beta = c / (s * s + 1.0 / tau_square)
    if trace:
        return sigma_square, tau_square, beta, n_iter, sigma_hist, tau_hist, beta_hist
    return sigma_square, tau_square, beta, n_iter


def em_max_marginal_posterior_ridge_multi_target(c, s, n, p, y_sqnorm, epsilon=1e-8,
                                                 t2=True, closed_form_m_step=True,
                                                 verbose=False, trace=False):
    """Per-target EM for multi-output ridge regression in SVD-projected space.

    Each target column runs an independent EM via em_max_marginal_posterior_ridge,
    producing independent sigma_square and tau_square estimates per target.

    Parameters
    ----------
    c : ndarray, shape (r, q)
        Projected observations for each of the q targets.
    s : ndarray, shape (r,)
        Singular values.
    n, p : int
        Original dataset dimensions.
    y_sqnorm : ndarray, shape (q,)
        Per-target squared norms of the preprocessed target vectors.
    epsilon, t2, closed_form_m_step, verbose : same as em_max_marginal_posterior_ridge.
    trace : bool
        If True, additionally return per-target iteration histories (ragged: each
        target may converge in a different number of iterations).

    Returns
    -------
    sigma_arr, tau_arr : ndarray, shape (q,)
    beta_mat : ndarray, shape (r, q)
    n_iter_arr : ndarray of int, shape (q,)
        When trace=True, additionally returns sigma_hist, tau_hist, beta_hist --
        each a list of length q whose t-th entry is the history for target t.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> n, p = 500, 3
    >>> beta1, beta2 = np.array([1., -1., 0.5]), np.array([-0.5, 2., 0.])
    >>> X = rng.standard_normal((n, p))
    >>> Y = X @ np.column_stack([beta1, beta2]) + 0.1 * rng.standard_normal((n, 2))
    >>> Xn = (X - X.mean(0)) / X.std(0)
    >>> Yn = (Y - Y.mean(0)) / Y.std(0)
    >>> u, s_sv, vt = svd(Xn, full_matrices=False)
    >>> c2 = (u.T @ Yn) * s_sv[:, None]
    >>> sigma_arr, tau_arr, beta_mat, n_iter_arr = em_max_marginal_posterior_ridge_multi_target(
    ...     c2, s_sv, n, p, (Yn ** 2).sum(0))
    >>> beta_mat.shape
    (3, 2)
    >>> coef0 = vt.T @ beta_mat[:, 0] * Y.std(0)[0] / X.std(0)
    >>> coef1 = vt.T @ beta_mat[:, 1] * Y.std(0)[1] / X.std(0)
    >>> np.allclose(coef0, beta1, atol=0.1) and np.allclose(coef1, beta2, atol=0.1)
    True
    """
    q = c.shape[1]
    r = len(s)
    sigma_arr = np.empty(q)
    tau_arr = np.empty(q)
    beta_mat = np.empty((r, q))
    n_iter_arr = np.empty(q, dtype=int)
    if trace:
        sigma_hist_list, tau_hist_list, beta_hist_list = [], [], []
    for t in range(q):
        result = em_max_marginal_posterior_ridge(
            c[:, t], s, n, p, y_sqnorm[t], epsilon, t2, closed_form_m_step, verbose, trace)
        if trace:
            sigma_arr[t], tau_arr[t], beta_mat[:, t], n_iter_arr[t], sh, th, bh = result
            sigma_hist_list.append(sh)
            tau_hist_list.append(th)
            beta_hist_list.append(bh)
        else:
            sigma_arr[t], tau_arr[t], beta_mat[:, t], n_iter_arr[t] = result
    if trace:
        return (sigma_arr, tau_arr, beta_mat, n_iter_arr,
                sigma_hist_list, tau_hist_list, beta_hist_list)
    return sigma_arr, tau_arr, beta_mat, n_iter_arr


class RidgeEM(MultiOutputMixin, BaseEstimator, RegressorMixin):
    """Bayesian ridge regression via Expectation-Maximization.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((2000, 1))
    >>> y = 2.0 * X[:, 0] + 0.5 * rng.standard_normal(2000)
    >>> est = RidgeEM().fit(X, y)
    >>> round(float(est.coef_[0]), 1)
    2.0
    >>> round(float(est.sigma_square_), 2)
    0.25
    >>> round(est.score(X, y), 2)
    0.94
    """

    def __init__(self, epsilon=0.00000001, fit_intercept=True, normalize=True,
                 closed_form_m_step=True, t2=True, trace=False, trace_space='original',
                 verbose=False):
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.trace = trace
        self.verbose = verbose
        self.closed_form_m_step = closed_form_m_step
        self.t2 = t2
        self.trace_space = trace_space

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        n, p = x.shape

        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        svd_start_time = time.time()
        u, s, v_trans = svd(x, full_matrices=False)
        self.svd_time_ = time.time() - svd_start_time

        squeeze = y.ndim == 1
        y = y[:, None] if squeeze else y
        q = y.shape[1]
        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y_norm = (y - a_y) / b_y
        y_sqnorm = (y_norm ** 2).sum(axis=0)
        c = (u.T @ y_norm) * s[:, None]

        result = em_max_marginal_posterior_ridge_multi_target(
            c, s, n, p, y_sqnorm, self.epsilon, self.t2,
            self.closed_form_m_step, self.verbose, self.trace)

        if self.trace:
            sigma_arr, tau_arr, beta_mat, n_iter_arr, sh_list, th_list, bh_list = result
            self.sigma_squares_ = sh_list
            self.tau_squares_ = th_list
            if self.trace_space == 'original':
                coefs_per_target = [
                    [v_trans.T @ b * b_y[t] / b_x for b in bh_list[t]]
                    for t in range(q)]
            else:
                coefs_per_target = bh_list
            self.coefs_ = coefs_per_target
        else:
            sigma_arr, tau_arr, beta_mat, n_iter_arr = result

        self.coef_ = (v_trans.T @ beta_mat).T * b_y[:, None] / b_x
        self.intercept_ = a_y - self.coef_ @ a_x
        self.sigma_square_ = sigma_arr * b_y ** 2
        self.tau_square_ = tau_arr
        self.alpha_ = 1.0 / tau_arr
        self.iterations_ = n_iter_arr

        if squeeze:
            self.coef_ = self.coef_[0]
            self.intercept_ = self.intercept_[0]
            self.sigma_square_ = self.sigma_square_[0]
            self.tau_square_ = self.tau_square_[0]
            self.alpha_ = self.alpha_[0]
            self.iterations_ = int(self.iterations_[0])
            if self.trace:
                self.sigma_squares_ = self.sigma_squares_[0]
                self.tau_squares_ = self.tau_squares_[0]
                self.coefs_ = self.coefs_[0]
        return self

    def predict(self, x):
        return x @ self.coef_.T + self.intercept_


class RidgeLOOCV(MultiOutputMixin, BaseEstimator, RegressorMixin):

    def __init__(self, alphas=np.logspace(-10, 10, 11, endpoint=True, base=10), fit_intercept=True, normalize=True):
        self.alphas=alphas
        self.fit_intercept=fit_intercept
        self.normalize=normalize

    @staticmethod
    def alpha_range_GMLNET(x, y):
        n, p = x.shape
        # x_mu = x.mean(axis=0)
        # x_star = ((x - x_mu)/(1/n**0.5*np.sum((x - x_mu)**2, axis=0)))
        alpha_max = 1/((0.001)*n) * np.max(np.abs(x.T.dot(y)))
        alpha_min = 0.0001*alpha_max if n >= p else 0.01*alpha_max
        return alpha_min, alpha_max

    @staticmethod
    def alpha_log_grid(alpha_min, alpha_max, l=100, base=10.0):
        log_min = np.log(alpha_min) / np.log(base)
        log_max = np.log(alpha_max) / np.log(base)
        return np.logspace(log_min, log_max, l, endpoint=True)

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        n, p = x.shape

        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        if np.isscalar(self.alphas):
            alpha_min, alpha_max = self.alpha_range_GMLNET(x, y)
            self.alphas_ = self.alpha_log_grid(alpha_min, alpha_max, self.alphas)
        else:
            self.alphas_ = self.alphas

        svd_start_time = time.time()
        u, s, v_trans = svd(x, full_matrices=False)
        self.svd_time_ = time.time() - svd_start_time
        r = u * s

        h_per_alpha = []
        for alpha in self.alphas_:
            z = u * (s ** 2 / (s ** 2 + alpha))
            h_per_alpha.append((z * u).sum(axis=1))

        squeeze = y.ndim == 1
        y = y[:, None] if squeeze else y
        q = y.shape[1]
        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y_norm = (y - a_y) / b_y
        c_mat = (u.T @ y_norm) * s[:, None]

        loo_mse_mat = np.zeros((q, len(self.alphas_)))
        for t in range(q):
            c_t = c_mat[:, t]
            for i, alpha in enumerate(self.alphas_):
                beta_t = c_t / (s ** 2 + alpha)
                err = y_norm[:, t] - r.dot(beta_t)
                loo_mse_mat[t, i] = np.mean((err / (1 - h_per_alpha[i])) ** 2)

        i_stars = np.argmin(loo_mse_mat, axis=1)
        self.alpha_ = self.alphas_[i_stars]
        self.loo_mse_ = loo_mse_mat

        beta_mat = np.empty((len(s), q))
        for t in range(q):
            beta_mat[:, t] = c_mat[:, t] / (s ** 2 + self.alpha_[t])

        self.coef_ = (v_trans.T @ beta_mat).T * b_y[:, None] / b_x
        self.sigma_square_ = loo_mse_mat[np.arange(q), i_stars] * b_y ** 2
        self.intercept_ = a_y - self.coef_ @ a_x

        if squeeze:
            self.coef_ = self.coef_[0]
            self.intercept_ = self.intercept_[0]
            self.sigma_square_ = self.sigma_square_[0]
            self.alpha_ = self.alpha_[0]
        return self

    def predict(self, x):
        return x @ self.coef_.T + self.intercept_