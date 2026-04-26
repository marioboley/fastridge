# Multioutput Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-target support to `RidgeEM` and `RidgeLOOCV` by extracting the EM loop into public free functions and dispatching on `y.ndim` in both class wrappers.

**Architecture:** Two public free functions (`em_max_marginal_posterior_ridge`, `em_max_marginal_posterior_ridge_multi_target`) operate in SVD-projected space. The `closed_form_m_step` parameter and both M-step branches live inside the free function — the wrapper is thin (preprocess → SVD → call free function → postprocess). `neg_q_function` moves from `RidgeEM` static method to module level to support the numerical M-step inside the free function. `RidgeLOOCV` multi-target support is added inline without free functions.

**Tech Stack:** numpy, scipy.linalg.svd, scipy.optimize.minimize, pytest

**Spec:** `docs/superpowers/specs/2026-04-26-multioutput-regression-design.md`

---

## File Structure

- **Modify:** `fastridge.py` — move `neg_q_function` to module level; add two free functions before `RidgeEM`; update `RidgeEM.__init__` (add `trace_space`) and `RidgeEM.fit` (thin wrapper, remove `neg_q_function` static method); update `RidgeLOOCV.fit`
- **Create:** `tests/test_fastridge.py` — pytest tests for new behaviour (backward compat, 2D shapes, trace)

---

## Task 1: `em_max_marginal_posterior_ridge` free function

**Files:**
- Modify: `fastridge.py` (insert `neg_q_function` and new function before `class RidgeEM`)
- Test: `fastridge.py` (doctest in the function itself)

**Behaviour notes:**
- `verbose` output changes from `print(tau_square, sigma_square, coef)` to `print(tau_square, sigma_square)` — the projected coef is not available inside the free function. Acceptable: verbose was debug output, not public API.
- `sigma_square` init changes from `y.var()` to `y_sqnorm / n`. For centered `y` (default `fit_intercept=True`) these are identical. For `fit_intercept=False` there is a negligible difference that does not affect the converged solution.
- Doctests run in the module namespace: `np`, `svd`, and the function itself are all available without any imports.

- [ ] **Step 1: Move `neg_q_function` to module level**

In `fastridge.py`, insert the following before `class RidgeEM` (keep the static method on `RidgeEM` for now — it will be removed in Task 3):

```python
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
```

Run the full suite to confirm nothing breaks:

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate
pytest fastridge.py --doctest-modules -v
```

Expected: all existing doctests pass.

- [ ] **Step 2: Write the failing doctest**

Insert the following stub just before `class RidgeEM` (after `neg_q_function`):

```python
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
    raise NotImplementedError
```

- [ ] **Step 3: Run doctest to verify it fails**

```bash
pytest fastridge.py::fastridge.em_max_marginal_posterior_ridge --doctest-modules -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 4: Implement the function**

Replace the `raise NotImplementedError` body with:

```python
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
```

- [ ] **Step 5: Run doctest to verify it passes**

```bash
pytest fastridge.py::fastridge.em_max_marginal_posterior_ridge --doctest-modules -v
```

Expected: PASS

- [ ] **Step 6: Run full suite**

```bash
pytest fastridge.py --doctest-modules -v
```

Expected: all existing doctests still pass.

- [ ] **Step 7: Commit**

```bash
git add fastridge.py
git commit -m "feat: add neg_q_function at module level and em_max_marginal_posterior_ridge free function"
```

---

## Task 2: `em_max_marginal_posterior_ridge_multi_target` free function

**Files:**
- Modify: `fastridge.py` (insert after `em_max_marginal_posterior_ridge`, before `class RidgeEM`)
- Test: `fastridge.py` (doctest in the function itself)

- [ ] **Step 1: Write the failing doctest**

Insert this stub after `em_max_marginal_posterior_ridge` and before `class RidgeEM`:

```python
def em_max_marginal_posterior_ridge_multi_target(c, s, n, p, y_sqnorm, epsilon=1e-8,
                                                 t2=True, closed_form_m_step=True,
                                                 verbose=False, trace=False):
    """Per-target EM for multi-output ridge regression in SVD-projected space.

    Each target column runs an independent EM via em_max_marginal_posterior_ridge,
    producing independent sigma_square and tau_square estimates per target.

    Parameters
    ----------
    c : ndarray, shape (r, n_targets)
        Projected observations for each target.
    s : ndarray, shape (r,)
        Singular values.
    n, p : int
        Original dataset dimensions.
    y_sqnorm : ndarray, shape (n_targets,)
        Per-target squared norms of the preprocessed target vectors.
    epsilon, t2, closed_form_m_step, verbose : same as em_max_marginal_posterior_ridge.
    trace : bool
        If True, additionally return per-target iteration histories (ragged: each
        target may converge in a different number of iterations).

    Returns
    -------
    sigma_arr, tau_arr : ndarray, shape (n_targets,)
    beta_mat : ndarray, shape (r, n_targets)
    n_iter_arr : ndarray of int, shape (n_targets,)
        When trace=True, additionally returns sigma_hist, tau_hist, beta_hist —
        each a list of length n_targets whose t-th entry is the history for target t.

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
    raise NotImplementedError
```

- [ ] **Step 2: Run doctest to verify it fails**

```bash
pytest fastridge.py::fastridge.em_max_marginal_posterior_ridge_multi_target --doctest-modules -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement the function**

Replace `raise NotImplementedError` with:

```python
    n_targets = c.shape[1]
    r = len(s)
    sigma_arr = np.empty(n_targets)
    tau_arr = np.empty(n_targets)
    beta_mat = np.empty((r, n_targets))
    n_iter_arr = np.empty(n_targets, dtype=int)
    if trace:
        sigma_hist_list, tau_hist_list, beta_hist_list = [], [], []
    for t in range(n_targets):
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
```

- [ ] **Step 4: Run doctest to verify it passes**

```bash
pytest fastridge.py::fastridge.em_max_marginal_posterior_ridge_multi_target --doctest-modules -v
```

Expected: PASS

- [ ] **Step 5: Run full suite**

```bash
pytest fastridge.py --doctest-modules -v
```

Expected: all existing doctests pass.

- [ ] **Step 6: Commit**

```bash
git add fastridge.py
git commit -m "feat: add em_max_marginal_posterior_ridge_multi_target free function"
```

---

## Task 3: Update `RidgeEM` — thin wrapper for 1D, `trace_space`, no loop duplication

**Files:**
- Modify: `fastridge.py` — `RidgeEM.__init__` and `RidgeEM.fit`
- Create: `tests/test_fastridge.py`

The `fit` method is replaced with a thin wrapper: preprocess → SVD → call `em_max_marginal_posterior_ridge` → postprocess. The entire EM loop (both M-step branches) is inside the free function. The `neg_q_function` static method is removed from `RidgeEM`. The 2D path raises `NotImplementedError` as a placeholder for Task 4.

- [ ] **Step 1: Write failing tests**

Create `tests/test_fastridge.py`:

```python
import numpy as np
import pytest
from fastridge import RidgeEM, RidgeLOOCV


def _data(seed=0, n=200, p=5):
    rng = np.random.default_rng(seed)
    beta = np.array([1., -2., 0.5, 3., -1.5])
    X = rng.standard_normal((n, p))
    y = X @ beta + 0.05 * rng.standard_normal(n)
    return X, y, beta


def test_ridge_em_1d_reconstruction():
    X, y, beta = _data()
    est = RidgeEM().fit(X, y)
    np.testing.assert_allclose(est.coef_, beta, atol=0.1)
    assert hasattr(est, 'sigma_square_')
    assert hasattr(est, 'tau_square_')
    assert hasattr(est, 'alpha_')
    assert est.iterations_ > 0


def test_ridge_em_trace_space_original():
    X, y, _ = _data()
    est = RidgeEM(trace=True, trace_space='original').fit(X, y)
    # Last trace entry must match coef_ (same space)
    np.testing.assert_allclose(est.coefs_[-1], est.coef_, rtol=1e-10)


def test_ridge_em_trace_space_projected():
    X, y, _ = _data()
    est = RidgeEM(trace=True, trace_space='projected').fit(X, y)
    # Projected beta has length r = min(n, p) = p here
    assert est.coefs_[-1].shape == (5,)
    # Must NOT equal coef_ (different space)
    assert not np.allclose(est.coefs_[-1], est.coef_)


def test_ridge_em_trace_length():
    X, y, _ = _data()
    est = RidgeEM(trace=True).fit(X, y)
    # initial state + one entry per iteration
    assert len(est.coefs_) == est.iterations_ + 1
    assert len(est.sigma_squares_) == est.iterations_ + 1
    assert len(est.tau_squares_) == est.iterations_ + 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fastridge.py -v
```

Expected: `test_ridge_em_trace_space_original`, `_projected`, `_length` FAIL (no `trace_space` yet).

- [ ] **Step 3: Update `RidgeEM.__init__`**

Replace the existing `__init__` with:

```python
def __init__(self, epsilon=0.00000001, fit_intercept=True, normalize=True,
             closed_form_m_step=True, trace=False, verbose=False, t2=True,
             trace_space='original'):
    self.epsilon = epsilon
    self.fit_intercept = fit_intercept
    self.normalize = normalize
    self.trace = trace
    self.verbose = verbose
    self.closed_form_m_step = closed_form_m_step
    self.t2 = t2
    self.trace_space = trace_space
```

- [ ] **Step 4: Replace `RidgeEM.fit` with thin wrapper; remove `neg_q_function` static method**

Delete the `neg_q_function` static method from `RidgeEM`. Replace the entire `fit` method body with:

```python
def fit(self, x, y):
    n, p = x.shape

    a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
    b_x = x.std(axis=0) if self.normalize else np.ones(p)
    x = (x - a_x) / b_x

    svd_start_time = time.time()
    u, s, v_trans = svd(x, full_matrices=False)
    self.svdTime = time.time() - svd_start_time

    if y.ndim == 2:
        raise NotImplementedError("Multi-target support: see Task 4")

    # --- 1D path ---
    a_y = y.mean() if self.fit_intercept else 0.0
    b_y = y.std() if self.normalize else 1.0
    y = (y - a_y) / b_y
    y_sqnorm = float(y.dot(y))
    c = u.T.dot(y) * s

    result = em_max_marginal_posterior_ridge(
        c, s, n, p, y_sqnorm, self.epsilon, self.t2,
        self.closed_form_m_step, self.verbose, self.trace)
    if self.trace:
        sigma_square, tau_square, beta, self.iterations_, sh, th, bh = result
        self.sigma_squares_ = sh
        self.tau_squares_ = th
        if self.trace_space == 'original':
            self.coefs_ = [v_trans.T @ b * b_y / b_x for b in bh]
        else:
            self.coefs_ = list(bh)
    else:
        sigma_square, tau_square, beta, self.iterations_ = result

    beta = v_trans.T.dot(beta)
    self.coef_ = beta * b_y / b_x
    self.intercept_ = a_y - self.coef_.dot(a_x)
    self.sigma_square_ = sigma_square * b_y ** 2
    self.tau_square_ = tau_square
    self.alpha_ = 1 / tau_square
    return self
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_fastridge.py fastridge.py --doctest-modules -v
```

Expected: all 4 new tests pass; all existing doctests pass.

- [ ] **Step 6: Commit**

```bash
git add fastridge.py tests/test_fastridge.py
git commit -m "feat: make RidgeEM.fit a thin wrapper; add trace_space parameter"
```

---

## Task 4: Update `RidgeEM` for 2D dispatch

**Files:**
- Modify: `fastridge.py` — `RidgeEM.fit` 2D branch
- Modify: `tests/test_fastridge.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fastridge.py`:

```python
def test_ridge_em_2d_coef_shape():
    X, y, _ = _data()
    Y = np.column_stack([y, -y, 0.5 * y])
    est = RidgeEM().fit(X, Y)
    assert est.coef_.shape == (3, 5)
    assert est.alpha_.shape == (3,)
    assert est.sigma_square_.shape == (3,)


def test_ridge_em_2d_column_matches_1d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est_2d = RidgeEM().fit(X, Y)
    est_1d = RidgeEM().fit(X, y)
    np.testing.assert_allclose(est_2d.coef_[0], est_1d.coef_, rtol=1e-10)


def test_ridge_em_2d_trace():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est = RidgeEM(trace=True).fit(X, Y)
    # sigma_squares_ and tau_squares_ are lists of per-target history lists
    assert len(est.sigma_squares_) == 2
    assert len(est.tau_squares_) == 2
    assert len(est.coefs_) == 2


def test_ridge_em_2d_requires_closed_form():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    with pytest.raises(ValueError, match='closed_form_m_step'):
        RidgeEM(closed_form_m_step=False).fit(X, Y)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fastridge.py::test_ridge_em_2d_coef_shape tests/test_fastridge.py::test_ridge_em_2d_column_matches_1d tests/test_fastridge.py::test_ridge_em_2d_trace tests/test_fastridge.py::test_ridge_em_2d_requires_closed_form -v
```

Expected: FAIL (NotImplementedError from placeholder)

- [ ] **Step 3: Implement 2D branch in `RidgeEM.fit`**

Replace `raise NotImplementedError("Multi-target support: see Task 4")` with the following block (the 1D path that follows handles `y.ndim == 1`):

```python
    if y.ndim == 2:
        if not self.closed_form_m_step:
            raise ValueError(
                "Multi-target fitting requires closed_form_m_step=True")
        n_targets = y.shape[1]
        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(n_targets)
        b_y = y.std(axis=0) if self.normalize else np.ones(n_targets)
        y_norm = (y - a_y) / b_y
        y_sqnorm = (y_norm ** 2).sum(axis=0)
        c = (u.T @ y_norm) * s[:, None]

        result = em_max_marginal_posterior_ridge_multi_target(
            c, s, n, p, y_sqnorm, self.epsilon, self.t2, True, self.verbose, self.trace)

        if self.trace:
            sigma_arr, tau_arr, beta_mat, n_iter_arr, sh_list, th_list, bh_list = result
            self.sigma_squares_ = sh_list
            self.tau_squares_ = th_list
            if self.trace_space == 'original':
                self.coefs_ = [
                    [v_trans.T @ b * b_y[t] / b_x for b in bh_list[t]]
                    for t in range(n_targets)]
            else:
                self.coefs_ = bh_list
        else:
            sigma_arr, tau_arr, beta_mat, n_iter_arr = result

        self.coef_ = (v_trans.T @ beta_mat).T * b_y[:, None] / b_x
        self.intercept_ = a_y - self.coef_ @ a_x
        self.sigma_square_ = sigma_arr * b_y ** 2
        self.tau_square_ = tau_arr
        self.alpha_ = 1.0 / tau_arr
        self.iterations_ = n_iter_arr
        return self
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_fastridge.py fastridge.py --doctest-modules -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add fastridge.py tests/test_fastridge.py
git commit -m "feat: add RidgeEM multi-target support via 2D y dispatch"
```

---

## Task 5: Update `RidgeLOOCV` for 2D `y`

**Files:**
- Modify: `fastridge.py` — `RidgeLOOCV.fit`
- Modify: `tests/test_fastridge.py`

The hat-matrix diagonal `h` depends only on `X` and `alpha`, not on `y`. Pre-compute it per alpha outside the target loop to avoid redundant work.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fastridge.py`:

```python
def test_ridge_loocv_2d_coef_shape():
    X, y, _ = _data()
    Y = np.column_stack([y, -y, 0.5 * y])
    est = RidgeLOOCV().fit(X, Y)
    assert est.coef_.shape == (3, 5)
    assert est.alpha_.shape == (3,)
    assert est.sigma_square_.shape == (3,)


def test_ridge_loocv_2d_column_matches_1d():
    X, y, _ = _data()
    Y = np.column_stack([y, -y])
    est_2d = RidgeLOOCV().fit(X, Y)
    est_1d = RidgeLOOCV().fit(X, y)
    np.testing.assert_allclose(est_2d.coef_[0], est_1d.coef_, rtol=1e-10)
    np.testing.assert_allclose(est_2d.alpha_[0], est_1d.alpha_, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_fastridge.py::test_ridge_loocv_2d_coef_shape tests/test_fastridge.py::test_ridge_loocv_2d_column_matches_1d -v
```

Expected: FAIL

- [ ] **Step 3: Implement 2D branch in `RidgeLOOCV.fit`**

Replace the entire `RidgeLOOCV.fit` method with:

```python
def fit(self, x, y):
    n, p = x.shape

    a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
    b_x = x.std(axis=0) if self.normalize else np.ones(p)
    x = (x - a_x) / b_x

    if np.isscalar(self.alphas):
        alpha_min, alpha_max = self.alpha_range_GMLNET(x, y)
        self.alphas_ = self.alpha_log_grid(alpha_min, alpha_max, self.alphas)
    else:
        self.alphas_ = self.alphas

    u, s, v_trans = svd(x, full_matrices=False)
    r = u * s  # shape (n, rank)

    # Pre-compute hat-diagonal for each alpha (shared across all targets)
    h_per_alpha = []
    for alpha in self.alphas_:
        z = u * (s ** 2 / (s ** 2 + alpha))
        h_per_alpha.append((z * u).sum(axis=1))

    if y.ndim == 2:
        n_targets = y.shape[1]
        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(n_targets)
        b_y = y.std(axis=0) if self.normalize else np.ones(n_targets)
        y_norm = (y - a_y) / b_y
        c_mat = (u.T @ y_norm) * s[:, None]

        loo_mse_mat = np.zeros((n_targets, len(self.alphas_)))
        for t in range(n_targets):
            c_t = c_mat[:, t]
            for i, alpha in enumerate(self.alphas_):
                beta_t = c_t / (s ** 2 + alpha)
                err = y_norm[:, t] - r.dot(beta_t)
                loo_mse_mat[t, i] = np.mean((err / (1 - h_per_alpha[i])) ** 2)

        i_stars = np.argmin(loo_mse_mat, axis=1)
        self.alpha_ = self.alphas_[i_stars]
        self.loo_mse_ = loo_mse_mat

        beta_mat = np.empty((len(s), n_targets))
        for t in range(n_targets):
            beta_mat[:, t] = c_mat[:, t] / (s ** 2 + self.alpha_[t])
        self.coef_ = (v_trans.T @ beta_mat).T * b_y[:, None] / b_x
        self.sigma_square_ = loo_mse_mat[np.arange(n_targets), i_stars] * b_y ** 2
        self.intercept_ = a_y - self.coef_ @ a_x
        return self

    # --- 1D path (unchanged logic) ---
    a_y = y.mean() if self.fit_intercept else 0.0
    b_y = y.std() if self.normalize else 1.0
    y = (y - a_y) / b_y
    c = u.T.dot(y) * s

    self.loo_mse_ = np.zeros_like(self.alphas_)
    for i in range(len(self.alphas_)):
        beta = c / (s ** 2 + self.alphas_[i])
        err = y - r.dot(beta)
        self.loo_mse_[i] = np.mean((err / (1 - h_per_alpha[i])) ** 2)

    i_star = np.argmin(self.loo_mse_)
    self.alpha_ = self.alphas_[i_star]
    beta = c / (s ** 2 + self.alpha_)
    beta = v_trans.T.dot(beta)
    self.sigma_square_ = self.loo_mse_[i_star] * b_y ** 2
    self.coef_ = beta * b_y / b_x
    self.intercept_ = a_y - self.coef_.dot(a_x)
    return self
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ fastridge.py --doctest-modules -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

```bash
git add fastridge.py tests/test_fastridge.py
git commit -m "feat: add RidgeLOOCV multi-target support via 2D y dispatch"
git push origin dev
```
