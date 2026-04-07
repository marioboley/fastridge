# Experiments Folder Establishment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a flat `experiments/` folder with a refactored `double_asymptotic_trends.ipynb`, rng-threaded `problems.py` and `experiments.py`, figure output to `output/`, and pytest coverage.

**Architecture:** Copy and refactor `problems.py` and `experiments.py` from `Analysis/Simulated_data/` to thread a `numpy.random.Generator` through all sampling calls. Copy `plotting.py` verbatim. Create the notebook as a refactored copy of `increasing_p.ipynb` with seeds fixed and `plt.savefig` enabled. Update `pytest.ini` and `.gitignore`. `Analysis/Simulated_data/` is left intact.

**Tech Stack:** numpy, scipy, matplotlib, nbmake, pytest

---

### Task 1: Create folder structure and copy plotting.py

**Files:**
- Create: `experiments/plotting.py`
- Create: `output/.gitkeep`

- [ ] **Step 1: Create folders and copy plotting.py**

```bash
mkdir -p experiments output
cp Analysis/Simulated_data/plotting.py experiments/plotting.py
touch output/.gitkeep
```

- [ ] **Step 2: Update .gitignore**

Add to `.gitignore`:
```
output/*
!output/.gitkeep
!output/*.pdf
```

- [ ] **Step 3: Commit**

```bash
git add experiments/plotting.py output/.gitkeep .gitignore
git commit -m "feat: create experiments/ and output/ folder structure"
```

---

### Task 2: Create refactored problems.py

**Files:**
- Create: `experiments/problems.py`

The key change: all sampling functions accept an explicit `rng` parameter (`numpy.random.Generator`). When `rng=None`, a fresh unseeded generator is created. `linear_problem.rvs` also accepts `rng`.

`scipy.stats` distributions that accept `random_state` (e.g. `wishart`, `multivariate_normal`) are passed the `rng` directly. `numpy` operations use `rng` methods.

- [ ] **Step 1: Create `experiments/problems.py`**

```python
import numpy as np
from numpy.random import choice
from scipy.stats import wishart, multivariate_normal, norm, uniform


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
    from eye_covariates import eye_covariates  # kept for completeness, not used in experiments
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, eye_covariates(p))


def random_problem(p, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = p if r is None else r
    x_cov = wishart.rvs(p, np.eye(p), random_state=rng)
    x_mu = rng.multivariate_normal(np.zeros(p), np.eye(p))
    x_dist = multivariate_normal(x_mu, x_cov)
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, x_dist)
```

Note: `random_multiple_means_problem` references `eye_covariates` which was in the original `problems.py` — include it in the class definition in the file (copy from `Analysis/Simulated_data/problems.py` lines 18-26). The import above is illustrative; put the class directly in the file.

- [ ] **Step 2: Fix eye_covariates inclusion**

The final `experiments/problems.py` should include `eye_covariates` as a class (not imported). Add it before `multivariate_bernoulli`:

```python
class eye_covariates:

    def __init__(self, p):
        self.p = p

    def rvs(self, n, random_state=None):
        I = np.eye(self.p)
        rnd_idx = np.random.choice(self.p, size=n % self.p)
        return np.row_stack((n // self.p * (I,) + (I[rnd_idx],)))
```

And fix `random_multiple_means_problem` to not import it:

```python
def random_multiple_means_problem(p=100, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    r = p if r is None else r
    x_dist = eye_covariates(p)
    beta = random_sparse_vector(p, r, sigma_beta, rng=rng)
    return linear_problem(beta, sigma_eps, x_dist)
```

- [ ] **Step 3: Verify manually**

```bash
cd experiments
python3 -c "
import numpy as np
import problems
rng = np.random.default_rng(1)
prob = problems.random_problem(10, rng=rng)
x, y = prob.rvs(20, rng=rng)
print('x shape:', x.shape, 'y shape:', y.shape)
"
cd ..
```

Expected: `x shape: (20, 10) y shape: (20,)`

- [ ] **Step 4: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add rng-threaded problems.py to experiments/"
```

---

### Task 3: Create refactored experiments.py

**Files:**
- Create: `experiments/experiments.py`

Key changes from `Analysis/Simulated_data/experiments.py`:
- Remove unused `from scipy.stats import norm`
- Add `seed=None` to `Experiment.__init__`, store `self.rng = np.random.default_rng(seed)`
- Pass `rng=self.rng` to all `prob.rvs()` calls in `run()`

- [ ] **Step 1: Create `experiments/experiments.py`**

```python
import time
import copy
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar


class ParameterMeanSquaredError:

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.coef_ - prob.beta)**2).mean()

    @staticmethod
    def __str__():
        return 'parameter_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{\beta}-\beta\|^2/p$'


class PredictionMeanSquaredError:

    @staticmethod
    def __call__(est, prob, x, y):
        return ((est.predict(x) - y)**2).mean()

    @staticmethod
    def __str__():
        return 'prediction_mean_squared_errors'

    @staticmethod
    def symbol():
        return r'$\|\hat{y}-y\|^2/m$'


class RegularizationParameter:

    @staticmethod
    def __call__(est, prob, x, y):
        return est.alpha_

    @staticmethod
    def __str__():
        return 'lambda'

    @staticmethod
    def symbol():
        return r'$\lambda$'


class NumberOfIterations:

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


class VarianceAbsoluteError:

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


class FittingTime:

    @staticmethod
    def __call__(est, prob, x, y):
        return est.fitting_time_

    @staticmethod
    def __str__():
        return 'fitting_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{fit}$ [s]'


parameter_mean_squared_error = ParameterMeanSquaredError()
prediction_mean_squared_error = PredictionMeanSquaredError()
regularization_parameter = RegularizationParameter()
number_of_iterations = NumberOfIterations()
variance_abs_error = VarianceAbsoluteError()
fitting_time = FittingTime()

default_stats = [parameter_mean_squared_error, prediction_mean_squared_error,
                 regularization_parameter, number_of_iterations, fitting_time]


class Experiment:

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
```

- [ ] **Step 2: Verify manually**

```bash
cd experiments
python3 -c "
import numpy as np
import problems
from experiments import Experiment, parameter_mean_squared_error
from fastridge import RidgeEM

rng = np.random.default_rng(1)
prob = problems.random_problem(5, rng=rng)
probs = [prob]
est = RidgeEM(fit_intercept=False)
exp = Experiment(probs, [est], [20], 2, seed=1)
exp.run()
print('MSE shape:', exp.parameter_mean_squared_errors_.shape)
"
cd ..
```

Expected: `MSE shape: (2, 1, 1, 1)`

- [ ] **Step 3: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: add rng-threaded experiments.py to experiments/"
```

---

### Task 4: Create double_asymptotic_trends.ipynb

**Files:**
- Create: `experiments/double_asymptotic_trends.ipynb`

This is a copy of `Analysis/Simulated_data/increasing_p.ipynb` with these changes:
- Cell 2 (`exp0`): add `seed=1` to `Experiment(...)`, update imports to not use `from experiments import *`
- Cell 5 (`exp1`, `skip-execution`): add `seed=1` to `Experiment(...)`
- Cell 6 (`exp1` viz, `skip-execution`): uncomment `plt.savefig`

- [ ] **Step 1: Copy notebook**

```bash
cp Analysis/Simulated_data/increasing_p.ipynb experiments/double_asymptotic_trends.ipynb
```

- [ ] **Step 2: Update cell 2 (exp0) — imports and seed**

Open `experiments/double_asymptotic_trends.ipynb`. Replace cell 2 source with:

```python
import numpy as np
import problems
from experiments import Experiment, parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time
from matplotlib import pyplot as plt
from plotting import plot_metrics
from fastridge import RidgeEM, RidgeLOOCV

ps = [100, 200]
rng = np.random.default_rng(1)
probs = [problems.random_problem(p, rng=rng) for p in ps]
ns = [100, 200, 300, 400, 500]

ridgeEM = RidgeEM(fit_intercept=False)
ridgeCV_GLM = RidgeLOOCV(alphas=100, fit_intercept=False)
ridgeCV_fixed = RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10), fit_intercept=False)

estimators = [ridgeEM, ridgeCV_fixed, ridgeCV_GLM]
est_names = ['EM', 'CV_fix', 'CV_glm']

exp0 = Experiment(probs, estimators, ns, 10, est_names, seed=1)
exp0.run()
```

- [ ] **Step 3: Update cell 5 (exp1) — seed**

Replace cell 5 source with:

```python
ps1 = [50, 100, 200, 400, 800]
rng1 = np.random.default_rng(1)
probs1 = [problems.random_problem(p, rng=rng1) for p in ps1]
ns1 = [50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]

exp1 = Experiment(probs1, estimators, ns1, 100, est_names, seed=1)
exp1.run()
```

- [ ] **Step 4: Update cell 6 (exp1 viz) — uncomment savefig**

Replace the `plt.savefig` line in cell 6:

```python
plt.savefig('../output/paper2023_figure2.pdf', dpi=600, bbox_inches="tight")
```

(Remove the `#` comment prefix.)

- [ ] **Step 5: Commit**

```bash
git add experiments/double_asymptotic_trends.ipynb
git commit -m "feat: add double_asymptotic_trends notebook to experiments/"
```

---

### Task 5: Update pytest.ini and verify

**Files:**
- Modify: `pytest.ini`

- [ ] **Step 1: Update pytest.ini**

Change:
```ini
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake Analysis/Simulated_data/increasing_p.ipynb
```

To:
```ini
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake Analysis/Simulated_data/increasing_p.ipynb --nbmake experiments/double_asymptotic_trends.ipynb
```

Note: keep the old notebook in pytest for now — it lives in `Analysis/` which is untouched.

- [ ] **Step 2: Run pytest and verify**

```bash
source .venv/bin/activate
pytest
```

Expected: all tests pass including both notebooks. Runtime ~20s (exp0 cells only, exp1 tagged skip-execution).

- [ ] **Step 3: Commit**

```bash
git add pytest.ini
git commit -m "feat: add double_asymptotic_trends to pytest nbmake"
```

---

### Task 6: Push to dev

- [ ] **Step 1: Push**

```bash
git push origin dev
```
