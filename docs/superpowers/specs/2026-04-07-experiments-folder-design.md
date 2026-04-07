---
name: Experiments Folder Establishment
description: Create experiments/ folder with restructured increasing_p notebook, rng-threaded support modules, and output/ for figures
type: project
---

# Experiments Folder Establishment Design

## Goal

Establish a flat `experiments/` folder containing a restructured `double_asymptotic_trends.ipynb` (formerly `increasing_p.ipynb`) with reproducible random seeding, figure output to `output/`, and pytest coverage. This is the first step in gradually migrating all notebooks out of `Analysis/`.

## Folder Structure

```
experiments/
  double_asymptotic_trends.ipynb (copy + refactored from Analysis/Simulated_data/increasing_p.ipynb)
  problems.py                    (copy + refactored from Analysis/Simulated_data/)
  experiments.py                 (copy + refactored from Analysis/Simulated_data/)
  plotting.py                    (copy from Analysis/Simulated_data/, no changes needed)
output/
  .gitkeep
```

`Analysis/Simulated_data/` is left intact (no files to change there).

`output/` is gitignored except `.gitkeep` and committed figures (`.pdf` files).

## Module Refactoring

### `problems.py`

All functions that generate random data currently use `scipy.stats` distributions with implicit global random state. Replace with explicit `numpy.random.Generator` (`rng`) parameter throughout:

- `random_problem(p, r=None, sigma_beta=1.0, sigma_eps=0.5, rng=None)` — default `rng=None` uses `np.random.default_rng()` (unseeded). Replace:
  - `wishart.rvs(p, np.eye(p))` → `rng.multivariate_normal` is not sufficient for Wishart; use `scipy.stats.wishart.rvs(p, np.eye(p), random_state=rng)`
  - `multivariate_normal.rvs(np.zeros(p))` → `rng.multivariate_normal(np.zeros(p), np.eye(p))`
  - `multivariate_normal(x_mu, x_cov)` distribution stored in `linear_problem` — pass `rng` to `rvs` call
  - `random_sparse_vector`: replace `multivariate_normal.rvs` and `choice` with `rng` equivalents
- `linear_problem.rvs(number, rng=None)` — pass `rng` through to sampling
- Same treatment for `random_sparse_factor_problem` and `random_multiple_means_problem`

**Breaking changes with real benefits are acceptable**, as all callers will be in new refactored copies of old notebooks.

### `experiments.py`

- Remove `from scipy.stats import norm` (unused)
- `Experiment.__init__` adds `seed=None` parameter; stores `self.rng = np.random.default_rng(seed)`
- `Experiment.run()` passes `self.rng` to `prob.rvs(n, rng=self.rng)` and `prob.rvs(test_size, rng=self.rng)` for all sampling calls
- Estimator cloning: keep `clone(est, safe=False)` from `sklearn.base`

### `plotting.py`

Copied as-is — no changes required for this project.

## Notebook: `double_asymptotic_trends.ipynb`

Copy of `increasing_p.ipynb` renamed, with the following changes:

**Cell 1 (`exp0`):** Add `seed=1` to `Experiment(...)` constructor.

**Cell 2 (`exp0` visualization):** No change.

**Cell 3 (`exp1`, `skip-execution`):** Add `seed=1` to `Experiment(...)` constructor.

**Cell 4 (`exp1` visualization, `skip-execution`):** Uncomment `plt.savefig('../output/paper2023_figure2.pdf', ...)`.

**Cells 5–6 (test_loocv):** Remove — these were scratch cells not relevant to the notebook's purpose.

## pytest.ini

Update `--nbmake` path to new notebook:
```ini
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake experiments/double_asymptotic_trends.ipynb
```

## .gitignore

Add:
```
output/*
!output/.gitkeep
!output/*.pdf
```

## Out of Scope

- `Analysis/Simulated_data/` cleanup (done after subsequent notebooks are migrated)
- `plotting.py` refactoring (deferred to when `unimodality_convexity` is migrated)
- Any changes to the existing `increasing_p.ipynb` test in `Analysis/`
