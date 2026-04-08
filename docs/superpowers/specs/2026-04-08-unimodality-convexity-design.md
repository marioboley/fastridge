---
name: Unimodality Convexity Notebook
description: Integrate Analysis/Unimodality_Convexity/ into experiments/ with new RidgePathExperiment class and plotting2d module
type: project
---

# Unimodality Convexity Integration Design

## Goal

Integrate `Analysis/Unimodality_Convexity/` into `experiments/` by creating `experiments/unimodality_convexity.ipynb` following the established pattern. Adds `RidgePathExperiment` to `experiments.py`, landscape plotting functions to `experiments/plotting2d.py`, and general risk plotting functions to `experiments/plotting.py`.

## Modules

### `experiments/experiments.py` — add `RidgePathExperiment`

New class alongside the existing `Experiment` class:

```python
class RidgePathExperiment:
    def __init__(self, x_train, y_train, x_test, y_test, alphas,
                 fit_intercept=True, normalize=True):
        ...
    def run(self):
        # fits Ridge over alphas on training data
        # evaluates true_risk on test data
        # stores: alphas_, coef_path_, true_risk_, ols_coef_
        return self
```

Attributes after `run()`:
- `alphas_` — shape `(n_alphas,)`, the evaluated alpha grid
- `coef_path_` — shape `(n_features, n_alphas)`, ridge coefficients per alpha
- `true_risk_` — shape `(n_alphas,)`, MSE on test set per alpha
- `ols_coef_` — shape `(n_features,)`, OLS coefficients (α=0 limit), computed on training data in the same normalized space as the path

The class handles normalization internally (controlled by `fit_intercept` and `normalize`) so `ols_coef_` and `coef_path_` are always in the same space.

### `experiments/plotting2d.py` — new module

Verbatim copy of the landscape functions from `Analysis/Unimodality_Convexity/plotting.py`:
- `plot_marg_profile(x_train, y_train, t2, ax, text, dpi)`
- `profile_marg(X, y, t2)` — computation helper
- `Q_function(x, y, sigma2, tau2)` — computation helper
- `compute_marginal_likelihood(x, y, sig2, t2, sigma2, tau2)` — computation helper
- `plot_EM_step(z, sig2, t2, sigma2, tau2, levels, ...)` — contour plot of posterior landscape

No changes to function signatures or behaviour.

### `experiments/plotting.py` — add two functions

Add `plot_lambda_risks` and `plot_pathway_risk` from `Analysis/Unimodality_Convexity/plotting.py` with minimal mechanical updates:
- `true_risk` → `true_risk_` (attribute rename to match `RidgePathExperiment`)
- `lr_coef` → `ols_coef_` (attribute rename)

No other changes to logic or signatures.

### `experiments/unimodality_convexity.ipynb` — new notebook

Adapted from `Analysis/Unimodality_Convexity/LOOCV Risk Function vs Bayesian Ridge posterior.ipynb`. All cells run during pytest (no `skip-execution` tags). Updated imports:

```python
from experiments import RidgePathExperiment
from plotting import plot_lambda_risks, plot_pathway_risk
from plotting2d import plot_marg_profile, compute_marginal_likelihood, plot_EM_step
from fastridge import RidgeEM, RidgeLOOCV
```

Call sites updated to use `RidgePathExperiment(...).run()` and renamed attributes (`true_risk_`, `ols_coef_`). Markdown cells and narrative copied verbatim. `random_state=180` for `train_test_split` stays as-is (it's a fixed data split for reproducibility, not a simulation seed).

### `pytest.ini`

Add `--nbmake experiments/unimodality_convexity.ipynb` to `addopts`.

## Cell Structure

```
Cell 0  markdown   Title + problem description
Cell 1  code       Imports + data loading
Cell 2  markdown   ## Ridge Regression
Cell 3  markdown   Ridge regression equation (math)
Cell 4  code       Pathway risk plot (RidgePathExperiment + plot_pathway_risk)
Cell 5  markdown   Narrative on LOOCV weakness
Cell 6  markdown   ## LOOCV Risk
Cell 7  code       LOOCV risk comparison (RidgePathExperiment + RidgeLOOCV + RidgeEM + plot_lambda_risks)
Cell 8  markdown   Narrative on unimodality
Cell 9  markdown   ## Bayesian Ridge
Cell 10 markdown   Bayesian model equations
Cell 11 markdown   EM algorithm description
Cell 12 code       EM step contour plot (RidgeEM + compute_marginal_likelihood + plot_EM_step)
Cell 13 markdown   Theorem 3.1
Cell 14 markdown   ### Marginal posterior profile
Cell 15 code       Marginal profile plot over n_size (loop with RidgePathExperiment + plot_marg_profile)
Cell 16 markdown   Interpretation of marginal profile
```

## Out of Scope

- `Analysis/Unimodality_Convexity/` — left untouched
- `fastridge.py`, `problems.py` — no changes
- Refactoring `plot_lambda_risks` signature (deferred — requires deeper analysis of coupling)
- `EmpiricalExperiment` generalisation (deferred — future project)
