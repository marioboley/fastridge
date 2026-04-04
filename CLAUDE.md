# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`fastridge` is a Python research library (NeurIPS 2023) implementing two fast ridge regression approaches:
- **`RidgeEM`**: Bayesian ridge regression via Expectation-Maximization — simultaneously estimates model coefficients and hyperparameters (`tau_square`, `sigma_square`) without cross-validation
- **`RidgeLOOCV`**: Fast Leave-One-Out Cross-Validation ridge — vectorized computation over an alpha grid using SVD

The entire library lives in a single module: [fastridge.py](fastridge.py).

## Setup

```bash
pip install -e .
```

## Development Workflow

- Day-to-day work goes on the `dev` branch; `main` is kept green
- Run tests locally: `pytest --doctest-modules fastridge.py`
- CI runs on push to both `dev` and `main` via `.github/workflows/ci.yml`

## Architecture

Both classes follow the scikit-learn estimator API (`fit(X, y)` / `predict(X)`) and share the same preprocessing pattern: optional intercept centering and feature normalization applied before SVD, then reversed when storing `coef_` and `intercept_`.

**Core computation pattern** (both classes):
1. Center/normalize `X` and `y` if configured
2. Compute thin SVD: `U, s, V^T = svd(X)`
3. Project `y` onto singular vectors: `c = U^T y * s`
4. Solve in the compressed space, avoiding explicit matrix inversion

**`RidgeEM` internals**: Iterates EM updates — E-step computes sufficient statistics `w` (expected squared coefficient norm) and `z` (expected RSS); M-step has a closed-form solution for `tau_square` and `sigma_square` under a Beta Prime prior on `tau_square` (`t2=True`, default) or a half-Cauchy prior on `tau` (`t2=False`). Convergence is checked on relative RSS change (`epsilon`). The `alpha_` attribute is a derived convenience alias for `1/tau_square`.

**`RidgeLOOCV` internals**: Vectorizes LOOCV over the alpha grid. Hat matrix diagonal `h` is computed without forming the full hat matrix: `h_i = sum_j (U_ij * s_j)^2 / (s_j^2 + alpha)`. LOO MSE is `mean((residual / (1 - h))^2)`.

## Analysis

Jupyter notebooks and experiment scripts in `Analysis/` use the library for empirical evaluation. These are research/exploration code, not part of the package.
