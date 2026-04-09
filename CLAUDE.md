# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`fastridge` is a Python research library (NeurIPS 2023) implementing two fast ridge regression approaches:
- **`RidgeEM`**: Bayesian ridge regression via Expectation-Maximization — simultaneously estimates model coefficients and hyperparameters (`tau_square`, `sigma_square`) without cross-validation
- **`RidgeLOOCV`**: Fast Leave-One-Out Cross-Validation ridge — vectorized computation over an alpha grid using SVD

The entire library lives in a single module: [fastridge.py](fastridge.py).

## Setup

See the **Project Setup** section in `README.md` for environment setup instructions (adapt depending on system using venv or conda).

## Development Workflow

- Day-to-day work goes on the `dev` branch; `main` is kept green
- CI runs on push to both `dev` and `main` via `.github/workflows/ci.yml` — two jobs: `package-test` (package deps only, doctest) and `project-test` (full requirements, all tests)
- Run all tests via `pytest` before commit
- All local Python processs use dedicated virtual environment 

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

Selected notebooks are run as part of `pytest` via `nbmake` (see `pytest.ini`). Cells tagged `skip-execution` are skipped during testing — used for heavy experiment cells that are too slow for routine runs.

**Editing notebooks:** Always use the `Read` tool (not Bash) before `NotebookEdit`, and specify `cell_id` from the Read output. The notebook must not be open in VSCode at the same time — the extension modifies the file on save, causing write conflicts. The `NotebookEdit` tool does not show a diff preview; announce the intended change in plain text before applying it.

## Refactoring Rules

When integrating or refactoring existing code (e.g. moving analysis scripts into `experiments/`), every behaviour change must be explicitly identified, communicated to the user, and approved before implementation — even changes that appear neutral or beneficial.

**Before completing any refactoring step:**
- Audit every public method or function being replaced: do the new implementations preserve exact input/output behaviour?
- Identify all differences, including subtle ones such as normalization conventions, attribute naming, data flow through notebook session state, and global matplotlib rc state.
- Present each difference to the user with options (preserve or change) before proceeding.

This applies even to "obviously correct" fixes — the user decides whether to fix or preserve existing behaviour. Undiscovered behaviour changes in notebooks are particularly costly because they are only detectable by visual inspection of figure output.
