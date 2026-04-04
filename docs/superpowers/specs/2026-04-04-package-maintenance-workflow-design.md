# Package Maintenance Workflow Design

**Date:** 2026-04-04
**Scope:** Add declared package dependencies to `setup.py` and establish a minimal CI workflow that verifies the package installs and functions correctly in a clean environment.

## Goal

Establish a test cycle before any further modernisation of the packaging infrastructure. The CI workflow must verify that the declared package dependencies are sufficient — i.e., the package installs and its core functionality runs using only what `install_requires` declares.

## Changes

### `setup.py`

Three changes:
- Replace `from distutils.core import setup` with `from setuptools import setup` — required for Python 3.12+ where `distutils` was removed
- Add `install_requires=['numpy>=1.21.5', 'scipy>=1.8.1']`, matching the lower bounds already declared in `requirements.txt`
- Bump version from `v1.0.0` to `v1.1.0` at the start of development on `dev`

### `requirements.txt`

Add `pytest` as a project-level dependency (not a package dependency).

### `fastridge.py` — module-level docstring

Add a module-level docstring containing a doctest covering both `RidgeEM` and `RidgeLOOCV`. The doctest uses a synthetic dataset with large n (~500) and small noise (sigma=0.05) so that shrinkage is negligible and coefficient recovery is reliable. Assertions use `np.allclose(coef_, beta, atol=0.1)` returning `True`, which is exact doctest output while tolerating floating point variation. The doctest imports only `numpy` and `fastridge`, validating that declared dependencies are sufficient.

### `.github/workflows/ci.yml`

Triggers on push to `dev` and push to `main`. Steps:

1. Checkout repository
2. Set up Python 3.13
3. `pip install .` — installs the package and its declared dependencies in a clean environment (this is the dependency check)
4. `pip install pytest` — installs the test runner explicitly, separate from package dependencies; `requirements.txt` is not used in CI to avoid polluting the clean environment with project-only dependencies (matplotlib, scikit-learn, etc.)
5. `pytest --doctest-modules fastridge.py`

### Branching

Create a `dev` branch from `main`. Day-to-day development happens on `dev`; CI runs on every push to `dev`. Merging to `main` is done by pushing directly; CI then runs on `main` as well.

## Implementation Order (TDD)

The order matters to preserve a genuine RED state:

1. Create `dev` branch and bump version to `v1.1.0` in `setup.py`
2. Add `.github/workflows/ci.yml` and push to `dev` — CI runs but has nothing to test yet
3. Add module-level docstring with doctest to `fastridge.py` and push to `dev` — **RED**: CI fails because `install_requires` is absent, so numpy is not installed in the clean environment and the import fails
4. Add `install_requires` to `setup.py` (and fix `distutils` import) and push to `dev` — **GREEN**: CI passes
5. Push to `main`

## Out of Scope

- Modernising `setup.py` to `pyproject.toml`
- Fixing `requirements.txt` (removing `fastprogress`, adding `pandas`)
- Removing dead imports from `fastridge.py`
- Multi-version Python matrix
- Testing soundness of `requirements.txt` against analysis code
