# Package Maintenance Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a minimal CI workflow that verifies the package installs correctly using only its declared dependencies, exercised via a module-level doctest.

**Architecture:** A GitHub Actions workflow installs the package in a clean environment via `pip install .`, then runs `pytest --doctest-modules fastridge.py`. The doctest in the module-level docstring of `fastridge.py` serves as both the install test and usage documentation. The TDD sequence is strict: the doctest is committed first (RED — CI fails because `install_requires` is absent so numpy/scipy are not installed), then `setup.py` is fixed (GREEN).

**Tech Stack:** Python 3.13, pytest, GitHub Actions, setuptools

---

### Task 1: Create dev branch

- [ ] **Step 1: Create and push the dev branch**

```bash
git checkout -b dev
git push -u origin dev
```

No file changes. This establishes `dev` as the working branch for all subsequent tasks.

---

### Task 2: Add CI workflow and update requirements

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `requirements.txt`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches:
      - dev
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install .
      - run: pip install pytest
      - run: pytest --doctest-modules fastridge.py
```

- [ ] **Step 2: Add pytest to requirements.txt**

Append to `requirements.txt`:

```
pytest>=7.0
```

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml requirements.txt
git commit -m "ci: add GitHub Actions workflow and pytest to requirements"
git push
```

Go to the repository on GitHub → Actions tab and confirm the workflow run completes. It will pass trivially (zero doctests collected) — that is expected at this stage.

---

### Task 3: Add doctest to fastridge.py (RED)

**Files:**
- Modify: `fastridge.py`

- [ ] **Step 1: Add module-level docstring with doctest**

Insert the following at the very top of `fastridge.py`, before the imports:

```python
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
```

- [ ] **Step 2: Commit and push**

```bash
git add fastridge.py
git commit -m "docs: add module-level doctest covering RidgeEM and RidgeLOOCV"
git push
```

- [ ] **Step 3: Verify RED on GitHub**

Go to Actions tab. The workflow triggered by this push should fail. Expected failure: `ModuleNotFoundError: No module named 'numpy'` during collection — `pip install .` does not install numpy or scipy because `install_requires` is not yet declared in `setup.py`.

---

### Task 4: Fix setup.py (GREEN)

**Files:**
- Modify: `setup.py`

- [ ] **Step 1: Replace setup.py**

Replace the entire content of `setup.py` with:

```python
from setuptools import setup

setup(
    name = 'fastridge',
    py_modules = ['fastridge'],
    version = 'v1.1.0',
    description = 'Fast and robust approach to ridge regression with simultaneous estimation of model parameters and hyperparameter tuning within a Bayesian framework via expectation-maximization (EM). ',
    author = 'Mario Boley',
    author_email = 'mario.boley@monash.edu',
    url = 'https://github.com/marioboley/fastridge.git',
    install_requires = ['numpy>=1.21.5', 'scipy>=1.8.1'],
    keywords = ['Ridge regression', 'EM'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
```

Changes from the current file:
- `from distutils.core import setup` → `from setuptools import setup` (distutils removed in Python 3.12)
- Added `install_requires = ['numpy>=1.21.5', 'scipy>=1.8.1']`
- Version bumped from `v1.0.0` to `v1.1.0`

- [ ] **Step 2: Commit and push**

```bash
git add setup.py
git commit -m "fix: migrate to setuptools, declare install_requires, bump to v1.1.0"
git push
```

- [ ] **Step 3: Verify GREEN on GitHub**

Go to Actions tab. The latest workflow run should pass. Expected output from pytest: `1 passed` (the doctest module).

---

### Task 5: Merge to main

- [ ] **Step 1: Merge dev into main and push**

```bash
git checkout main
git merge dev
git push
```

- [ ] **Step 2: Verify GREEN on main**

Go to Actions tab. The push to `main` should trigger a run that passes.
