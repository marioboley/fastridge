# Usage Example Cleanup and Local Test Workflow Design

**Date:** 2026-04-05
**Scope:** Clean up the README usage example, establish a local test workflow via `pytest.ini` and a project virtual environment, add `pytest-codeblocks` for README testing, split CI into two environment-based jobs, and fix the missing `pandas` dependency in `requirements.txt`.

## Goals

1. README usage example runs without error, demonstrates the full API (`fit`, `predict`, RMSE), and uses only package dependencies (numpy + fastridge) so it is testable in both CI environments.
2. `pytest` with no arguments runs all tests locally.
3. CI has two jobs mapping to two distinct environments: package-only and project-wide.
4. Local virtual environment setup is documented and works for both venv and conda users.

## Changes

### `requirements.txt`

- Add `pandas` (used in `Analysis/` scripts but currently missing)
- Add `pytest-codeblocks` alongside `pytest`

### `.gitignore`

Add `.venv/` and `.vscode/`.

### `pytest.ini`

```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md
```

Running bare `pytest` locally runs both the module doctest and the README code block.

### `README.md`

Three changes, grouped:

**1. Usage example** — replace the current example with a clean version that:
- Imports only `numpy` and `fastridge` (no `pandas`, no `sklearn`)
- Generates a synthetic dataset with numpy
- Fits both `RidgeEM` and `RidgeLOOCV`
- Calls `predict` on both
- Computes and prints RMSE for both

Using synthetic data keeps the example self-contained and testable with package dependencies only. The diabetes dataset import is removed.

**2. Package Installation section** (for users who want to use fastridge in their own project):

```bash
pip install fastridge
```

Or directly from the repo:

```bash
pip3 install git+https://github.com/marioboley/fastridge.git
```

(`pip` or `pip3` depending on the local Python setup.)

**3. Project Setup section** (for contributors and anyone working with the analysis code):

Create a virtual environment at the project root, install all project dependencies, and install the package in editable mode so that `import fastridge` works from any subdirectory:

```bash
python3 -m venv .venv
source .venv/bin/activate   # or: conda create/activate for Anaconda users
pip3 install -r requirements.txt
pip3 install -e .
```

Run the test suite:

```bash
pytest
```

### `.github/workflows/ci.yml`

Rename existing job and add a second:

**Job 1: `package-test`** (existing job, renamed)
- Install package deps only: `pip install .`
- Install test runner: `pip install pytest`
- Run explicitly, bypassing `pytest.ini`: `pytest --doctest-modules fastridge.py`

**Job 2: `project-test`** (new)
- Install project deps: `pip install -r requirements.txt`
- Install package: `pip install .`
- Run: `pytest` (picks up `pytest.ini`, runs doctest + codeblocks)

Both jobs use Python 3.13. `pytest-codeblocks` is only needed in Job 2.

### `CLAUDE.md`

Update setup section to reference the Project Setup section in README. Note that `pytest` runs all tests. Remove the current bare `pip install -e .` instruction which lacks context.

## Out of Scope

- Jupyter notebook execution in CI
- Performance benchmark
- CI matrix for multiple Python versions
- Testing `requirements.txt` lower bounds against old package versions
