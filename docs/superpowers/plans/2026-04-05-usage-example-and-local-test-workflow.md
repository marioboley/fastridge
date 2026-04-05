# Usage Example Cleanup and Local Test Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the README usage example, add `pytest.ini` so bare `pytest` runs all tests, add a second CI job for project-wide testing, and document the local virtual environment setup.

**Architecture:** `pytest.ini` defines the default test suite (`--doctest-modules fastridge.py --codeblocks README.md`). The README usage example uses only numpy + fastridge so it passes in both CI environments. CI has two jobs: `package-test` (clean install, explicit doctest only) and `project-test` (full requirements, bare `pytest`).

**Tech Stack:** Python 3.13, pytest, pytest-codeblocks, GitHub Actions

---

## File Map

| File | Change |
|------|--------|
| `pytest.ini` | Create — defines default addopts |
| `requirements.txt` | Add `pandas>=1.3`, `pytest-codeblocks>=0.14` |
| `.gitignore` | Add `.vscode/` |
| `README.md` | Replace usage example; add Package Installation and Project Setup sections |
| `.github/workflows/ci.yml` | Rename existing job, add `project-test` job |
| `CLAUDE.md` | Update setup and workflow sections |

---

### Task 1: Add pytest.ini and update README usage example (RED)

**Files:**
- Create: `pytest.ini`
- Modify: `README.md`

- [ ] **Step 1: Create pytest.ini**

Create `pytest.ini` at the project root:

```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md
```

- [ ] **Step 2: Replace the README usage example**

Replace the entire `## Usage` section code block in `README.md` with:

````markdown
## Usage
```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV

# generate synthetic regression data
rng = np.random.default_rng(0)
n, p = 100, 10
beta_true = rng.standard_normal(p)
X_train = rng.standard_normal((n, p))
y_train = X_train @ beta_true + 0.1 * rng.standard_normal(n)
X_test = rng.standard_normal((50, p))
y_test = X_test @ beta_true + 0.1 * rng.standard_normal(50)

# fit using EM algorithm (no cross-validation required)
em = RidgeEM()
em.fit(X_train, y_train)
y_pred_em = em.predict(X_test)
rmse_em = np.sqrt(np.mean((y_test - y_pred_em) ** 2))
print(f'RidgeEM    RMSE: {rmse_em:.4f}')

# fit using fast LOOCV
loocv = RidgeLOOCV()
loocv.fit(X_train, y_train)
y_pred_loocv = loocv.predict(X_test)
rmse_loocv = np.sqrt(np.mean((y_test - y_pred_loocv) ** 2))
print(f'RidgeLOOCV RMSE: {rmse_loocv:.4f}')
```
````

- [ ] **Step 3: Run pytest to confirm RED**

```bash
pytest
```

Expected: error — `pytest-codeblocks` is not installed:

```
ERROR: unrecognized arguments: --codeblocks README.md
```

- [ ] **Step 4: Commit**

```bash
git add pytest.ini README.md
git commit -m "test: add pytest.ini and replace usage example with synthetic data"
```

---

### Task 2: Install pytest-codeblocks and fix requirements.txt (GREEN)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add missing dependencies to requirements.txt**

Replace the contents of `requirements.txt` with:

```
numpy>=1.21.5
scipy>=1.8.1
scikit-learn>=1.0.2
matplotlib>=3.5.2
fastprogress>=1.0.3
pandas>=1.3
pytest>=7.0
pytest-codeblocks>=0.14
```

- [ ] **Step 2: Install pytest-codeblocks into local environment**

```bash
pip3 install pytest-codeblocks
```

- [ ] **Step 3: Run pytest to confirm GREEN**

```bash
pytest
```

Expected output (both tests pass):

```
collected 2 items

fastridge.py::fastridge PASSED
README.md::README.md PASSED

2 passed
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "fix: add pandas and pytest-codeblocks to requirements"
```

---

### Task 3: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add .vscode/ to .gitignore**

`.venv` is already present. Add `.vscode/` directly after it in the Environments section:

```
# Environments
.env
.venv
.vscode/
env/
venv/
ENV/
env.bak/
venv.bak/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .vscode/ to .gitignore"
```

---

### Task 4: Update CI workflow

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Replace ci.yml with two-job workflow**

```yaml
name: Tests

on:
  push:
    branches:
      - dev
      - main

jobs:
  package-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install .
      - run: pip install pytest
      - run: pytest --doctest-modules fastridge.py

  project-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: pip install -r requirements.txt
      - run: pip install .
      - run: pytest
```

Note: `package-test` passes its command explicitly to bypass `pytest.ini` — it must not run `--codeblocks README.md` since `pytest-codeblocks` is not installed in that environment.

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: split into package-test and project-test jobs"
git push
```

- [ ] **Step 3: Verify both jobs GREEN on GitHub**

Go to Actions tab. Both `package-test` and `project-test` should pass.

---

### Task 5: Update README with installation sections

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add Package Installation and Project Setup sections**

Add the following after the `## Citation` section at the end of `README.md`:

```markdown
## Package Installation

For users who want to use fastridge in their own project:

```bash
pip install fastridge
```

Or directly from the repository:

```bash
pip3 install git+https://github.com/marioboley/fastridge.git
```

(`pip` or `pip3` depending on the local Python setup.)

## Project Setup

For contributors and anyone working with the analysis code. Create a virtual environment at the project root, install all project dependencies, and install the package in editable mode so that `import fastridge` works from any subdirectory:

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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Package Installation and Project Setup sections to README"
```

---

### Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Setup and Development Workflow sections**

Replace the current `## Setup` and `## Development Workflow` sections with:

```markdown
## Setup

See the **Project Setup** section in `README.md` for environment setup instructions (works for both venv and conda).

## Development Workflow

- Day-to-day work goes on the `dev` branch; `main` is kept green
- CI runs on push to both `dev` and `main` via `.github/workflows/ci.yml` — two jobs: `package-test` (package deps only, doctest) and `project-test` (full requirements, all tests)
- Run all tests: `pytest`
```

- [ ] **Step 2: Commit and push**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md setup and workflow sections"
git push
```

---

### Task 7: Merge to main

- [ ] **Step 1: Merge dev into main and push**

```bash
git checkout main
git merge dev
git push
git checkout dev
```

- [ ] **Step 2: Verify both CI jobs GREEN on main**

Go to Actions tab. Both `package-test` and `project-test` should pass on `main`.
