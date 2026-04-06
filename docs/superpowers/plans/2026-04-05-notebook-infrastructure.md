# Notebook Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `increasing_p.ipynb` runnable in the project `.venv` via VSCode and verifiable via `pytest` using a lightweight experiment cell.

**Architecture:** Restructure `increasing_p.ipynb` so a fast `exp0` (imports + light params, ~8s) runs in pytest, followed by the original heavy `exp1` tagged `skip-execution`. Add `ipykernel` and `pytest-notebook` to `requirements.txt` and configure `pytest.ini` to run the notebook with output diffs ignored. CI `project-test` job requires no changes — it already runs `pip install -r requirements.txt` and `pytest`.

**Tech Stack:** `pytest-notebook>=0.10`, `ipykernel>=6.0`

---

### Task 1: Add dependencies to requirements.txt and install pytest-notebook

**Files:**
- Modify: `requirements.txt`

Note: `ipykernel` is already installed in `.venv`. Still add it to `requirements.txt` so the CI environment gets it too.

- [ ] **Step 1: Edit `requirements.txt`**

Change the tail from:
```
pytest>=7.0
pytest-codeblocks>=0.14
```
to:
```
pytest>=7.0
pytest-codeblocks>=0.14
pytest-notebook>=0.10
ipykernel>=6.0
```

- [ ] **Step 2: Install `pytest-notebook`**

```bash
source .venv/bin/activate
pip install pytest-notebook>=0.10
```

Expected: installed successfully, no errors.

---

### Task 2: Restructure the notebook

**Files:**
- Modify: `Analysis/Simulated_data/increasing_p.ipynb`

Target cell layout:
- 0: markdown (unchanged)
- 1: `exp0` — imports + light params + `exp0.run()` — runs in pytest
- 2: `exp0` visualization — runs in pytest
- 3: `exp1` — original heavy params + `exp1.run()` — tagged `skip-execution`
- 4: `exp1` visualization — tagged `skip-execution`
- 5: quick `RidgeLOOCV` fit (currently cell 4, unchanged)
- 6: `test_loocv.alphas_` (currently cell 5, unchanged)

- [ ] **Step 1: Replace cell 1 (imports) with `exp0` cell**

Replace the content of cell 1 with (use the exact `ns` and rep count you confirmed locally — 20 reps, `ps=[100, 200]`, trimmed `ns`):

```python
import problems
from experiments import *
from fastridge import RidgeEM, RidgeLOOCV

ps = [100, 200]
probs = [problems.random_problem(p) for p in ps]
ns = [100, 200, 300, 400, 600, 800]

ridgeEM = RidgeEM(fit_intercept=False)
ridgeCV_GLM = RidgeLOOCV(alphas=100, fit_intercept=False)
ridgeCV_fixed = RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10), fit_intercept=False)

estimators = [ridgeEM, ridgeCV_fixed, ridgeCV_GLM]
est_names = ['EM', 'CV_fix', 'CV_glm']

exp0 = Experiment(probs, estimators, ns, 20, est_names)
exp0.run()
```

- [ ] **Step 2: Insert `exp0` visualization as new cell 2**

Insert a new cell after cell 1:

```python
from importlib import reload
import plotting
reload(plotting)
from plotting import plot_metrics
from matplotlib import pyplot as plt

prob_idx = [0, 1]
fig, axs = plot_metrics(exp0, [parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time], prob_idx=prob_idx, figsize=(6, 6))
axs[0, 0].set_yscale('log')
axs[1, 0].set_yscale('log')
axs[2, 0].set_ylabel('$k$ (or $T$)')
for i in range(len(prob_idx)):
    axs[0, i].set_title(f'$p={ps[prob_idx[i]]}$')
axs[0, 0].legend()
plt.show()
```

- [ ] **Step 3: Tag the original `exp1` cell (now cell 3) with `skip-execution`**

Content of the cell stays as-is. Add the tag:
- In VSCode: select the cell → "..." menu → "Add Cell Tag" → type `skip-execution`

- [ ] **Step 4: Tag the original `exp1` visualization (now cell 4) with `skip-execution`**

Same — content unchanged, add `skip-execution` tag.

---

### Task 3: Configure pytest for notebook execution

**Files:**
- Modify: `pytest.ini`

- [ ] **Step 1: Update `pytest.ini`**

Current:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md
```

New:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nb-test-files Analysis/Simulated_data/increasing_p.ipynb
nb_diff_ignore =
    /cells/*/outputs
    /cells/*/execution_count
```

- [ ] **Step 2: Run pytest and verify**

```bash
source .venv/bin/activate
pytest
```

Expected: all existing tests pass, `exp0` and its visualization cell execute without error, `exp1` cells skipped. Total runtime under ~30 seconds.

---

### Task 4: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add requirements.txt pytest.ini Analysis/Simulated_data/increasing_p.ipynb
git commit -m "feat: add notebook testing via pytest-notebook, restructure increasing_p with light exp0"
```
