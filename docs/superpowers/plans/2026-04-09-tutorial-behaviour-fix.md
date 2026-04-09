# Tutorial Behaviour Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `tutorial.ipynb` figures to match the original notebook output, and commit the approved conditional LaTeX change.

**Architecture:** Three targeted changes applied together: fix test normalization in `RidgePathExperiment.run()`, restore in-place pre-normalization in `tutorial.ipynb` cell 4, and commit the conditional LaTeX detection already in the working tree. All verified by pytest before a single commit.

**Tech Stack:** numpy, sklearn, matplotlib, shutil, nbmake, pytest

---

### Task 1: Fix test normalization in RidgePathExperiment

**Files:**
- Modify: `experiments/experiments.py` (lines 182–183 of `run()`)

- [ ] **Step 1: Apply the fix**

In `experiments/experiments.py`, replace the two lines that normalize test data using train stats:

```python
        x_te = (self.x_test - a_x) / b_x
        y_te = (self.y_test - a_y) / b_y
```

with test-stats-based normalization:

```python
        a_x_te = self.x_test.mean(axis=0) if self.fit_intercept else np.zeros(p)
        a_y_te = self.y_test.mean() if self.fit_intercept else 0.0
        b_x_te = self.x_test.std(axis=0) if self.normalize else np.ones(p)
        b_y_te = self.y_test.std() if self.normalize else 1.0
        x_te = (self.x_test - a_x_te) / b_x_te
        y_te = (self.y_test - a_y_te) / b_y_te
```

- [ ] **Step 2: Verify the class works**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'experiments')
from experiments import RidgePathExperiment
import numpy as np
rng = np.random.default_rng(1)
x_tr = rng.normal(size=(30, 5))
y_tr = rng.normal(size=30)
x_te = rng.normal(size=(200, 5)) * 2 + 5   # deliberately different scale
y_te = rng.normal(size=200) * 2 + 5
exp = RidgePathExperiment(x_tr, y_tr, x_te, y_te, np.logspace(-3, 3, 10),
                          fit_intercept=True, normalize=True).run()
# true_risk_ should be ~O(1) since y_te is normalized by its own std
print('true_risk_ range:', exp.true_risk_.min(), exp.true_risk_.max())
assert exp.true_risk_.max() < 10, 'true_risk_ unexpectedly large — test normalization may still be wrong'
print('ok')
"
```

Expected: values in a reasonable range (not inflated by the difference in scale between train and test).

---

### Task 2: Apply cell 4 pre-normalization to tutorial.ipynb

**Files:**
- Modify: `experiments/tutorial.ipynb` cell `cell-4`

Note: the notebook must NOT be open in VSCode when editing. Close it first.

- [ ] **Step 1: Read the notebook** (required before NotebookEdit)

Read `experiments/tutorial.ipynb` to confirm current cell-4 content and get the cell ID.

- [ ] **Step 2: Apply the edit**

Replace cell `cell-4` source with:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)

a_x, a_y = x_train.mean(axis=0), y_train.mean()
b_x, b_y = x_train.std(axis=0), y_train.std()
x_train = (x_train - a_x) / b_x
y_train = (y_train - a_y) / b_y

alphas = np.logspace(-5, 5, 400, endpoint=True, base=10)

path_exp = RidgePathExperiment(x_train, y_train, x_test, y_test, alphas,
                               fit_intercept=True, normalize=True).run()

plot_pathway_risk(path_exp, variable_names=[f.upper() for f in diabetes.feature_names],
                  best_lambda=True, dpi=100, save_file='../output/tutorial_figure1.pdf')
```

---

### Task 3: Verify and commit all changes together

**Files:**
- `experiments/experiments.py`
- `experiments/tutorial.ipynb`
- `experiments/plotting.py` (already modified in working tree — conditional LaTeX)
- `experiments/plotting2d.py` (already modified in working tree — conditional LaTeX)

- [ ] **Step 1: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
pytest
```

Expected:
```
5 passed in ...s
```

If `tutorial.ipynb` fails, check the error. Common causes:
- Notebook still open in VSCode (close it)
- Cell 4 edit not saved (re-run NotebookEdit)
- LaTeX not found on CI but test runs locally (the conditional LaTeX fix handles this)

- [ ] **Step 2: Manual visual verification**

Run the tutorial notebook locally end-to-end (e.g. via VSCode or `jupyter nbconvert --to notebook --execute`) and compare each figure against the original:

| Figure | Original cell | Tutorial cell | What to check |
|---|---|---|---|
| Figure 1 (pathway risk) | cell 4 | cell 4 | coefficient path shape and scale, prediction risk curve scale |
| Figure 2 (LOOCV risk) | cell 7 | cell 7 | LOOCV curve, EM vertical line position, true risk curve scale |
| Figure 3 (EM contour) | cell 12 | cell 12 | contour range, scatter points within plot bounds |
| Figure 4 (marginal profiles) | cell 15 | cell 15 | true risk curve shape, marginal posterior profile shape |

Only proceed to commit once all four figures match the originals (modulo font differences from conditional LaTeX).

- [ ] **Step 3: Commit**

```bash
git add experiments/experiments.py experiments/tutorial.ipynb experiments/plotting.py experiments/plotting2d.py
git commit -m "fix: restore tutorial figure behaviour and add conditional LaTeX detection"
```

- [ ] **Step 3: Push dev and merge to main**

```bash
git push origin dev
git checkout main
git merge dev
git push origin main
git checkout dev
```

Also commit and push the spec and plan docs:

```bash
git add docs/superpowers/specs/2026-04-09-tutorial-behaviour-fix-design.md docs/superpowers/plans/2026-04-09-tutorial-behaviour-fix.md
git commit -m "docs: add tutorial behaviour fix spec and plan"
git push origin dev
git checkout main
git merge dev
git push origin main
git checkout dev
```
