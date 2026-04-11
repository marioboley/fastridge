# NaN Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `nan_policy` parameter to `EmpiricalDataProblem` so that autompg, automobile, and facebook can be included in real-data experiments, replicating legacy n_train counts.

**Architecture:** `EmpiricalDataProblem.get_X_y()` unconditionally drops rows where the target is NaN, then optionally drops rows where any feature is NaN depending on `nan_policy`. The three affected datasets are uncommented in the notebook with `nan_policy='drop_rows'`. `automobile.csv` is committed to the repo so CI-facing doctests run without a network fetch.

**Tech Stack:** Python, pandas (`dropna`), pytest doctests, nbmake.

---

### Task 1: Track automobile.csv in git

**Files:**
- Modify: `datasets/.gitignore`

- [ ] **Step 1: Add automobile.csv to the gitignore allowlist**

Open `datasets/.gitignore`. It currently reads:
```
*
!.gitignore
!abalone.csv
!airfoil.csv
!concrete.csv
!forest.csv
!student.csv
!yacht.csv
```

Add one line so it becomes:
```
*
!.gitignore
!abalone.csv
!airfoil.csv
!automobile.csv
!concrete.csv
!forest.csv
!student.csv
!yacht.csv
```

- [ ] **Step 2: Stage and commit**

```bash
git add datasets/.gitignore datasets/automobile.csv
git commit -m "feat: track automobile.csv in git for CI doctests"
```

---

### Task 2: Add nan_policy to EmpiricalDataProblem (TDD)

**Files:**
- Modify: `experiments/problems.py` (module docstring + `EmpiricalDataProblem` class)

- [ ] **Step 1: Add failing doctests to the module docstring**

The current module docstring in `experiments/problems.py` ends after the yacht doctest. Extend it with two new tests for `nan_policy`:

```python
"""
Problem classes for simulated and empirical data experiments.

>>> p = EmpiricalDataProblem('diabetes', 'target')
>>> X, y = p.get_X_y()
>>> X.shape
(442, 10)
>>> p_drop = EmpiricalDataProblem('yacht', 'Residuary_resistance', drop=[])
>>> X2, y2 = p_drop.get_X_y()
>>> X2.shape
(308, 6)
>>> p_none = EmpiricalDataProblem('automobile', 'price')
>>> X_none, y_none = p_none.get_X_y()
>>> X_none.shape[0]
201
>>> p_drop2 = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
>>> X_drop2, y_drop2 = p_drop2.get_X_y()
>>> X_drop2.shape[0]
159
"""
```

- [ ] **Step 2: Run doctests to confirm they fail**

```bash
cd experiments && python -m doctest problems.py -v 2>&1 | grep -A2 "automobile"
```

Expected: `TypeError` — `__init__() got an unexpected keyword argument 'nan_policy'`

- [ ] **Step 3: Implement nan_policy**

Replace the `EmpiricalDataProblem` class in `experiments/problems.py` with:

```python
class EmpiricalDataProblem:

    def __init__(self, dataset, target, drop=None, nan_policy=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []
        self.nan_policy = nan_policy

    def get_X_y(self):
        df = get_dataset(self.dataset)
        missing = [c for c in self.drop if c not in df.columns]
        if missing:
            warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        df = df.dropna(subset=[self.target])
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
        return df.drop(columns=[self.target]), df[self.target]
```

- [ ] **Step 4: Run doctests to confirm they pass**

```bash
pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -10
```

Expected: all doctests PASS, including the two new automobile cases.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add nan_policy to EmpiricalDataProblem, always drop target-NaN rows"
```

---

### Task 3: Update notebook task specs

**Files:**
- Modify: `experiments/real_data.ipynb` (cell-2 preview, cell-4 full experiment)

**Important:** Close the notebook in VSCode before editing — the extension rewrites the file on save and causes conflicts with NotebookEdit.

- [ ] **Step 1: Update preview cell (cell-2)**

Replace the source of cell-2 with the following. Changes from current:
- Uncomment and add `nan_policy='drop_rows'` to `autompg`, `automobile`, `facebook`
- Update the facebook comment to note `like` and `share` as candidate targets
- Insert the three datasets in n_train-descending order relative to the others (autompg≈274, automobile≈111, facebook≈346 — facebook goes between airfoil and concrete)

```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import run_real_data_experiments
from problems import EmpiricalDataProblem

problems = [
    EmpiricalDataProblem('abalone',    'Rings'),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure'),
    EmpiricalDataProblem('facebook',   'Total Interactions', nan_policy='drop_rows'),  # 'like' and 'share' are also candidate targets
    EmpiricalDataProblem('concrete',   'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',   'target'),
    EmpiricalDataProblem('forest',     'area'),
    EmpiricalDataProblem('autompg',    'mpg',   nan_policy='drop_rows'),
    # EmpiricalDataProblem('parkinsons', 'motor_UPDRS', drop=['total_UPDRS']),  # expensive
    # EmpiricalDataProblem('real_estate', 'Y house price of unit area'),        # expensive
    EmpiricalDataProblem('student',    'G3'),
    EmpiricalDataProblem('yacht',      'Residuary_resistance'),   # much lower r2 (0.65 vs. 0.97)
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows'),
    # EmpiricalDataProblem('crime', 'ViolentCrimesPerPop'),     # expensive
]

estimators = {
    'EM':     RidgeEM(),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
}

results = run_real_data_experiments(problems, estimators, n_iterations=10, seed=1, verbose=True)
print()

rows = []
for problem, data_result in zip(problems, results):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = data_result['EM']['p']
    row['n_train']  = data_result['EM']['n_train']
    row['n:p']      = data_result['EM']['n_train'] / data_result['EM']['p']
    rows.append(row)
pd.DataFrame(rows).sort_values('n_train', ascending=False)
```

- [ ] **Step 2: Update full experiment cell (cell-4)**

Replace the source of cell-4. Changes from current:
- Uncomment and add `nan_policy='drop_rows'` to `autompg`, `automobile`, `facebook`
- Update the facebook comment

```python
problems_full = [
    EmpiricalDataProblem('abalone',          'Rings'),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('automobile',       'price',                              nan_policy='drop_rows'),
    EmpiricalDataProblem('autompg',          'mpg',                                nan_policy='drop_rows'),
    # EmpiricalDataProblem('crime',          'ViolentCrimesPerPop'),             # SVM non-converging for some run
    EmpiricalDataProblem('boston',           'medv'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',         'target'),
    EmpiricalDataProblem('facebook',         'Total Interactions',                 nan_policy='drop_rows'),  # 'like' and 'share' are also candidate targets
    EmpiricalDataProblem('forest',           'area'),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=['GT_turbine_decay']),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=['GT_compressor_decay']),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=['total_UPDRS']),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=['motor_UPDRS']),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area'),
    EmpiricalDataProblem('student',          'G3'),
    EmpiricalDataProblem('yacht',            'Residuary_resistance'),
]

estimators_full = {
    'EM':     RidgeEM(),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
}

results_full = run_real_data_experiments(problems_full, estimators_full,
                                         n_iterations=100, seed=1, verbose=True)
print()

rows_full = []
for problem, data_result in zip(problems_full, results_full):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = data_result['EM']['p']
    row['n_train']        = data_result['EM']['n_train']
    row['n:p']            = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_full.append(row)
pd.DataFrame(rows_full).sort_values('n_train', ascending=False)
```

- [ ] **Step 3: Run the preview cell to verify no crashes**

In a terminal (not the notebook), run:
```bash
cd experiments && source ../.venv/bin/activate && jupyter nbconvert --to notebook --execute real_data.ipynb --ExecutePreprocessor.timeout=300 --output /tmp/real_data_test.ipynb 2>&1 | tail -5
```

Expected: execution completes without ValueError. All three new datasets appear in the output table with correct n_train values (facebook≈346, autompg≈274, automobile≈111).

- [ ] **Step 4: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: uncomment autompg, automobile, facebook with nan_policy='drop_rows'"
```

---

### Task 4: Run full test suite

- [ ] **Step 1: Run pytest**

```bash
cd /path/to/fastridge && source .venv/bin/activate && pytest
```

Expected: all tests pass. The `--nbmake` run of `real_data.ipynb` will skip the cell-4 full experiment (it is tagged `skip-execution`); the preview cell runs and must complete without error.

- [ ] **Step 2: If tests pass, proceed to finishing-a-development-branch**

Use superpowers:finishing-a-development-branch to merge to main.
