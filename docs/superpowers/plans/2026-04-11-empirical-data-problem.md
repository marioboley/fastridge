# EmpiricalDataProblem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `EmpiricalDataProblem` in `experiments/problems.py`, update `run_real_data_experiments` to accept a list of problems and return a parallel list of results, and update both notebook cells to use the new interface with naval and parkinsons expanded to two tasks each.

**Architecture:** `EmpiricalDataProblem` is a plain class holding `dataset`, `target`, and `drop`; its `get_X_y()` method is the empirical analogue of `linear_problem.rvs()`. `run_real_data_experiments` replaces the three parallel list parameters (`dataframes`, `targets`, `names`) with a single `problems` list and returns a list instead of a dict. The notebook assembles the display table by zipping problems and results.

**Tech Stack:** Python, pandas, pytest (doctest via `--doctest-modules`), nbmake

---

## File Map

| File | Change |
|---|---|
| `experiments/problems.py` | Add `EmpiricalDataProblem` class with `get_X_y()` and module doctest |
| `experiments/experiments.py` | Update `run_real_data_experiments` signature and body |
| `experiments/real_data.ipynb` | Update preview cell (cell-2) and full experiment cell (cell-4) |
| `pytest.ini` | Add `--doctest-modules experiments/problems.py` |

---

### Task 1: Add `EmpiricalDataProblem` to `experiments/problems.py`

**Files:**
- Modify: `experiments/problems.py`
- Modify: `pytest.ini`

- [ ] **Step 1: Add the class and module docstring to `experiments/problems.py`**

Insert the following at the top of `experiments/problems.py`, before the existing imports:

```python
"""
Problem classes for simulated and empirical data experiments.

>>> from data import get_dataset
>>> p = EmpiricalDataProblem('diabetes', 'target')
>>> X, y = p.get_X_y()
>>> X.shape
(442, 10)
>>> p_drop = EmpiricalDataProblem('yacht', 'Residuary_resistance', drop=[])
>>> X2, y2 = p_drop.get_X_y()
>>> X2.shape
(308, 6)
"""
```

Then add the class after the existing imports (after the `from scipy.stats import ...` line), before `class linear_problem`:

```python
class EmpiricalDataProblem:

    def __init__(self, dataset, target, drop=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []

    def get_X_y(self):
        from data import get_dataset
        df = get_dataset(self.dataset)
        df = df.drop(columns=self.drop)
        return df.drop(columns=[self.target]), df[self.target]
```

Note: `from data import get_dataset` is inlined to avoid a circular import risk and because `data` is not imported elsewhere in `problems.py`.

- [ ] **Step 2: Add `experiments/problems.py` to pytest doctests in `pytest.ini`**

Add one line to the `addopts` block in `pytest.ini`:

```ini
[pytest]
addopts =
    --doctest-modules fastridge.py
    --doctest-modules experiments/data.py
    --doctest-modules experiments/problems.py
    --codeblocks README.md
    --nbmake experiments/double_asymptotic_trends.ipynb
    --nbmake experiments/sparse_designs.ipynb
    --nbmake experiments/tutorial.ipynb
    --nbmake experiments/real_data.ipynb
```

- [ ] **Step 3: Run the new doctests to verify they pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && \
source .venv/bin/activate && \
pytest --doctest-modules experiments/problems.py -v
```

Expected output: two doctests collected and passing — one for `X.shape` and one for `X2.shape`.

- [ ] **Step 4: Commit**

```bash
git add experiments/problems.py pytest.ini
git commit -m "feat: add EmpiricalDataProblem with get_X_y to problems.py"
```

---

### Task 2: Update `run_real_data_experiments` in `experiments/experiments.py`

**Files:**
- Modify: `experiments/experiments.py`

The current signature is:
```python
def run_real_data_experiments(dataframes, targets, names, estimators={}, n_iterations=100,
                       test_prop=0.3, seed=None, polynomial=None, classification=False,
                       verbose=True):
```

The current body starts a results dict and iterates with:
```python
results = {}
for j, df in enumerate(dataframes):
    name = names[j]
    target = targets[j]
    X = df.drop([target], axis=1)
    y = df[target]
    if verbose:
        print(name, end=' ')
    ...
    results[name] = data_results
return results
```

- [ ] **Step 1: Replace the function signature, docstring, and loop setup**

Replace the entire function with the following (the body from the categorical encoding onward is unchanged):

```python
def run_real_data_experiments(problems, estimators={}, n_iterations=100,
                              test_prop=0.3, seed=None, polynomial=None,
                              classification=False, verbose=True):
    """Run repeated train/test experiments on a list of EmpiricalDataProblem instances.

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
        Each problem specifies dataset, target column, and columns to drop.
    estimators : dict mapping str to estimator
    n_iterations : int
    test_prop : float
    seed : int or None
    polynomial : int or None
    classification : bool
    verbose : bool

    Returns
    -------
    list of dict
        One result dict per problem, parallel to the input list. Each dict maps
        estimator name to a dict of aggregated metrics.
    """
    results = []
    for problem in problems:
        X, y = problem.get_X_y()

        if verbose:
            print(problem.dataset, end=' ')

        categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        if categorical_cols:
            encoded_X = encoder.fit_transform(X[categorical_cols])
            X = pd.concat([
                X.drop(categorical_cols, axis=1),
                pd.DataFrame(encoded_X, columns=encoder.get_feature_names_out(categorical_cols))
            ], axis=1)

        if polynomial is not None:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
            npoly, ppoly = X_poly.shape
            if npoly * ppoly > 35000000:
                X_poly = X_poly.drop(X.columns, axis=1)
                pnew = int(np.ceil(35000000 / npoly))
                X_poly = X_poly.iloc[:, np.random.choice(X_poly.shape[1], size=pnew, replace=False)]
                X = pd.concat([X, X_poly], axis=1)
            else:
                X = X_poly

        if verbose:
            print(f'(n={X.shape[1]}, p={X.shape[1]})', end='')

        estimator_results = {
            est_name: {'mse': [], 'r2': [], 'time': [], 'p': [], 'lambda': [], 'iter': [], 'CA': [], 'q': []}
            for est_name in estimators
        }

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_iterations):
            if verbose:
                print('.', end='')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
            std = X_train.std()
            non_zero_std_cols = std[std != 0].index
            X_train = X_train[non_zero_std_cols]
            X_test = X_test[non_zero_std_cols]

            for est_name, estimator in estimators.items():
                t0 = time.time()
                estimator.fit(X_train, y_train)
                elapsed = time.time() - t0

                if classification:
                    estimator_results[est_name]['CA'].append(estimator.score(X_test, y_test))
                    estimator_results[est_name]['p'].append(X_train.shape[1])
                    estimator_results[est_name]['q'].append(len(estimator.classes_))
                else:
                    y_pred = estimator.predict(X_test)
                    estimator_results[est_name]['mse'].append(mean_squared_error(y_test, y_pred))
                    estimator_results[est_name]['r2'].append(r2_score(y_test, y_pred))
                    estimator_results[est_name]['p'].append(len(estimator.coef_))
                    estimator_results[est_name]['lambda'].append(estimator.alpha_)

                estimator_results[est_name]['time'].append(elapsed)
                if est_name == 'EM':
                    estimator_results[est_name]['iter'].append(estimator.iterations_)

        data_results = {}
        for est_name, er in estimator_results.items():
            data_results[est_name] = {
                'mse':     np.mean(er['mse']) if er['mse'] else float('nan'),
                'r2':      np.mean(er['r2']) if er['r2'] else float('nan'),
                'time':    np.mean(er['time']),
                'p':       np.mean(er['p']),
                'n_train': int(X_train.shape[0]),
                'lambda':  np.mean(er['lambda']) if er['lambda'] else float('nan'),
                'iter':    np.mean(er['iter']) if er['iter'] else 100,
                'CA':      np.mean(er['CA']) if er['CA'] else float('nan'),
                'q':       np.mean(er['q']) if er['q'] else float('nan'),
            }
        results.append(data_results)
        if verbose:
            print()

    return results
```

- [ ] **Step 2: Run the full test suite (excluding notebooks) to verify nothing is broken**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && \
source .venv/bin/activate && \
pytest --doctest-modules fastridge.py --doctest-modules experiments/data.py \
       --doctest-modules experiments/problems.py --codeblocks README.md -v
```

Expected: all doctests pass. The nbmake tests are skipped here — the notebook still uses the old interface and will fail until Task 3.

- [ ] **Step 3: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: run_real_data_experiments takes problems list, returns parallel list"
```

---

### Task 3: Update `experiments/real_data.ipynb`

**Files:**
- Modify: `experiments/real_data.ipynb` (cell-2 preview, cell-4 full experiment)

**Before editing:** confirm the notebook is NOT open in VSCode (the extension modifies it on save, causing write conflicts with NotebookEdit).

Read the notebook first to get current cell IDs, then apply changes.

- [ ] **Step 1: Read the notebook to confirm cell IDs**

Use the Read tool on `experiments/real_data.ipynb`. Confirm:
- `cell-2` is the preview experiment cell (imports `run_real_data_experiments`, defines `tasks`)
- `cell-4` is the full experiment cell (defines `tasks_full`)

- [ ] **Step 2: Replace preview cell (cell-2)**

Announce: replacing the preview cell to use `EmpiricalDataProblem`, update `run_real_data_experiments` call, and update table assembly to zip problems and results and add a `target` column.

Apply via NotebookEdit with `cell_id="cell-2"`, new source:

```python
import numpy as np
import pandas as pd

from tabulate import tabulate
from fastridge import RidgeEM, RidgeLOOCV
from experiments import run_real_data_experiments
from problems import EmpiricalDataProblem

problems = [
    EmpiricalDataProblem('abalone',  'Rings'),
    EmpiricalDataProblem('airfoil',  'scaled-sound-pressure'),
    EmpiricalDataProblem('concrete', 'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes', 'target'),
    EmpiricalDataProblem('forest',   'area'),
    EmpiricalDataProblem('student',  'G3'),
    EmpiricalDataProblem('yacht',    'Residuary_resistance'),
]

estimators = {
    'EM':     RidgeEM(),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
}

results = run_real_data_experiments(problems, estimators, n_iterations=10, seed=1, verbose=True)
print()

table = {}
for problem, data_result in zip(problems, results):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {est: data_result[est]['r2'] for est in data_result}
    row['target']  = problem.target
    row['Speed-Up'] = cv_time / em_time
    row['p']       = data_result['EM']['p']
    row['n_train'] = data_result['EM']['n_train']
    row['n:p']     = data_result['EM']['n_train'] / data_result['EM']['p']
    table[problem.dataset] = row
tdf = pd.DataFrame(table)
print(tabulate(tdf.transpose().sort_values('n_train', ascending=False),
               headers='keys', tablefmt='psql', floatfmt='.2f'))
```

- [ ] **Step 3: Replace full experiment cell (cell-4)**

Announce: replacing the full experiment cell to use `EmpiricalDataProblem`, with naval and parkinsons each expanded to two rows.

Apply via NotebookEdit with `cell_id="cell-4"`, new source:

```python
problems_full = [
    EmpiricalDataProblem('abalone',          'Rings'),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('boston',           'medv'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',         'target'),
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

table_full = {}
for problem, data_result in zip(problems_full, results_full):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {est: data_result[est]['r2'] for est in data_result}
    row['target']        = problem.target
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']             = data_result['EM']['p']
    row['n_train']       = data_result['EM']['n_train']
    row['n:p']           = data_result['EM']['n_train'] / data_result['EM']['p']
    table_full[f"{problem.dataset}:{problem.target}"] = row
tdf_full = pd.DataFrame(table_full)
print(tabulate(tdf_full.transpose().sort_values('n_train', ascending=False),
               headers='keys', tablefmt='psql', floatfmt='.2f'))
```

Note: the full experiment table uses `f"{problem.dataset}:{problem.target}"` as the key to avoid collision between the two naval and two parkinsons rows.

- [ ] **Step 4: Run the full test suite including the preview notebook**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && \
source .venv/bin/activate && \
pytest --doctest-modules fastridge.py --doctest-modules experiments/data.py \
       --doctest-modules experiments/problems.py --codeblocks README.md \
       --nbmake experiments/real_data.ipynb -v
```

Expected: all tests pass. The full experiment cell is tagged `skip-execution` so nbmake skips it; only the preview cell runs.

- [ ] **Step 5: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: update real_data.ipynb to use EmpiricalDataProblem, expand naval and parkinsons to two tasks each"
```

---

### Task 4: Run the full test suite

- [ ] **Step 1: Run all tests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && \
source .venv/bin/activate && \
pytest -v
```

Expected: all tests pass, including nbmake for all four notebooks.

- [ ] **Step 2: If any nbmake test fails, investigate the specific notebook**

For a specific failing notebook:
```bash
pytest --nbmake experiments/<name>.ipynb -v
```

Check whether the failure is in a cell that imports from `experiments` or `problems` — if so, the notebook may need its import updated.

- [ ] **Step 3: Commit spec and plan together if not already committed**

```bash
git add docs/superpowers/specs/2026-04-11-empirical-data-problem-design.md \
        docs/superpowers/plans/2026-04-11-empirical-data-problem.md
git commit -m "docs: add EmpiricalDataProblem spec and implementation plan"
```
