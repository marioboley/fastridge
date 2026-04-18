# Preprocessing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move OHE and polynomial expansion from experiment runners into `EmpiricalDataProblem` via new `x_transforms`/`y_transforms` parameters, making problems fully self-contained.

**Architecture:** Replace `EmpiricalDataProblem.transforms` with `x_transforms` (DataFrame→DataFrame callables, applied to X after split) and `y_transforms` (Series→Series callables, applied to y after split). Two new value objects — `PolynomialExpansion` and `OneHotEncodeCategories` — are added to `problems.py`. Experiment runners have their OHE/polynomial blocks removed.

**Tech Stack:** Python, pandas, scikit-learn (`OneHotEncoder`, `PolynomialFeatures`), numpy, pytest (doctests via `--doctest-modules`)

---

## File Structure

- **Modify**: `experiments/problems.py`
  - Add top-level imports: `pandas`, `OneHotEncoder`, `PolynomialFeatures`
  - Add `PolynomialExpansion` class (value object, callable)
  - Add `OneHotEncodeCategories` class (value object, callable)
  - Refactor `EmpiricalDataProblem`: replace `transforms` with `x_transforms`/`y_transforms`; new `__repr__`, `__eq__`, `__hash__`, `get_X_y()`
  - Update `NEURIPS2023` problem set: migrate `transforms` → `y_transforms`, add OHE where needed
  - Replace `NEURIPS2023_D2` / `NEURIPS2023_D3` with `_with_polynomial`-based construction
- **Modify**: `experiments/experiments.py`
  - Remove OHE and polynomial expansion blocks from `EmpiricalDataExperiment.run()` and `run_real_data_experiments()`
  - Remove `polynomial` parameter from both
  - Remove now-unused `OneHotEncoder` and `PolynomialFeatures` imports

---

## Task 1: Add `PolynomialExpansion` and `OneHotEncodeCategories`

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add doctests for both classes**

Insert the following docstrings into `problems.py`, immediately before `class linear_problem:` (i.e. after the `EmpiricalDataProblem` class ends at line 149). These doctests will fail until the classes exist.

```python
class PolynomialExpansion:
    """Callable value object that applies polynomial feature expansion.

    Parameters
    ----------
    degree : int
        Polynomial degree passed to PolynomialFeatures.
    max_entries : int, optional
        Maximum total entries (n * p) before column subsampling is applied.
        Default 35_000_000. When exceeded, linear terms are always kept and
        interaction columns are subsampled deterministically.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    >>> pe = PolynomialExpansion(2)
    >>> list(pe(X).columns)
    ['a', 'b', 'a^2', 'a b', 'b^2']
    >>> pe(X).shape
    (3, 5)
    >>> PolynomialExpansion(2) == PolynomialExpansion(2)
    True
    >>> PolynomialExpansion(2) == PolynomialExpansion(3)
    False
    >>> len({PolynomialExpansion(2), PolynomialExpansion(2)})
    1
    >>> repr(PolynomialExpansion(2))
    'PolynomialExpansion(2)'
    >>> repr(PolynomialExpansion(2, max_entries=1000))
    'PolynomialExpansion(2, max_entries=1000)'
    >>> small = PolynomialExpansion(2, max_entries=10)
    >>> result = small(X)
    >>> 'a' in result.columns and 'b' in result.columns
    True
    >>> result.shape[1] < 5
    True
    """
```

```python
class OneHotEncodeCategories:
    """Callable value object that one-hot encodes all categorical columns.

    Detects non-numeric columns via pd.api.types.is_numeric_dtype. Encodes
    them with OneHotEncoder(drop='first'), reconstructs a DataFrame. No-op
    when all columns are numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> enc = OneHotEncodeCategories()
    >>> X_num = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    >>> enc(X_num).equals(X_num)
    True
    >>> X_cat = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': [1.0, 2.0, 3.0]})
    >>> result = enc(X_cat)
    >>> 'color' in result.columns
    False
    >>> 'size' in result.columns
    True
    >>> result.shape
    (3, 2)
    >>> OneHotEncodeCategories() == OneHotEncodeCategories()
    True
    >>> len({OneHotEncodeCategories(), OneHotEncodeCategories()})
    1
    >>> repr(OneHotEncodeCategories())
    'OneHotEncodeCategories()'
    """
```

- [ ] **Step 2: Run doctests to verify they fail**

```bash
cd experiments && python -m pytest --doctest-modules problems.py -v 2>&1 | grep -E "FAILED|ERROR|passed|failed"
```

Expected: failures for `PolynomialExpansion` and `OneHotEncodeCategories` (classes not defined yet).

- [ ] **Step 3: Add imports to `problems.py`**

Replace the existing imports block at the top of `experiments/problems.py`:

```python
"""
Problem classes for simulated and empirical data experiments.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wishart, multivariate_normal, uniform
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from data import get_dataset, DATASETS
```

- [ ] **Step 4: Implement `PolynomialExpansion`**

Insert the full class body after the docstring added in Step 1:

```python
class PolynomialExpansion:
    """..."""  # keep docstring from Step 1

    def __init__(self, degree, max_entries=35_000_000):
        self.degree = degree
        self.max_entries = max_entries

    def __call__(self, X):
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = pd.DataFrame(
            poly.fit_transform(X),
            columns=poly.get_feature_names_out(X.columns),
            index=X.index
        )
        n, p = X_poly.shape
        if n * p > self.max_entries:
            linear_cols = list(X.columns)
            interaction_cols = [c for c in X_poly.columns if c not in linear_cols]
            pnew = int(np.ceil(self.max_entries / n)) - len(linear_cols)
            rng = np.random.default_rng(self.degree)
            sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
            return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
        return X_poly

    def __eq__(self, other):
        return isinstance(other, PolynomialExpansion) and \
               (self.degree, self.max_entries) == (other.degree, other.max_entries)

    def __hash__(self):
        return hash((type(self).__name__, self.degree, self.max_entries))

    def __repr__(self):
        if self.max_entries == 35_000_000:
            return f'PolynomialExpansion({self.degree})'
        return f'PolynomialExpansion({self.degree}, max_entries={self.max_entries})'
```

- [ ] **Step 5: Implement `OneHotEncodeCategories`**

Insert the full class body after the docstring added in Step 1:

```python
class OneHotEncodeCategories:
    """..."""  # keep docstring from Step 1

    def __call__(self, X):
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if not cat_cols:
            return X
        enc = OneHotEncoder(drop='first', sparse_output=False)
        encoded = enc.fit_transform(X[cat_cols])
        return pd.concat([
            X.drop(columns=cat_cols),
            pd.DataFrame(encoded,
                         columns=enc.get_feature_names_out(cat_cols),
                         index=X.index)
        ], axis=1)

    def __eq__(self, other):
        return isinstance(other, OneHotEncodeCategories)

    def __hash__(self):
        return hash(type(self).__name__)

    def __repr__(self):
        return 'OneHotEncodeCategories()'
```

- [ ] **Step 6: Run doctests to verify they pass**

```bash
cd experiments && python -m pytest --doctest-modules problems.py::problems.PolynomialExpansion problems.py::problems.OneHotEncodeCategories -v
```

Expected: all new doctests pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add PolynomialExpansion and OneHotEncodeCategories value objects"
```

---

## Task 2: Refactor `EmpiricalDataProblem`

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Update the `EmpiricalDataProblem` docstring and doctests**

Replace the entire docstring of `EmpiricalDataProblem` (lines 13–113 in the original file) with the following. Key changes: `transforms` → `y_transforms`; add `x_transforms`; add `__repr__` test; remove the `ValueError` test for nonexistent column (no longer applicable).

```python
class EmpiricalDataProblem:
    """A prediction problem defined by a dataset and a target variable.

    Optionally defines preprocessing: column dropping, NaN handling, and
    per-axis transforms applied after the X/y split.

    Parameters
    ----------
    dataset : str
        Name of the dataset as registered in data.DATASETS.
    target : str
        Name of the target column.
    drop : list of str, optional
        Column names to drop before returning X.
    nan_policy : {'drop_rows', 'drop_cols'} or None, optional
        How to handle NaN values after dropping rows where the target is NaN.
    x_transforms : list of callable, optional
        Ordered transforms applied to X after the X/y split. Each callable
        must map pd.DataFrame -> pd.DataFrame. PolynomialExpansion and
        OneHotEncodeCategories satisfy this contract.
    y_transforms : list of callable, optional
        Ordered transforms applied to y after the X/y split. Each callable
        must map pd.Series -> pd.Series. Numpy ufuncs (np.log, np.log1p)
        satisfy this contract directly.

    Examples
    --------
    Basic usage:

    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X, y = diabetes.get_X_y()
    >>> X.shape
    (442, 10)
    >>> yacht = EmpiricalDataProblem('yacht', 'Residuary_resistance')
    >>> X, y = yacht.get_X_y()
    >>> X.shape
    (308, 6)
    >>> automobile = EmpiricalDataProblem('automobile', 'price')
    >>> X, y = automobile.get_X_y()
    >>> X.shape[0]
    201
    >>> list(X.index) == list(y.index) == list(range(201))
    True

    Dropping columns:

    >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...                                        drop=['Froude_number'])
    >>> X, y = yacht_no_froude.get_X_y()
    >>> X.shape
    (308, 5)

    NaN handling:

    >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
    >>> X, y = auto.get_X_y()
    >>> X.shape[0]
    159
    >>> list(X.index) == list(y.index) == list(range(159))
    True
    >>> auto_cols = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_cols')
    >>> X, y = auto_cols.get_X_y()
    >>> X.shape
    (201, 19)

    y_transforms — applied to the target after the X/y split:

    >>> import numpy as np
    >>> yacht_log = EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...                                  y_transforms=[np.log])
    >>> X, y_log = yacht_log.get_X_y()
    >>> X_base, y_base = yacht.get_X_y()
    >>> np.allclose(y_log.values, np.log(y_base.values))
    True

    x_transforms — applied to X after the X/y split:

    >>> enc_problem = EmpiricalDataProblem('automobile', 'price',
    ...                                    x_transforms=[OneHotEncodeCategories()])
    >>> X_enc, y_enc = enc_problem.get_X_y()
    >>> X_enc.shape[0]
    201
    >>> all(pd.api.types.is_numeric_dtype(X_enc[c]) for c in X_enc.columns)
    True

    Value-object identity — repr, hash and equality based on full definition:

    >>> p1 = EmpiricalDataProblem('diabetes', 'target')
    >>> p2 = EmpiricalDataProblem('diabetes', 'target')
    >>> p1 == p2
    True
    >>> p1 is p2
    False
    >>> len(frozenset({p1, p2}))
    1
    >>> p3 = EmpiricalDataProblem('yacht', 'Residuary_resistance')
    >>> len(frozenset({p1, p2, p3}))
    2
    >>> repr(p1)
    "EmpiricalDataProblem('diabetes', 'target')"
    >>> repr(p1) == repr(p2)
    True
    >>> repr(p1) == repr(p3)
    False
    >>> EmpiricalDataProblem('diabetes', 'target').drop
    ()
    >>> EmpiricalDataProblem('diabetes', 'target', drop=['bmi']).drop
    ('bmi',)
    """
```

- [ ] **Step 2: Run doctests to verify updated tests fail**

```bash
cd experiments && python -m pytest --doctest-modules problems.py::problems.EmpiricalDataProblem -v 2>&1 | tail -20
```

Expected: failures (old `transforms` parameter still present, new methods absent).

- [ ] **Step 3: Implement the new `EmpiricalDataProblem` body**

Replace the `__init__`, `__eq__`, `__hash__`, and `get_X_y()` methods (lines 115–149 of the original) with:

```python
    def __init__(self, dataset, target, drop=None, nan_policy=None,
                 x_transforms=None, y_transforms=None):
        self.dataset = dataset
        self.target = target
        self.drop = tuple(drop or [])
        self.nan_policy = nan_policy
        self.x_transforms = tuple(x_transforms or [])
        self.y_transforms = tuple(y_transforms or [])
        self._repr = (
            f'EmpiricalDataProblem({self.dataset!r}, {self.target!r}'
            + (f', drop={list(self.drop)!r}' if self.drop else '')
            + (f', nan_policy={self.nan_policy!r}' if self.nan_policy else '')
            + (f', x_transforms={list(self.x_transforms)!r}' if self.x_transforms else '')
            + (f', y_transforms={list(self.y_transforms)!r}' if self.y_transforms else '')
            + ')'
        )

    def __repr__(self):
        return self._repr

    def __eq__(self, other):
        if not isinstance(other, EmpiricalDataProblem):
            return NotImplemented
        return self._repr == other._repr

    def __hash__(self):
        return hash(self._repr)

    def get_X_y(self):
        df = get_dataset(self.dataset)
        missing = [c for c in self.drop if c not in df.columns]
        if missing:
            warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        df = df.dropna(subset=[self.target])
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
        elif self.nan_policy == 'drop_cols':
            df = df.dropna(axis=1)
        df = df.reset_index(drop=True)
        X = df.drop(columns=[self.target])
        y = df[self.target]
        for fn in self.y_transforms:
            y = fn(y)
        for fn in self.x_transforms:
            X = fn(X)
        return X, y
```

- [ ] **Step 4: Run doctests to verify they pass**

```bash
cd experiments && python -m pytest --doctest-modules problems.py::problems.EmpiricalDataProblem -v
```

Expected: all doctests pass.

- [ ] **Step 5: Run full pytest to catch any regressions**

```bash
cd experiments && python -m pytest --doctest-modules problems.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: replace transforms with x_transforms/y_transforms on EmpiricalDataProblem"
```

---

## Task 3: Update `NEURIPS2023` problem set

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Audit which datasets in `NEURIPS2023` have categorical columns**

Run the following from the `experiments/` directory to identify which datasets need `OneHotEncodeCategories()`:

```bash
cd experiments && python - <<'EOF'
import pandas as pd
from problems import NEURIPS2023
for p in sorted(NEURIPS2023, key=lambda p: p.dataset):
    X, y = p.get_X_y()
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cats:
        print(f"{p.dataset} ({p.target}): {cats[:5]}")
EOF
```

Note which datasets print output — those need `x_transforms=[OneHotEncodeCategories()]`.

- [ ] **Step 2: Update `NEURIPS2023` problem definitions**

In `experiments/problems.py`, update the `NEURIPS2023` frozenset:

- Replace `transforms=[('price', np.log)]` with `y_transforms=[np.log]` for automobile
- Replace `transforms=[('area', np.log1p)]` with `y_transforms=[np.log1p]` for forest
- Replace `transforms=[('Residuary_resistance', np.log)]` with `y_transforms=[np.log]` for yacht
- Add `x_transforms=[OneHotEncodeCategories()]` to each dataset identified in Step 1

Example — automobile before:
```python
EmpiricalDataProblem('automobile', 'price',
                     nan_policy='drop_rows',
                     transforms=[('price', np.log)]),
```

After:
```python
EmpiricalDataProblem('automobile', 'price',
                     nan_policy='drop_rows',
                     x_transforms=[OneHotEncodeCategories()],
                     y_transforms=[np.log]),
```

Example — yacht before:
```python
EmpiricalDataProblem('yacht', 'Residuary_resistance',
                     transforms=[('Residuary_resistance', np.log)]),
```

After:
```python
EmpiricalDataProblem('yacht', 'Residuary_resistance',
                     y_transforms=[np.log]),
```

- [ ] **Step 3: Verify problem count is unchanged**

```bash
cd experiments && python -c "from problems import NEURIPS2023; print(len(NEURIPS2023))"
```

Expected: same count as before (23 problems, or whatever the current count is — compare against `git stash; python -c "from problems import NEURIPS2023; print(len(NEURIPS2023))"; git stash pop`).

- [ ] **Step 4: Run doctests**

```bash
cd experiments && python -m pytest --doctest-modules problems.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: migrate NEURIPS2023 problem set to x_transforms/y_transforms"
```

---

## Task 4: Rebuild `NEURIPS2023_D2` and `NEURIPS2023_D3`

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Record current D2 and D3 problem counts**

```bash
cd experiments && python -c "
from problems import NEURIPS2023_D2, NEURIPS2023_D3
print('D2:', len(NEURIPS2023_D2))
print('D3:', len(NEURIPS2023_D3))
"
```

Note these numbers — they must be preserved.

- [ ] **Step 2: Replace `NEURIPS2023_D2` and `NEURIPS2023_D3`**

Replace the current definitions (lines 269–279 of the original `problems.py`) with:

```python
def _with_polynomial(p, degree):
    return EmpiricalDataProblem(
        p.dataset, p.target, p.drop, p.nan_policy,
        x_transforms=list(p.x_transforms) + [PolynomialExpansion(degree)],
        y_transforms=list(p.y_transforms),
    )


NEURIPS2023_D2 = frozenset(
    _with_polynomial(p, 2)
    for p in NEURIPS2023
    if 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    _with_polynomial(p, 3)
    for p in NEURIPS2023
    if 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)
```

Note: both D2 and D3 are derived from `NEURIPS2023`, not from each other, so `PolynomialExpansion` is added exactly once with the right degree.

- [ ] **Step 3: Verify counts match**

```bash
cd experiments && python -c "
from problems import NEURIPS2023_D2, NEURIPS2023_D3
print('D2:', len(NEURIPS2023_D2))
print('D3:', len(NEURIPS2023_D3))
"
```

Expected: same counts as recorded in Step 1. Also verify transforms are correct:

```bash
cd experiments && python -c "
from problems import NEURIPS2023_D2, PolynomialExpansion
p = next(iter(NEURIPS2023_D2))
print(p.x_transforms[-1])
assert p.x_transforms[-1] == PolynomialExpansion(2), 'wrong degree'
print('OK')
"
```

- [ ] **Step 4: Run doctests**

```bash
cd experiments && python -m pytest --doctest-modules problems.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: rebuild NEURIPS2023_D2/D3 using PolynomialExpansion transforms"
```

---

## Task 5: Clean up `experiments.py`

**Files:**
- Modify: `experiments/experiments.py`

- [ ] **Step 1: Remove OHE and polynomial block from `EmpiricalDataExperiment.run()`**

In `EmpiricalDataExperiment.run()`, the block starting after `X, y = problem.get_X_y()` (around line 457) and ending before `self.ns[prob_idx, 0] = ...` currently contains OHE detection and polynomial expansion. Remove it entirely. The method should go directly from `get_X_y()` to `self.ns[prob_idx, 0] = int(X.shape[0] * (1 - self.test_prop))`.

Before (lines ~457–481):
```python
X, y = problem.get_X_y()

if self.verbose:
    print(problem.dataset, end=' ')

categorical_cols = [col for col in X.columns
                    if not pd.api.types.is_numeric_dtype(X[col])]
if categorical_cols:
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded = encoder.fit_transform(X[categorical_cols])
    X = pd.concat([
        X.drop(categorical_cols, axis=1),
        pd.DataFrame(encoded,
                     columns=encoder.get_feature_names_out(categorical_cols))
    ], axis=1)

if self.polynomial is not None:
    poly = PolynomialFeatures(degree=self.polynomial, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly = pd.DataFrame(X_poly,
                          columns=poly.get_feature_names_out(X.columns))
    npoly, ppoly = X_poly.shape
    if npoly * ppoly > 35_000_000:
        X_poly = X_poly.drop(X.columns, axis=1)
        pnew = int(np.ceil(35_000_000 / npoly))
        X_poly = X_poly.iloc[
            :, np.random.choice(X_poly.shape[1], size=pnew, replace=False)]
        X = pd.concat([X, X_poly], axis=1)
    else:
        X = X_poly

self.ns[prob_idx, 0] = int(X.shape[0] * (1 - self.test_prop))
```

After:
```python
X, y = problem.get_X_y()

if self.verbose:
    print(problem.dataset, end=' ')

self.ns[prob_idx, 0] = int(X.shape[0] * (1 - self.test_prop))
```

- [ ] **Step 2: Remove `polynomial` from `EmpiricalDataExperiment.__init__`**

In `EmpiricalDataExperiment.__init__` (around line 430), remove `polynomial=None` from the signature and remove `self.polynomial = polynomial` from the body. Update the docstring to remove the `polynomial` parameter entry.

Before:
```python
def __init__(self, problems, estimators, n_iterations,
             test_prop=0.3, seed=None, polynomial=None, stats=None,
             est_names=None, verbose=True):
    ...
    self.polynomial = polynomial
```

After:
```python
def __init__(self, problems, estimators, n_iterations,
             test_prop=0.3, seed=None, stats=None,
             est_names=None, verbose=True):
```

- [ ] **Step 3: Remove OHE and polynomial block from `run_real_data_experiments()`**

Apply the same removal to `run_real_data_experiments()` (around lines 305–326). Remove the categorical detection + OHE block and the polynomial block. Remove `polynomial=None` from the function signature and from the docstring.

Before signature:
```python
def run_real_data_experiments(problems, estimators={}, n_iterations=100,
                              test_prop=0.3, seed=None, polynomial=None,
                              classification=False, verbose=True):
```

After:
```python
def run_real_data_experiments(problems, estimators={}, n_iterations=100,
                              test_prop=0.3, seed=None,
                              classification=False, verbose=True):
```

The loop body goes directly from `X, y = problem.get_X_y()` to the verbose print and then the rep loop.

- [ ] **Step 4: Remove unused imports from `experiments.py`**

Check whether `OneHotEncoder` and `PolynomialFeatures` are still used anywhere in `experiments.py` after the removals:

```bash
cd experiments && grep -n "OneHotEncoder\|PolynomialFeatures" experiments.py
```

If neither appears in any remaining code, remove them from the import line:

Before:
```python
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
```

After:
```python
# (remove line entirely if both are unused)
```

- [ ] **Step 5: Run full test suite**

```bash
cd experiments && python -m pytest --doctest-modules problems.py --doctest-modules experiments.py -v
```

Expected: all pass. If any doctest in `experiments.py` references `polynomial=`, update it to remove that argument.

- [ ] **Step 6: Run full pytest suite**

```bash
cd experiments && python -m pytest --doctest-modules problems.py --doctest-modules experiments.py --nbmake tutorial.ipynb double_asymptotic_trends.ipynb sparse_designs.ipynb -v
```

Expected: all pass. `real_data.ipynb` is tested in Task 6 after notebook cells are updated.

- [ ] **Step 7: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: remove OHE and polynomial expansion from experiment runners"
```

---

## Task 6: Update notebooks

**Files:**
- Modify: `experiments/real_data.ipynb`
- Modify: `experiments/real_data_neurips2023.ipynb`

`real_data.ipynb` defines its own problem sets inline (it is not based on the
`NEURIPS2023` sets). These lists relied on the experiment runner for OHE and
polynomial expansion. After this refactoring, each problem definition must carry
its own `x_transforms`. `OneHotEncodeCategories` is safe to add universally — it
is a no-op on numeric DataFrames. Experiment cells that pass `polynomial=N` must
also have that argument removed.

Use `NotebookEdit` with the cell IDs below. **Do not open either notebook in
VSCode while editing.**

- [ ] **Step 1: Update imports in cell `76d89b51`**

```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import EmpiricalDataExperiment
from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion
```

- [ ] **Step 2: Update `problems` (d=1) in cell `76d89b51`**

Add `x_transforms=[OneHotEncodeCategories()]` to every `EmpiricalDataProblem`
constructor call in the `problems` list. Keep the `exp = EmpiricalDataExperiment(...).run()` call unchanged (no `polynomial=` was ever passed here). Full updated cell:

```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import EmpiricalDataExperiment
from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion

problems = [
    EmpiricalDataProblem('abalone',    'Rings',                        x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',        x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength', x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('diabetes',   'target',                       x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('eye',        'y',                            x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('forest',     'area',                         x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('student',    'G3', drop=['G1', 'G2'],        x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',         x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories()]),
]

estimators = {
    'EM':     RidgeEM(),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
    'CV_glm': RidgeLOOCV(alphas=100),
}

exp = EmpiricalDataExperiment(
    problems, list(estimators.values()),
    n_iterations=10, seed=123,
    est_names=list(estimators.keys())).run()
print()
```

- [ ] **Step 3: Update `problems_d2` in cell `229399e5`**

Add `x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]` to every entry:

```python
problems_d2 = [
    EmpiricalDataProblem('abalone',    'Rings',                        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('diabetes',   'target',                       x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('eye',        'y',                            x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('forest',     'area',                         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),  # OHE interaction columns cause SVD failure
    EmpiricalDataProblem('student',    'G3', drop=['G1', 'G2'],        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
]
```

- [ ] **Step 4: Update `problems_d3` in cell `994819b1`**

Same pattern with `PolynomialExpansion(3)`:

```python
problems_d3 = [
    EmpiricalDataProblem('abalone',    'Rings',                        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('diabetes',   'target',                       x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    # EmpiricalDataProblem('eye',        'y',  ...),  # excluded in paper for d=3
    EmpiricalDataProblem('forest',     'area',                         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),  # OHE interaction columns cause SVD failure
    EmpiricalDataProblem('student',    'G3', drop=['G1', 'G2'],        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
]
```

- [ ] **Step 5: Update `problems_full` (d=1) in cell `2bfbd407`**

Add `x_transforms=[OneHotEncodeCategories()]` to every entry. Keep `estimators_full` dict and `exp_full = EmpiricalDataExperiment(...).run()` unchanged. Full updated problem list:

```python
problems_full = [
    EmpiricalDataProblem('abalone',          'Rings',                          x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',          x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('automobile',       'price',               nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows',            x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols',                               x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('ribo',             'y',                              x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('eye',              'y',                              x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('boston',           'medv',                           x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',  x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('diabetes',         'target',                         x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows',                               x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('forest',           'area',                           x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=['GT_turbine_decay'], x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=['GT_compressor_decay'], x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=['total_UPDRS'],   x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=['motor_UPDRS'],   x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',     x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('student',          'G3',                  drop=['G1', 'G2'],      x_transforms=[OneHotEncodeCategories()]),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',           x_transforms=[OneHotEncodeCategories()]),
]
```

- [ ] **Step 6: Update `problems_full_d2` in cell `b12d0253`**

Same as `problems_full` with `x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]` on every entry (preserve all existing comments):

```python
problems_full_d2 = [
    EmpiricalDataProblem('abalone',          'Rings',                          x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',          x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('automobile',       'price',               nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows',            x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols',                               x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    # EmpiricalDataProblem('ribo',             'y'),  # memory exhaustion at d=2
    EmpiricalDataProblem('eye',              'y',                              x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('boston',           'medv',                           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',  x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('diabetes',         'target',                         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows',                               x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('forest',           'area',                           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),  # OHE interaction columns cause SVD failure
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=['GT_turbine_decay'], x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=['GT_compressor_decay'], x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=['total_UPDRS'],   x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=['motor_UPDRS'],   x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',     x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('student',          'G3',                  drop=['G1', 'G2'],      x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
]
```

- [ ] **Step 7: Update `problems_full_d3` in cell `d245ce96`**

Same pattern with `PolynomialExpansion(3)` (preserve all existing comments):

```python
problems_full_d3 = [
    EmpiricalDataProblem('abalone',          'Rings',                          x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',          x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('automobile',       'price',               nan_policy='drop_rows', x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows',            x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols',                               x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    # EmpiricalDataProblem('ribo',             'y'),  # memory exhaustion at d>=2
    # EmpiricalDataProblem('eye',              'y'),  # excluded in paper for d=3
    EmpiricalDataProblem('boston',           'medv',                           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',  x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('diabetes',         'target',                         x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows',                               x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('forest',           'area',                           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),  # OHE interaction columns cause SVD failure
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=['GT_turbine_decay'], x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=['GT_compressor_decay'], x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=['total_UPDRS'],   x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=['motor_UPDRS'],   x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',     x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('student',          'G3',                  drop=['G1', 'G2'],      x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',           x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(3)]),
]
```

- [ ] **Step 8: Update `problems_large_d2` in cell `e4ee47eb`**

```python
problems_large_d2 = [
    EmpiricalDataProblem('twitter',   'V78',        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('tomshw',    'V97',        x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('blog',      'V281',       x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
    EmpiricalDataProblem('ct_slices', 'reference',  x_transforms=[OneHotEncodeCategories(), PolynomialExpansion(2)]),
]
```

Note: `problems_large` (cell `502c1187`) has no polynomial and these datasets are all numeric, so no `x_transforms` change is needed there — the OHE is a no-op and was never applied meaningfully. Leave that cell unchanged.

- [ ] **Step 9: Remove `polynomial=` from experiment cells in `real_data.ipynb`**

Cell `f6bc8769`:
```python
exp_d2 = EmpiricalDataExperiment(
    problems_d2, list(estimators.values()),
    n_iterations=10, seed=123,
    est_names=list(estimators.keys())).run()
print()
```

Cell `094956a8`:
```python
exp_d3 = EmpiricalDataExperiment(
    problems_d3, list(estimators.values()),
    n_iterations=10, seed=123,
    est_names=list(estimators.keys())).run()
print()
```

Cell `28f3974b`:
```python
exp_full_d2 = EmpiricalDataExperiment(
    problems_full_d2, list(estimators_full.values()),
    n_iterations=30, seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `a96f98ee`:
```python
exp_full_d3 = EmpiricalDataExperiment(
    problems_full_d3, list(estimators_full.values()),
    n_iterations=30, seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `300ca33b`:
```python
exp_large_d2 = EmpiricalDataExperiment(
    problems_large_d2, list(estimators_full.values()),
    n_iterations=30, seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

- [ ] **Step 10: Remove `polynomial=` from experiment cells in `real_data_neurips2023.ipynb`**

Cell `a006`:
```python
from problems import NEURIPS2023_D2

problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
exp_d2 = EmpiricalDataExperiment(
    problems_d2, estimators, n_iterations=100, seed=123,
    est_names=est_names, verbose=True).run()
print()
```

Cell `a009`:
```python
from problems import NEURIPS2023_D3

problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
exp_d3 = EmpiricalDataExperiment(
    problems_d3, estimators, n_iterations=100, seed=123,
    est_names=est_names, verbose=True).run()
print()
```

- [ ] **Step 11: Run CI notebooks**

```bash
cd experiments && python -m pytest --nbmake tutorial.ipynb double_asymptotic_trends.ipynb sparse_designs.ipynb real_data.ipynb -v
```

Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add experiments/real_data.ipynb experiments/real_data_neurips2023.ipynb
git commit -m "feat: add x_transforms to notebook problem definitions; remove polynomial= from experiment calls"
```
