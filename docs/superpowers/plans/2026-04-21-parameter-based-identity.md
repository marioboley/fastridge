# Parameter-Based Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-written boilerplate in problem/transform classes with `@dataclass(frozen=True)`, add `BaseEstimator`/`RegressorMixin` to estimators, and switch cache key derivation to `joblib.hash()`.

**Architecture:** Three independent changes applied in sequence: (1) setup.cfg consolidation, (2) sklearn compliance for estimators in `fastridge.py`, (3) dataclass conversion of problem/transform classes in `experiments/problems.py` with call-site updates in tests and notebooks.

**Tech Stack:** Python dataclasses (stdlib), `sklearn.base.BaseEstimator`/`RegressorMixin`, `joblib.hash()`

---

## File Structure

- Modify: `setup.cfg` — add full metadata and `install_requires` including `scikit-learn>=1.2`
- Modify: `setup.py` — reduce to stub
- Modify: `fastridge.py` — add imports and inheritance for `BaseEstimator`, `RegressorMixin`; remove `RidgeEM.__repr__`; add `score()` doctest
- Modify: `experiments/problems.py` — convert `PolynomialExpansion`, `OneHotEncodeCategories`, `EmpiricalDataProblem` to `@dataclass(frozen=True)`; add `import dataclasses`; update `NEURIPS2023` call sites; update `NEURIPS2023_D2`/`NEURIPS2023_D3` to use `dataclasses.replace()`
- Modify: `tests/test_problems.py` — remove repr test; rewrite `__new__` hack test; update `drop=[...]` to tuples
- Modify: `experiments/real_data.ipynb` — update `_OHE`, `drop=`, `x_transforms=` to tuples
- Modify: `experiments/real_data_neurips2023.ipynb` — same

---

### Task 1: Consolidate setup.cfg and add sklearn dependency

**Files:**
- Modify: `setup.cfg`
- Modify: `setup.py`

- [ ] **Step 1: Replace setup.cfg content**

```ini
[metadata]
name = fastridge
version = v1.1.0
description = Fast and robust approach to ridge regression with simultaneous estimation of model parameters and hyperparameter tuning within a Bayesian framework via expectation-maximization (EM).
author = Mario Boley
author_email = mario.boley@monash.edu
url = https://github.com/marioboley/fastridge.git
keywords = Ridge regression, EM
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent
description-file = README.md

[options]
py_modules = fastridge
install_requires =
    numpy>=1.21.5
    scipy>=1.8.1
    scikit-learn>=1.2
```

- [ ] **Step 2: Reduce setup.py to stub**

```python
from setuptools import setup
setup()
```

- [ ] **Step 3: Verify package installs cleanly**

Run: `pip install -e . --dry-run`
Expected: no errors, sklearn listed as dependency

- [ ] **Step 4: Commit**

```bash
git add setup.cfg setup.py
git commit -m "chore: consolidate setup.cfg; add scikit-learn>=1.2 dependency"
```

---

### Task 2: Add sklearn compliance to estimators

**Files:**
- Modify: `fastridge.py`

- [ ] **Step 1: Add score() doctest to RidgeEM docstring**

Add this example block to the `RidgeEM` class docstring (create one if absent — insert after the class definition line `class RidgeEM:`):

```python
    """
    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> round(RidgeEM(epsilon=0.001).fit(X, y).score(X, y), 1)
    1.0
    """
```

- [ ] **Step 2: Run doctest to verify it fails**

Run: `source .venv/bin/activate && pytest --doctest-modules fastridge.py -v`
Expected: FAIL — `AttributeError: 'RidgeEM' object has no attribute 'score'`

- [ ] **Step 3: Add imports and inheritance**

At the top of `fastridge.py`, add after `from scipy.optimize import minimize`:

```python
from sklearn.base import BaseEstimator, RegressorMixin
```

Change class declarations:

```python
class RidgeEM(BaseEstimator, RegressorMixin):
```

```python
class RidgeLOOCV(BaseEstimator, RegressorMixin):
```

- [ ] **Step 4: Remove RidgeEM.__repr__**

Delete lines 36–37:
```python
    def __repr__(self):
        return f'RidgeEM(eps={self.epsilon})'
```

- [ ] **Step 5: Run all package doctests**

Run: `source .venv/bin/activate && pytest --doctest-modules fastridge.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add fastridge.py
git commit -m "feat: add BaseEstimator and RegressorMixin to RidgeEM and RidgeLOOCV"
```

---

### Task 3: Convert PolynomialExpansion to dataclass

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add dataclasses import**

At the top of `experiments/problems.py`, add after `import warnings`:

```python
import dataclasses
from dataclasses import dataclass
```

- [ ] **Step 2: Remove repr doctests from PolynomialExpansion docstring**

In the `PolynomialExpansion` docstring, remove these two lines and their expected output:

```python
    >>> repr(PolynomialExpansion(2))
    'PolynomialExpansion(2)'
    >>> repr(PolynomialExpansion(2, max_entries=1000))
    'PolynomialExpansion(2, max_entries=1000)'
```

Keep the equality and set doctests (`==`, `!=`, `len({...})`).

- [ ] **Step 3: Convert PolynomialExpansion to dataclass**

Replace the `PolynomialExpansion` class definition (keeping `__call__` unchanged):

```python
@dataclass(frozen=True)
class PolynomialExpansion:
    """Callable value object that applies polynomial feature expansion.

    Parameters
    ----------
    degree : int
        Polynomial degree passed to PolynomialFeatures.
    max_entries : int, optional
        Maximum total entries (n * p_expanded) before interaction columns are
        subsampled; linear columns are always kept. Default 50_000_000.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    >>> pe = PolynomialExpansion(2)
    >>> rng = np.random.default_rng(0)
    >>> list(pe(X, rng).columns)
    ['a', 'b', 'a^2', 'a b', 'b^2']
    >>> pe(X, rng).shape
    (3, 5)
    >>> PolynomialExpansion(2) == PolynomialExpansion(2)
    True
    >>> PolynomialExpansion(2) == PolynomialExpansion(3)
    False
    >>> len({PolynomialExpansion(2), PolynomialExpansion(2)})
    1

    With subsampling: total columns = ceil(max_entries / n).
    >>> small = PolynomialExpansion(2, max_entries=9)
    >>> result = small(X, np.random.default_rng(0))
    >>> result.shape[1]
    3
    >>> 'a' in result.columns and 'b' in result.columns
    True
    """
    degree: int
    max_entries: int = 50_000_000

    def __call__(self, X, rng):
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
            p_budget = int(np.ceil(self.max_entries / n))
            pnew = max(0, min(len(interaction_cols), p_budget - len(linear_cols)))
            sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
            return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
        return X_poly
```

- [ ] **Step 4: Update NEURIPS2023_D2 and NEURIPS2023_D3 to use dataclasses.replace()**

Replace `NEURIPS2023_D2`:

```python
NEURIPS2023_D2 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(2),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)
```

Replace `NEURIPS2023_D3`:

```python
NEURIPS2023_D3 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(3),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)
```

- [ ] **Step 5: Run tests**

Run: `source .venv/bin/activate && cd experiments && pytest ../tests/test_problems.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/problems.py
git commit -m "refactor: convert PolynomialExpansion to dataclass"
```

---

### Task 4: Convert OneHotEncodeCategories to dataclass

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Update _OHE definition**

Change line:
```python
_OHE = [OneHotEncodeCategories()]
```
to:
```python
_OHE = (OneHotEncodeCategories(),)
```

- [ ] **Step 2: Convert OneHotEncodeCategories to dataclass**

Replace the `OneHotEncodeCategories` class (keeping `__call__` unchanged):

```python
@dataclass(frozen=True)
class OneHotEncodeCategories:
    """Callable value object that one-hot encodes all categorical columns.

    Detects non-numeric columns via pd.api.types.is_numeric_dtype. Encodes
    them with OneHotEncoder(drop='first'), reconstructs a DataFrame. No-op
    when all columns are numeric.

    Examples
    --------
    >>> import pandas as pd
    >>> enc = OneHotEncodeCategories()
    >>> rng = np.random.default_rng(0)
    >>> X_num = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    >>> enc(X_num, rng).equals(X_num)
    True
    >>> X_cat = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': [1.0, 2.0, 3.0]})
    >>> result = enc(X_cat, rng)
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

    def __call__(self, X, rng):
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
```

- [ ] **Step 3: Run tests**

Run: `source .venv/bin/activate && cd experiments && pytest ../tests/test_problems.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add experiments/problems.py
git commit -m "refactor: convert OneHotEncodeCategories to dataclass"
```

---

### Task 5: Convert EmpiricalDataProblem to dataclass

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Replace EmpiricalDataProblem class definition**

Replace the entire `EmpiricalDataProblem` class (everything from `class EmpiricalDataProblem:` up to but not including `def n_train_from_proportion`) with:

```python
@dataclass(frozen=True)
class EmpiricalDataProblem:
    """A prediction problem defined by a dataset and a target variable.

    Optionally, can also define pre-processing steps of column dropping,
    missing value handling, and feature/target transformations (applied in
    this order).

    Parameters
    ----------
    dataset : str
        Name of the dataset as registered in data.DATASETS.
    target : str
        Name of the target column.
    drop : tuple of str, optional
        Column names to drop before returning X. Columns absent from the
        dataset are skipped with a warning.
    nan_policy : {'drop_rows', 'drop_cols'} or None, optional
        How to handle remaining NaN values after dropping rows where the
        target is NaN. 'drop_rows' drops any row with a NaN; 'drop_cols'
        drops any column with a NaN. None (default) leaves NaNs in place.
    x_transforms : tuple of callable, optional
        Ordered sequence of callables with signature ``(X, rng)`` applied to
        X after the X/y split. ``rng`` is always a ``Generator`` or
        ``RandomState``; deterministic transforms may ignore it.
        ``OneHotEncodeCategories`` and ``PolynomialExpansion`` satisfy this
        contract.
    y_transforms : tuple of callable, optional
        Ordered sequence of ``Series -> Series`` transforms applied to y
        after the X/y split. Numpy ufuncs (``np.log``, ``np.log1p``) satisfy
        this contract directly.

    Examples
    --------
    Basic usage — returns (X_train, X_test, y_train, y_test):

    >>> import numpy as np
    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X_train, X_test, y_train, y_test = diabetes.get_X_y(300)
    >>> X_train.shape
    (300, 10)

    Dropping columns:

    >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
    ...                                        drop=('Froude_number',))
    >>> X_train, _, _, _ = yacht_no_froude.get_X_y(200)
    >>> X_train.shape[1]
    5

    NaN handling:

    >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
    >>> X_train, _, _, _ = auto.get_X_y(100)
    >>> X_train.shape[0]
    100

    y_transforms:

    >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
    ...                                     y_transforms=(np.log,))
    >>> _, _, y_log, _ = diabetes_log.get_X_y(300, rng=0)
    >>> _, _, y_base, _ = diabetes.get_X_y(300, rng=0)
    >>> np.allclose(y_log.values, np.log(y_base.values))
    True

    x_transforms:

    >>> ohe = EmpiricalDataProblem('automobile', 'price',
    ...                            nan_policy='drop_rows',
    ...                            x_transforms=(OneHotEncodeCategories(),))
    >>> X_train_ohe, _, _, _ = ohe.get_X_y(100)
    >>> 'fuel-type_gas' in X_train_ohe.columns
    True

    zero_variance_filter drops constant train columns from both splits:

    >>> naval = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
    ...     drop=('GT_turbine_decay',))
    >>> naval_filt = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
    ...     drop=('GT_turbine_decay',), zero_variance_filter=True)
    >>> Xtr, _, _, _ = naval.get_X_y(50, rng=0)
    >>> std = Xtr.std()
    >>> zero_var = std[std == 0].index.tolist()
    >>> zero_var
    ['T1', 'P1']
    >>> Xtr_f, Xte_f, _, _ = naval_filt.get_X_y(50, rng=0)
    >>> list(Xtr_f.columns) == [c for c in Xtr.columns if c not in zero_var]
    True
    >>> list(Xte_f.columns) == list(Xtr_f.columns)
    True

    Value-object identity (eq and hash):

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
    >>> EmpiricalDataProblem('diabetes', 'target').drop
    ()
    >>> EmpiricalDataProblem('diabetes', 'target', drop=('bmi',)).drop
    ('bmi',)
    """
    dataset: str
    target: str
    drop: tuple = ()
    nan_policy: str = None
    x_transforms: tuple = ()
    y_transforms: tuple = ()
    zero_variance_filter: bool = False

    def get_X_y(self, n_train, rng=None):
        if not isinstance(rng, np.random.RandomState):
            rng = np.random.default_rng(rng)
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
            X = fn(X, rng)
        n_test = len(X) - n_train
        if n_test < 1:
            raise ValueError(
                f"n_train={n_train} leaves no test rows (dataset has {len(X)} rows "
                f"after preprocessing).")
        indices = rng.permutation(len(X))
        X_train = X.iloc[indices[n_test:]]
        X_test  = X.iloc[indices[:n_test]]
        y_train = y.iloc[indices[n_test:]]
        y_test  = y.iloc[indices[:n_test]]
        if self.zero_variance_filter:
            std = X_train.std()
            non_zero = std[std != 0].index
            X_train = X_train[non_zero]
            X_test = X_test[non_zero]
        return X_train, X_test, y_train, y_test
```

- [ ] **Step 2: Update NEURIPS2023 _OHE usage and drop= call sites**

In the `NEURIPS2023` frozenset, several problems pass `drop=[...]`. Update them to tuples:

```python
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'),
                         nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',
                         x_transforms=_OHE,
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=('GT_turbine_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=('GT_compressor_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=('total_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=('motor_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          'G3',
                         drop=('G1', 'G2'),
                         x_transforms=_OHE,
                         zero_variance_filter=True),
```

- [ ] **Step 3: Run tests (expect failures)**

Run: `source .venv/bin/activate && cd experiments && pytest ../tests/test_problems.py -v`
Expected: some FAIL — repr test and `__new__` hack test break. Note which tests fail.

- [ ] **Step 4: Commit**

```bash
git add experiments/problems.py
git commit -m "refactor: convert EmpiricalDataProblem to dataclass"
```

---

### Task 6: Fix test_problems.py

**Files:**
- Modify: `tests/test_problems.py`

- [ ] **Step 1: Remove test_zero_variance_filter_in_repr_only_when_true**

Delete the entire function:

```python
def test_zero_variance_filter_in_repr_only_when_true():
    assert 'zero_variance_filter' not in repr(EmpiricalDataProblem('diabetes', 'target'))
    assert repr(EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)) == \
        "EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)"
```

- [ ] **Step 2: Rewrite test_get_X_y_rng_threading_to_transforms**

Replace the `__new__` hack with a normal constructor call:

```python
def test_get_X_y_rng_threading_to_transforms():
    prob = EmpiricalDataProblem('diabetes', 'target',
                                x_transforms=(PolynomialExpansion(2, max_entries=50),))
    Xtr1, _, _, _ = prob.get_X_y(50, rng=1)
    Xtr2, _, _, _ = prob.get_X_y(50, rng=1)
    assert list(Xtr1.columns) == list(Xtr2.columns)
```

- [ ] **Step 3: Update drop=[...] to tuples throughout test file**

Change:
```python
    prob_no = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                   drop=['GT_turbine_decay'])
    prob_filt = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                     drop=['GT_turbine_decay'],
                                     zero_variance_filter=True)
```
to:
```python
    prob_no = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                   drop=('GT_turbine_decay',))
    prob_filt = EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                                     drop=('GT_turbine_decay',),
                                     zero_variance_filter=True)
```

- [ ] **Step 4: Run tests**

Run: `source .venv/bin/activate && cd experiments && pytest ../tests/test_problems.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_problems.py
git commit -m "test: update test_problems for dataclass conversion"
```

---

### Task 7: Update real_data.ipynb

**Files:**
- Modify: `experiments/real_data.ipynb`

The notebook must not be open in VSCode during editing.

- [ ] **Step 1: Read notebook to identify cells containing list-syntax x_transforms, drop, or _OHE**

Run: `Read experiments/real_data.ipynb` and note all cell IDs containing:
- `_OHE = [` (definition cell)
- `x_transforms=[` (problem definition cells)
- `drop=[` (problem definition cells)
- `_OHE + [` (concatenation cells)

- [ ] **Step 2: Update _OHE definition cell**

Using the cell ID from step 1, change:
```python
_OHE = [OneHotEncodeCategories()]
```
to:
```python
_OHE = (OneHotEncodeCategories(),)
```

- [ ] **Step 3: Update all x_transforms=[] to tuples**

For each cell containing `x_transforms=[PolynomialExpansion(N)]`, change to `x_transforms=(PolynomialExpansion(N),)`.

For each cell containing `x_transforms=_OHE + [PolynomialExpansion(N)]`, change to `x_transforms=_OHE + (PolynomialExpansion(N),)`.

- [ ] **Step 4: Update all drop=[...] to tuples**

For each cell containing `drop=['G1', 'G2']`, change to `drop=('G1', 'G2')`.

- [ ] **Step 5: Run notebook (skip expensive cells)**

Run: `source .venv/bin/activate && cd experiments && pytest --nbmake real_data.ipynb -v`
Expected: PASS (expensive cells tagged `skip-execution` are skipped by CI)

- [ ] **Step 6: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "refactor: update real_data.ipynb call sites to tuple syntax"
```

---

### Task 8: Update real_data_neurips2023.ipynb

**Files:**
- Modify: `experiments/real_data_neurips2023.ipynb`

The notebook must not be open in VSCode during editing.

- [ ] **Step 1: Read notebook to identify cells with list-syntax call sites**

Run: `Read experiments/real_data_neurips2023.ipynb` and note all cell IDs containing `x_transforms=[`, `drop=[`, or `_OHE`.

- [ ] **Step 2: Apply the same tuple updates as Task 7**

Update `_OHE` definition, `x_transforms=[...]`, `drop=[...]`, and `_OHE + [...]` concatenations to tuple syntax following the same pattern as Task 7.

- [ ] **Step 3: Run notebook**

Run: `source .venv/bin/activate && cd experiments && pytest --nbmake real_data_neurips2023.ipynb -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add experiments/real_data_neurips2023.ipynb
git commit -m "refactor: update real_data_neurips2023.ipynb call sites to tuple syntax"
```

---

### Task 9: Full test suite and push

- [ ] **Step 1: Run full test suite**

Run: `source .venv/bin/activate && cd experiments && pytest ../tests/ -v && pytest --doctest-modules ../fastridge.py -v`
Expected: all PASS

- [ ] **Step 2: Push to dev**

```bash
git push origin dev
```

Wait for CI to go green before merging to main.
