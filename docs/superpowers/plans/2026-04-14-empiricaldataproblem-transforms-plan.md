# EmpiricalDataProblem Transforms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `transforms` parameter to `EmpiricalDataProblem` that applies ordered ufunc-style column transforms inside `get_X_y()`.

**Architecture:** Single parameter `transforms: list[tuple[str, callable]]` added to `EmpiricalDataProblem.__init__`. Applied after NaN handling in `get_X_y()`. Class docstring added to document all parameters and host the new doctests. The existing module-level doctests are left in place.

**Tech Stack:** `experiments/problems.py`, pytest doctest-modules (already configured for this file).

---

## Files

- Modify: `experiments/problems.py` — add class docstring, add `transforms` parameter, apply in `get_X_y()`

---

### Task 1: Write failing doctests, implement transforms, verify and commit

**Files:** `experiments/problems.py:39-60`

- [ ] **Step 1: Read the current file**

Read `experiments/problems.py` in full to confirm line numbers and current content before editing.

- [ ] **Step 2: Replace the `EmpiricalDataProblem` class with the documented, transforms-aware version**

Replace the entire class (lines 39–60) with:

```python
class EmpiricalDataProblem:
    """A prediction problem defined by a dataset and a target variable.

    Optionally, can also define pre-processing steps of column dropping,
    missing value handling, and column transformations (applied in this
    order).

    Parameters
    ----------
    dataset : str
        Name of the dataset as registered in data.DATASETS.
    target : str
        Name of the target column.
    drop : list of str, optional
        Column names to drop before returning X. Columns absent from the
        dataset are skipped with a warning.
    nan_policy : {'drop_rows', 'drop_cols'} or None, optional
        How to handle remaining NaN values after dropping rows where the
        target is NaN. 'drop_rows' drops any row with a NaN; 'drop_cols'
        drops any column with a NaN. None (default) leaves NaNs in place.
    transforms : list of (str, callable) pairs, optional
        Ordered sequence of column transforms applied after NaN handling.
        Each pair ``(column_name, fn)`` applies ``fn`` to the named column
        in-place; ``fn`` must map a ``pd.Series`` to a ``pd.Series`` of the
        same length (numpy ufuncs satisfy this). Raises ``ValueError`` if a
        named column is absent from the DataFrame at transform time.

    >>> import numpy as np
    >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
    >>> X, y = diabetes.get_X_y()
    >>> X.shape
    (442, 10)
    >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
    ...                                     transforms=[('target', np.log)])
    >>> X_log, y_log = diabetes_log.get_X_y()
    >>> np.allclose(y_log.values, np.log(y.values))
    True
    >>> diabetes_bad = EmpiricalDataProblem('diabetes', 'target',
    ...                              transforms=[('nonexistent', np.log)])
    >>> diabetes_bad.get_X_y()
    Traceback (most recent call last):
        ...
    ValueError: Column 'nonexistent' not found in dataset 'diabetes' at transform time.
    """

    def __init__(self, dataset, target, drop=None, nan_policy=None, transforms=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []
        self.nan_policy = nan_policy
        self.transforms = transforms or []

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
        for col, fn in self.transforms:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in dataset '{self.dataset}' at transform time."
                )
            df[col] = fn(df[col])
        return df.drop(columns=[self.target]), df[self.target]
```

- [ ] **Step 3: Run the doctests to verify they pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -30
```

Expected: all doctests pass, including the three new ones (`X_base.shape`, `np.allclose`, `ValueError`).

- [ ] **Step 4: Run the full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add transforms parameter to EmpiricalDataProblem"
```
