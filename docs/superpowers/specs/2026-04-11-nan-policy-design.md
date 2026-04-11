---
name: NaN Policy for EmpiricalDataProblem
description: Design for nan_policy parameter on EmpiricalDataProblem to handle missing values in real-data experiments
type: project
---

# NaN Policy for EmpiricalDataProblem

## Background

Three datasets in the real-data experiment currently cause SVD crashes due to NaN values: `autompg`, `automobile`, and `facebook`. Current sources (ucimlrepo, ZIP) return proper NaN, which the SVD decomposition in the fastridge estimators cannot handle. The legacy code operated on local CSV files that are no longer in the repository. The forensic evidence (see below) strongly suggests those CSVs had rows with missing values pre-removed — the legacy code likely had no NaN handling of its own.

### NaN inventory

| Dataset | Target NaN rows | Feature NaN | Notes |
|---------|----------------|-------------|-------|
| automobile | 4 (in `price`) | ~45 rows across 6 cols (`normalized-losses`, `bore`, `stroke`, `horsepower`, `peak-rpm`, `num-of-doors`) | Both target and feature NaN |
| autompg | 0 | 6 rows (`horsepower` only) | Feature NaN only |
| facebook | 0 | 6 rows (`Paid`, `like`, `share`) | Feature NaN only |

### Forensic confirmation

Legacy `n_train` values are reproduced exactly by applying `dropna()` to current source data:

| Dataset | Legacy n_train | dropna → n_train |
|---------|---------------|-----------------|
| automobile | 111 | 111 ✓ |
| autompg | 274 | 274 ✓ |
| facebook | 346 | 346 ✓ |

This confirms the legacy CSVs were pre-cleaned (rows with any NaN removed), and `drop_rows` is the correct replication policy for all three.

---

## Design Principle

**Rows with NaN in the target are always dropped at problem level.** These rows can never enter the test set (no ground truth to evaluate against). Note: in semi-supervised settings, such rows could in principle be retained for training as unlabeled data — that extension is out of scope here.

**Rows with NaN in features only are a modelling decision.** Some estimators handle feature NaN natively; others require it to be resolved first. `nan_policy` controls what `EmpiricalDataProblem` presents to estimators:

- `None` (default): feature NaN are passed through. The estimator is responsible for handling them (e.g. via `SimpleImputer` in a sklearn Pipeline).
- `'drop_rows'`: rows with any feature NaN are also dropped at problem level, before train/test split.

Both target-NaN dropping and feature-NaN dropping occur before train/test split so that no information about dropped rows leaks into either set.

---

## Implementation

### `EmpiricalDataProblem` (`experiments/problems.py`)

Add `nan_policy` parameter:

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
        df = df.dropna(subset=[self.target])          # always drop target-NaN rows
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
        return df.drop(columns=[self.target]), df[self.target]
```

### Notebook task spec (`experiments/real_data.ipynb`)

Add `nan_policy='drop_rows'` to the three affected problems in both preview and full experiment cells:

```python
EmpiricalDataProblem('autompg',    'mpg',   nan_policy='drop_rows'),
EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows'),
EmpiricalDataProblem('facebook',   'Total Interactions', nan_policy='drop_rows'),
```

These were previously commented out. They are uncommented as part of this project.

### Doctests (`experiments/problems.py` module docstring)

`automobile` is the only dataset with NaN in both target and features, making it the most informative test case. It is added to `datasets/.gitignore` as a tracked file so the doctests run in CI without a network fetch.

Row counts (verified): 205 total → 201 after target-NaN drop → 159 after full drop.

```python
>>> p_none = EmpiricalDataProblem('automobile', 'price')
>>> X_none, y_none = p_none.get_X_y()
>>> X_none.shape[0]   # target-NaN rows always dropped; feature NaN passed through
201
>>> p_drop = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
>>> X_drop, y_drop = p_drop.get_X_y()
>>> X_drop.shape[0]   # feature-NaN rows also dropped
159
```

`datasets/.gitignore` — add:
```
!automobile.csv
```

---

## Future Steps

- **Facebook alternative targets**: `like` and `share` are currently used as features (target is `Total Interactions`). Both are engagement metrics in their own right and could be used as prediction targets in future experiments — analogous to the naval_propulsion and parkinsons multi-target treatment. Note that `like` and `share` themselves contain NaN (1 and 4 rows respectively), so they would require `nan_policy='drop_rows'` when used as targets.

- **Imputation as estimator strategy**: For `autompg` and `facebook`, where NaN appear only in features, a future improved treatment could use `nan_policy=None` and wrap estimators in a sklearn Pipeline with `SimpleImputer`, allowing imputation strategies to be compared as estimator variants on the same problem.

## Out of Scope

- Imputation strategies (`nan_policy='impute_mean'` etc.) — belong in estimator Pipelines, not in `EmpiricalDataProblem`
- Semi-supervised use of target-NaN rows
- Per-column NaN policy
- The `polynomial` parameter in `run_real_data_experiments` — a separate refactor concern
