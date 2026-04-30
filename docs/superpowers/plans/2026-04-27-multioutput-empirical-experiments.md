# Multi-Target Empirical Experiment Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `EmpiricalDataProblem` to accept tuple targets, update all metrics to aggregate correctly for multi-target fits, and update the `real_data.ipynb` notebook problem lists to use the natural multi-target datasets.

**Architecture:** Three independent change sites: `problems.py` (data model), `experiments.py` (metrics), and `real_data.ipynb` (notebook). Each task produces runnable, testable code. The `NEURIPS2023` constant in `problems.py` is intentionally left unchanged — `real_data_neurips2023.ipynb` imports it directly for published-result reproducibility; only the inline problem lists in `real_data.ipynb` are updated.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, pytest/doctest, nbmake.

---

## File map

| File | Change |
|---|---|
| `experiments/problems.py` | `EmpiricalDataProblem.target` type; `get_X_y` NaN/split logic; new doctest |
| `experiments/experiments.py` | `RegularizationParameter`, `NumberOfIterations`, `VarianceAbsoluteError`, `NumberOfFeatures` logic; class docstrings + doctests on all 8 metrics; `__str__` instance method fix |
| `experiments/real_data.ipynb` | 6 problem-list cells (student/naval/parkinsons/facebook); 6 result-display cells (`row['target']`) |

---

## Task 1: `EmpiricalDataProblem` multi-target support

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Update field type annotation and docstring**

In `experiments/problems.py`, change the `target` field and its docstring entry from:

```python
    target : str
        Name of the target column.
```
```python
    target: str
```

to:

```python
    target : str or tuple of str
        Name of the target column, or a tuple of column names for multi-target
        prediction. When a tuple is supplied, all named columns are excluded
        from X and get_X_y returns a DataFrame for y.
```
```python
    target: str | tuple
```

- [ ] **Step 2: Update `get_X_y` with `target_cols` variable**

Replace the three affected lines in `get_X_y` (currently lines 142, 148, 149):

```python
# current
        df = df.dropna(subset=[self.target])
        ...
        X = df.drop(columns=[self.target])
        y = df[self.target]
```

with:

```python
        target_cols = [self.target] if isinstance(self.target, str) else list(self.target)
        df = df.dropna(subset=target_cols)
        ...
        X = df.drop(columns=target_cols)
        y = df[self.target] if isinstance(self.target, str) else df[target_cols]
```

The `...` represents the unchanged `nan_policy` block between the two dropna calls; do not move or alter it.

- [ ] **Step 3: Add multi-target doctest**

Append to the existing `Examples` block in `EmpiricalDataProblem`, after the `zero_variance_filter` example and before the value-object identity block:

```python
    Multi-target usage — y is a DataFrame when target is a tuple:

    >>> student_mt = EmpiricalDataProblem('student', ('G1', 'G2', 'G3'),
    ...                                   x_transforms=_OHE, zero_variance_filter=True)
    >>> _, _, y_tr, _ = student_mt.get_X_y(454)
    >>> y_tr.shape
    (454, 3)
```

(The `_OHE` name is already used in the module; the doctest inherits it from the module namespace.)

- [ ] **Step 4: Run doctests to verify**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge/experiments && python -m doctest problems.py -v 2>&1 | tail -5
```

Expected: all existing tests still pass, new test passes.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: extend EmpiricalDataProblem.target to tuple for multi-target prediction"
```

---

## Task 2: Metrics — logic fixes, doctests, `__str__` fix

**Files:**
- Modify: `experiments/experiments.py`

### Step group A — logic fixes (all return Python scalars)

- [ ] **Step 1: Fix `ParameterMeanSquaredError.__call__`**

Change:
```python
        return ((est.coef_ - prob.beta)**2).mean()
```
to:
```python
        return float(((est.coef_ - prob.beta)**2).mean())
```

- [ ] **Step 2: Fix `PredictionMeanSquaredError.__call__`**

Change:
```python
        return ((est.predict(x) - y)**2).mean()
```
to:
```python
        return float(((est.predict(x) - np.asarray(y))**2).mean())
```

- [ ] **Step 3: Fix `RegularizationParameter.__call__`**

Change:
```python
        return est.alpha_
```
to:
```python
        return float(np.mean(est.alpha_))
```

- [ ] **Step 4: Fix `NumberOfIterations.__call__`**

Change the EM path from:
```python
            return est.iterations_
```
to:
```python
            return int(np.sum(est.iterations_))
```

Change the LOOCV path from:
```python
            return len(est.alphas_)
```
to:
```python
            return len(est.alphas_) * (y.shape[1] if np.ndim(y) > 1 else 1)
```

The `alphas` (no trailing `_`) path is unchanged.

- [ ] **Step 5: Fix `VarianceAbsoluteError.__call__`**

Change:
```python
            return abs(prob.sigma**2 - est.sigma_square_)
```
to:
```python
            return float(np.mean(np.abs(prob.sigma**2 - est.sigma_square_)))
```

- [ ] **Step 6: Fix `NumberOfFeatures.__call__`**

Change:
```python
            return len(est.coef_)
```
to:
```python
            return est.coef_.shape[-1]
```

- [ ] **Step 7: Fix `PredictionR2.__call__`**

Change:
```python
        return r2_score(y, est.predict(x))
```
to:
```python
        return float(r2_score(y, est.predict(x)))
```

### Step group B — `__str__` instance method fix

- [ ] **Step 8: Change `@staticmethod def __str__()` to `def __str__(self)` in all metric classes**

Apply to every metric class that has:
```python
    @staticmethod
    def __str__():
        return '...'
```

Change each occurrence to (preserving the string value):
```python
    def __str__(self):
        return '...'
```

This affects: `ParameterMeanSquaredError`, `PredictionMeanSquaredError`, `RegularizationParameter`, `NumberOfIterations`, `VarianceAbsoluteError`, `FittingTime`, `PredictionR2`, `NumberOfFeatures` — 8 classes total.

### Step group C — add class-level docstrings with doctests

For each class below, add a class docstring immediately after the `class Foo(Metric):` line. Classes that already have a docstring are extended.

- [ ] **Step 9: `ParameterMeanSquaredError`**

```python
class ParameterMeanSquaredError(Metric):
    """Mean squared error between estimated and true coefficients, averaged over all elements.

    Applicable only to synthetic experiments where prob.beta is defined.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     coef_ = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> class _P:
    ...     beta = np.array([[1.1, 1.9], [2.9, 4.1]])
    >>> parameter_mean_squared_error(_E(), _P(), None, None)
    0.01
    """
```

- [ ] **Step 10: `PredictionMeanSquaredError`**

```python
class PredictionMeanSquaredError(Metric):
    """Mean squared error between predictions and test targets, averaged over all elements.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X2 = np.eye(3)
    >>> Y2 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
    >>> est2 = Ridge(alpha=0.0001).fit(X2, Y2)
    >>> prediction_mean_squared_error(est2, None, X2, Y2) < 1e-6
    True
    """
```

- [ ] **Step 11: `RegularizationParameter`**

```python
class RegularizationParameter(Metric):
    """Mean regularization parameter alpha_ across targets.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     alpha_ = np.array([1.0, 3.0])
    >>> regularization_parameter(_E(), None, None, None)
    2.0
    """
```

- [ ] **Step 13: `NumberOfIterations`**

```python
class NumberOfIterations(Metric):
    """Total EM steps or total LOO alpha evaluations across all targets.

    Examples
    --------
    >>> import numpy as np
    >>> class _EM:
    ...     iterations_ = np.array([10, 8])
    >>> y_2t = np.zeros((100, 2))
    >>> number_of_iterations(_EM(), None, None, y_2t)
    18
    >>> class _LOOCV:
    ...     alphas_ = np.arange(10)
    >>> number_of_iterations(_LOOCV(), None, None, y_2t)
    20
    """
```

- [ ] **Step 14: `VarianceAbsoluteError`**

```python
class VarianceAbsoluteError(Metric):
    """Mean absolute error between estimated and true noise variance, averaged across targets.

    Returns NaN when the estimator has no sigma_square_ attribute.

    Examples
    --------
    >>> import numpy as np
    >>> class _E:
    ...     sigma_square_ = np.array([0.25, 0.36])
    >>> class _P:
    ...     sigma = 0.5
    >>> variance_abs_error(_E(), _P(), None, None)
    0.055
    """
```

- [ ] **Step 15: `PredictionR2` — append multi-target doctest**

`PredictionR2` already has a class docstring. Add the following example block after the existing example:

```python
    Multi-target: r2_score uses uniform_average by default, returning a scalar:

    >>> from sklearn.linear_model import Ridge
    >>> X3 = np.eye(3)
    >>> Y3 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
    >>> est3 = Ridge(alpha=0.0001).fit(X3, Y3)
    >>> prediction_r2(est3, None, X3, Y3) > 0.99
    True
```

- [ ] **Step 16: `NumberOfFeatures` — update docstring and add multi-target doctest**

Replace the current docstring:

```python
    """Returns the number of features used by the estimator (len of coef_).
```

with:

```python
    """Returns the number of features used by the estimator (last dim of coef_.shape).
```

Then append to the existing Examples block:

```python
    Multi-target: coef_ is (q, p); shape[-1] returns p correctly:

    >>> from sklearn.linear_model import Ridge
    >>> X4 = np.arange(20).reshape(10, 2).astype(float)
    >>> Y4 = np.column_stack([X4[:, 0], X4[:, 1]])
    >>> est4 = Ridge(alpha=0.0001).fit(X4, Y4)
    >>> number_of_features(est4, None, None, None)
    2
```

### Step group D — verify

- [ ] **Step 17: Run doctests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge/experiments && python -m doctest experiments.py -v 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 18: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: standardise metric return types to Python scalars; aggregate multi-target; add doctests; fix __str__"
```

---

## Task 3: Notebook — problem lists and result display

**Files:**
- Modify: `experiments/real_data.ipynb`

Use `Read` before every `NotebookEdit`. Cell IDs are stable; use them. The notebook must not be open in VSCode while editing.

**Summary of changes per problem list cell:**

| Dataset | Old entry | New entry |
|---|---|---|
| `student` | `'G3', drop=('G1', 'G2'),` | `('G1', 'G2', 'G3'),` (no `drop`) |
| `naval_propulsion` | two entries (each drops the other target) | one entry: `target=('GT_compressor_decay', 'GT_turbine_decay')`, no `drop` |
| `parkinsons` | two entries (each drops the other target) | one entry: `target=('motor_UPDRS', 'total_UPDRS')`, no `drop` |
| `facebook` | `'Total Interactions', drop=('comment', 'like', 'share'),` | `('comment', 'like', 'share'), drop=('Total Interactions',)` |

`student` appears in all 6 problem-list cells (2, 4, 7, 13, 15, 18).
`naval_propulsion`, `parkinsons`, `facebook` appear only in the full-experiment cells (13, 15, 18).

The result-display cells (3, 6, 9, 14, 17, 20) all have:
```python
row = {'dataset': problem.dataset, 'target': problem.target}
```
which must become:
```python
row = {'dataset': problem.dataset, 'target': ', '.join(problem.target) if isinstance(problem.target, tuple) else problem.target}
```

- [ ] **Step 1: Read and update cell 2 (preview, d=1) — `student`**

Read cell id `76d89b51`. Replace:
```python
    EmpiricalDataProblem('student',    'G3', drop=('G1', 'G2'),         x_transforms=_OHE, zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),              x_transforms=_OHE, zero_variance_filter=True),
```

- [ ] **Step 2: Read and update cell 4 (preview, d=2) — `student`**

Read cell id `229399e5`. Replace:
```python
    EmpiricalDataProblem('student',    'G3', drop=('G1', 'G2'),         x_transforms=_OHE + (PolynomialExpansion(2),),                         zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),              x_transforms=_OHE + (PolynomialExpansion(2),),                         zero_variance_filter=True),
```

- [ ] **Step 3: Read and update cell 7 (preview, d=3) — `student`**

Read cell id `994819b1`. Replace:
```python
    EmpiricalDataProblem('student',    'G3', drop=('G1', 'G2'),         x_transforms=_OHE + (PolynomialExpansion(3),),                         zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),              x_transforms=_OHE + (PolynomialExpansion(3),),                         zero_variance_filter=True),
```

- [ ] **Step 4: Read and update cell 13 (full, d=1) — all four datasets**

Read cell id `2bfbd407`.

Replace `student`:
```python
    EmpiricalDataProblem('student',          'G3',                  drop=('G1', 'G2'),         x_transforms=_OHE,        zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),                               x_transforms=_OHE,        zero_variance_filter=True),
```

Replace the two `naval_propulsion` entries:
```python
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=('GT_turbine_decay',),                          zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=('GT_compressor_decay',),                       zero_variance_filter=True),
```
with the single entry:
```python
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),                                zero_variance_filter=True),
```

Replace the two `parkinsons` entries:
```python
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=('total_UPDRS',),                               zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=('motor_UPDRS',),                               zero_variance_filter=True),
```
with the single entry:
```python
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),                                             zero_variance_filter=True),
```

Replace `facebook`:
```python
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',                    x_transforms=_OHE,                                   zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',),
                         nan_policy='drop_rows',                    x_transforms=_OHE,                                   zero_variance_filter=True),
```

- [ ] **Step 5: Read and update cell 15 (full, d=2) — all four datasets**

Read cell id `b12d0253`. Apply the same four substitutions as step 4, adding `x_transforms=(PolynomialExpansion(2),)` (or `_OHE + ...`) as appropriate. Exact replacements:

Replace `student`:
```python
    EmpiricalDataProblem('student',          'G3',                  drop=('G1', 'G2'),          x_transforms=_OHE + (PolynomialExpansion(2),),         zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),                                x_transforms=_OHE + (PolynomialExpansion(2),),         zero_variance_filter=True),
```

Replace the two `naval_propulsion` entries:
```python
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=('GT_turbine_decay',),   x_transforms=(PolynomialExpansion(2),),              zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=('GT_compressor_decay',), x_transforms=(PolynomialExpansion(2),),             zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),          x_transforms=(PolynomialExpansion(2),),              zero_variance_filter=True),
```

Replace the two `parkinsons` entries:
```python
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=('total_UPDRS',),        x_transforms=(PolynomialExpansion(2),),              zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=('motor_UPDRS',),        x_transforms=(PolynomialExpansion(2),),              zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),                       x_transforms=(PolynomialExpansion(2),),              zero_variance_filter=True),
```

Replace `facebook`:
```python
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',                    x_transforms=_OHE + (PolynomialExpansion(2),),                                    zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',),
                         nan_policy='drop_rows',                    x_transforms=_OHE + (PolynomialExpansion(2),),                                    zero_variance_filter=True),
```

- [ ] **Step 6: Read and update cell 18 (full, d=3) — all four datasets**

Read cell id `d245ce96`. Apply the same four substitutions with `PolynomialExpansion(3)`:

Replace `student`:
```python
    EmpiricalDataProblem('student',          'G3',                  drop=('G1', 'G2'),          x_transforms=_OHE + (PolynomialExpansion(3),),         zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),                                x_transforms=_OHE + (PolynomialExpansion(3),),         zero_variance_filter=True),
```

Replace the two `naval_propulsion` entries:
```python
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=('GT_turbine_decay',),   x_transforms=(PolynomialExpansion(3),),              zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=('GT_compressor_decay',), x_transforms=(PolynomialExpansion(3),),             zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),          x_transforms=(PolynomialExpansion(3),),              zero_variance_filter=True),
```

Replace the two `parkinsons` entries:
```python
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=('total_UPDRS',),        x_transforms=(PolynomialExpansion(3),),              zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=('motor_UPDRS',),        x_transforms=(PolynomialExpansion(3),),              zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),                       x_transforms=(PolynomialExpansion(3),),              zero_variance_filter=True),
```

Replace `facebook`:
```python
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',                    x_transforms=_OHE + (PolynomialExpansion(3),),                                    zero_variance_filter=True),
```
with:
```python
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',),
                         nan_policy='drop_rows',                    x_transforms=_OHE + (PolynomialExpansion(3),),                                    zero_variance_filter=True),
```

- [ ] **Step 7: Update result-display cells (3, 6, 9, 14, 17, 20)**

In each of cells `6b69df71`, `e8c4ddaa`, `59535500`, `bf1e6b1e`, `f54e9b3d`, `8ae40591`:

Replace:
```python
    row = {'dataset': problem.dataset, 'target': problem.target}
```
with:
```python
    row = {'dataset': problem.dataset, 'target': ', '.join(problem.target) if isinstance(problem.target, tuple) else problem.target}
```

Read each cell before editing; confirm the `row = {` line is present before applying the edit.

- [ ] **Step 8: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: update real_data.ipynb problem lists to multi-target for student/naval/parkinsons/facebook"
```

---

## Task 4: Full test suite

- [ ] **Step 1: Run bare pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && pytest
```

Expected: all tests pass, including notebook tests via nbmake. The preview experiment cells (2, 4, 7) run with `ignore_cache=True` and exercise the new multi-target student entries live.

- [ ] **Step 2: Commit if any test-driven fixes were made**

If step 1 required fixes, commit them before pushing.

- [ ] **Step 3: Push to dev**

```bash
git push
```
