# Multi-Target Empirical Experiment Support

Multi-target fitting is now available in `RidgeEM` and `RidgeLOOCV`. This spec extends the empirical experiment infrastructure to match: `EmpiricalDataProblem` gains multi-target definitions, every metric is updated to aggregate across targets and remain scalar-valued, and the `real_data` notebook demonstrates the new capability by realising multiple target prediction problems on all datasets where multiple targets are defined in their source description (or from logical inference). This includes the student dataset present in the preview experiment.

---

## `EmpiricalDataProblem` — multi-target `target` parameter

`target` is extended from `str` to `str | tuple[str, ...]`. When a tuple is supplied:

- All named columns are excluded from X.
- `get_X_y()` returns a DataFrame for Y (shape `(n, q)`) instead of a Series, aligned with existing single-target semantics (Series is the 1-column case of DataFrame). Estimators already unpack the numpy arrays via
```
    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        ...
```
- The tuple type preserves hashability, so the identity/caching key is unchanged.

### NaN handling

`get_X_y` currently has two separate NaN concerns that interact differently with multi-target:

Introduce a local variable at the top of `get_X_y` that normalises `target` to a list in both the string and tuple cases:

```python
target_cols = [self.target] if isinstance(self.target, str) else list(self.target)
```

Then use `target_cols` consistently throughout the method:

**Step 1 — target NaN filter (always applied):**
```python
# before
df = df.dropna(subset=[self.target])
# after
df = df.dropna(subset=target_cols)
```
`dropna(subset=...)` with a list drops any row where ANY listed column is NaN. For single-target this is unchanged; for multi-target it drops rows missing any target value — the correct semantics, since a row is only usable if all targets are observed.

**Step 2 — feature NaN policy (optional):**
`nan_policy='drop_rows'` and `nan_policy='drop_cols'` are unchanged. They apply after step 1 to handle NaN in feature columns. Because step 1 already cleaned the target columns, `drop_cols` will never remove a target column.

**X/y split:**
```python
# before
X = df.drop(columns=[self.target])
y = df[self.target]
# after
X = df.drop(columns=target_cols)
y = df[self.target] if isinstance(self.target, str) else df[target_cols]
```
The `y` line preserves the existing single-target return type (Series for str) while returning a DataFrame for tuple targets. Using `target_cols` for the X split works uniformly for both cases.

The `drop` parameter (for excluding additional columns from X) remains fully compatible with a tuple `target` — it applies before the target split and serves a different purpose.

### Dataset multi-target assessment

Every dataset was inspected for natural multi-target structure (multiple outcome columns that are conceptually co-predicted rather than input features). Columns in the cached CSV files were reviewed; the Twitter Buzz dataset was verified against the UCI source description.

| Dataset | Assessment |
|---|---|
| `abalone` | Single-target (`Rings`). Weight columns are physical features, not co-outcomes. |
| `airfoil` | Single-target. |
| `automobile` | Single-target (`price`). |
| `autompg` | Single-target (`mpg`). `car_name` is an identifier. |
| `blog` | Single-target (`V281`). Single buzz count per post. |
| `boston` | Single-target (`medv`). |
| `concrete` | Single-target. |
| `crime` | Single-target (`ViolentCrimesPerPop`). Confirmed as the single natural target; other crime-rate columns may be confounders rather than features and warrant separate investigation. |
| `ct_slices` | Single-target (`reference`). |
| `diabetes` | Single-target. |
| `eye` | Single-target. |
| `facebook` | **Multi-target candidate.** The 7 pre-publication columns (Page total likes, Type, Category, Post Month, Post Weekday, Post Hour, Paid) are the true features; `comment`, `like`, `share` are post-publication co-outcomes; `Total Interactions` is their sum (dropped as a derived quantity). The current single-target entry uses all 18 remaining columns — including post-publication metrics — as features, creating data leakage; the multi-target entry corrects this. All `Lifetime Post *` columns are also post-publication and should be dropped from X. |
| `forest` | Single-target (`area`). |
| `naval_propulsion` | **Multi-target:** `('GT_compressor_decay', 'GT_turbine_decay')`. |
| `parkinsons` | **Multi-target:** `('motor_UPDRS', 'total_UPDRS')`. |
| `real_estate` | Single-target. |
| `ribo` | Single-target. |
| `student` | **Multi-target:** `('G1', 'G2', 'G3')`. |
| `tomshw` | Single-target (`V97`). Buzz dataset analogous to Twitter. |
| `twitter` | Single-target (`V78`). V1–V77 are temporal feature windows for a single buzz outcome; verified against UCI source description. |
| `yacht` | Single-target. |

### Affected dataset entries

Four datasets have natural multi-target structure and are updated:

**student** — `('G1', 'G2', 'G3')`, no `drop` needed (all three excluded from X automatically).

**naval_propulsion** — `('GT_compressor_decay', 'GT_turbine_decay')`, no `drop` needed. The two existing single-target entries (each dropping the other target) become one entry.

**parkinsons** — `('motor_UPDRS', 'total_UPDRS')`, no `drop` needed. Same consolidation as naval.

**facebook** — `('comment', 'like', 'share')` with `drop` covering `Total Interactions` and all `Lifetime Post *` columns. This is the correct feature set (7 pre-publication columns only) and also resolves the pre-existing data leakage. The entry replaces the current single-target `Total Interactions` entry.

---

## Metrics — design principle

All metrics remain scalar-valued. When an estimator attribute is array-valued (multi-target fit), the metric aggregates across targets. The aggregation function is chosen to reflect the natural semantics of each metric:

- Means aggregate quantities whose scale is per-target (regularisation, variance).
- Sums aggregate quantities whose scale is per-computation (iterations).
- Some metrics are already scalar by construction regardless of the number of targets.
- Metric receives the same `y` that was passed to the experiment runner, i.e., they have to dynamically cope with different dimensionalities.

---

## Per-metric changes

### `ParameterMeanSquaredError`

Change:

```python
return ((est.coef_ - prob.beta)**2).mean()
```

to:

```python
return float(((est.coef_ - prob.beta)**2).mean())
```

`((est.coef_ - prob.beta)**2).mean()` already aggregates over all elements when both `coef_` and `prob.beta` are 2D `(q, p)`. The `float()` cast standardises the return type to a Python scalar consistent with all other modified metrics. Applicable only to synthetic experiments where `prob.beta` is defined.

Doctest to add (verifies the formula for 2D inputs):

```python
>>> class _E:
...     coef_ = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> class _P:
...     beta = np.array([[1.1, 1.9], [2.9, 4.1]])
>>> parameter_mean_squared_error(_E(), _P(), None, None)
0.01
```

### `PredictionMeanSquaredError`

Change:

```python
return ((est.predict(x) - y)**2).mean()
```

to:

```python
return float(((est.predict(x) - np.asarray(y))**2).mean())
```

`np.asarray(y)` converts a DataFrame to a plain ndarray before subtraction. Without it, `ndarray - DataFrame` returns a DataFrame and `.mean()` returns a column-wise Series instead of a scalar. The conversion is a no-op for ndarray inputs. The `float()` cast standardises the return type.

Doctest to add:

```python
>>> from sklearn.linear_model import Ridge
>>> X2 = np.eye(3)
>>> Y2 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
>>> est2 = Ridge(alpha=0.0001).fit(X2, Y2)
>>> prediction_mean_squared_error(est2, None, X2, Y2) < 1e-6
True
```

### `RegularizationParameter`

Change:

```python
return est.alpha_
```

to:

```python
return float(np.mean(est.alpha_))
```

`np.mean` on a scalar is the scalar itself, so single-target behaviour is identical. For multi-target, returns the mean regularisation parameter across targets.

Doctest to add:

```python
>>> class _E:
...     alpha_ = np.array([1.0, 3.0])
>>> regularization_parameter(_E(), None, None, None)
2.0
```

### `NumberOfIterations`

**EM path.** Change `return est.iterations_` to `return int(np.sum(est.iterations_))`. For single-target, `iterations_` is a Python int so `np.sum` is a no-op. For multi-target, sums across targets — reflecting that each target runs an independent EM loop.

**LOOCV path.** Change `return len(est.alphas_)` to `return len(est.alphas_) * (y.shape[1] if np.ndim(y) > 1 else 1)`. `np.ndim` works on any array-like or pandas object (`np.ndim(Series) == 1`, `np.ndim(DataFrame) == 2`), so the expression inlines without a helper.

**Reasoning for the LOOCV multiplier.** The current `RidgeLOOCV.fit` implementation has the following structure:

```python
# hat diagonal: one pass over the alpha grid, shared across all targets
h_per_alpha = [... for alpha in self.alphas_]

# LOO MSE: nested loop — one inner pass per (target, alpha) pair
for t in range(q):
    for i, alpha in enumerate(self.alphas_):
        beta_t = c_t / (s ** 2 + alpha)
        err = y_norm[:, t] - r.dot(beta_t)
        loo_mse_mat[t, i] = ...
```

The hat-diagonal precomputation is shared and runs `n_alphas` times total regardless of `q`. The LOO MSE loop runs `q * n_alphas` times. No intermediate results are reused across targets within the LOO MSE loop (each target needs its own `beta_t` and residual). The total number of LOO evaluations is therefore `q * n_alphas`, making `n_alphas * q` the correct count.

Doctests to add:

```python
>>> class _EM:
...     iterations_ = np.array([10, 8])
>>> y_2t = np.zeros((100, 2))
>>> number_of_iterations(_EM(), None, None, y_2t)
18
>>> class _LOOCV:
...     alphas_ = np.arange(10)
>>> number_of_iterations(_LOOCV(), None, None, y_2t)
20
```

### `VarianceAbsoluteError`

Change:

```python
return abs(prob.sigma**2 - est.sigma_square_)
```

to:

```python
return float(np.mean(np.abs(prob.sigma**2 - est.sigma_square_)))
```

For single-target, `sigma_square_` is a numpy scalar and the result equals `abs(prob.sigma**2 - est.sigma_square_)` — behaviour-preserving up to type (Python `float` vs numpy scalar). For multi-target, averages per-target absolute errors. `prob.sigma` is assumed scalar (one noise level for all targets in current synthetic problems).

All metrics that are modified in this spec return Python scalars (`float` or `int`): `RegularizationParameter` returns `float(np.mean(...))`, `NumberOfIterations` returns `int(np.sum(...))`, and `VarianceAbsoluteError` returns `float(np.mean(np.abs(...)))`. This is consistent with `FittingTime` (returns a Python `float` from `time.time()`) and avoids numpy scalar repr differences across numpy versions.

Doctest to add:

```python
>>> class _E:
...     sigma_square_ = np.array([0.25, 0.36])
>>> class _P:
...     sigma = 0.5
>>> variance_abs_error(_E(), _P(), None, None)
0.055
```

### `FittingTime`

No code change. `est.fitting_time_` is the wall-clock time for a single `fit()` call, always a scalar regardless of the number of targets. No aggregation needed.

No doctest: a meaningful test requires a real experiment run with wall-clock timing, which is non-deterministic and not suited to a doctest. Out of scope.

### `PredictionR2`

Change:

```python
return r2_score(y, est.predict(x))
```

to:

```python
return float(r2_score(y, est.predict(x)))
```

`r2_score` with 2D outputs uses `multioutput='uniform_average'` by default, returning a scalar mean R^2 across targets. For 1D y the return is also a scalar — behaviour is unchanged. The `float()` cast standardises the return type.

Doctest to add:

```python
>>> from sklearn.linear_model import Ridge
>>> X3 = np.eye(3)
>>> Y3 = np.column_stack([np.array([1., 2., 3.]), np.array([3., 2., 1.])])
>>> est3 = Ridge(alpha=0.0001).fit(X3, Y3)
>>> prediction_r2(est3, None, X3, Y3) > 0.99
True
```

### `NumberOfFeatures`

Bug fix: `len(est.coef_)` returns `q` (number of targets) for 2D `coef_` instead of `p` (number of features). Fix:

```python
return est.coef_.shape[-1]
```

`shape[-1]` is `p` for both `(p,)` and `(q, p)` shaped arrays.

Doctest to update (existing test still passes; add multi-target case):

```python
>>> from sklearn.linear_model import Ridge
>>> X4 = np.arange(20).reshape(10, 2).astype(float)
>>> Y4 = np.column_stack([X4[:, 0], X4[:, 1]])
>>> est4 = Ridge(alpha=0.0001).fit(X4, Y4)
>>> number_of_features(est4, None, None, None)
2
```

---

## `__str__` linter fix

All metrics currently define `__str__` as `@staticmethod`, which produces a linter warning since `__str__` must take `self`. Change each to an instance method:

```python
def __str__(self):
    return 'lambda'
```

No behavioural change; the returned string is identical. The module-level singletons are unaffected since they have no instance attributes that would change the string.

---

## Notebook (`real_data.ipynb`)

Every occurrence of the four affected datasets in the problem lists is updated. The notebook has four experiment sections (preview / d=1 / d=2 / d=3); all four are updated with the same replacements:

- `student`: `'G3'` with `drop=('G1', 'G2')` → `target=('G1', 'G2', 'G3')`, no `drop`
- `naval_propulsion`: two single-target entries → one entry with `target=('GT_compressor_decay', 'GT_turbine_decay')`, no `drop`
- `parkinsons`: two single-target entries → one entry with `target=('motor_UPDRS', 'total_UPDRS')`, no `drop`
- `facebook`: `'Total Interactions'` → `target=('comment', 'like', 'share')` with `drop=('Total Interactions',)`. The `Lifetime Post *` columns are retained as features — they are actionable metrics visible to the page manager at posting time and keep the problem in the large-p regime of the original setup.

The `row` dict in the result display cells currently reads `problem.target`; for a tuple target this produces a tuple string. Update to `', '.join(problem.target) if isinstance(problem.target, tuple) else problem.target` for readable display.

Cached results for the old single-target entries remain on disk but are no longer referenced. New multi-target results are computed and cached on first run. Cells tagged `skip-execution` retain their stored output in the notebook source file; nbmake does not write results back to the file, so those cells' displayed output only updates when the notebook is saved interactively.

---

## Future work

**`NumberOfIterations` brittleness.** The current implementation dispatches on three attribute names (`iterations_` for `RidgeEM`, `alphas_` for `RidgeLOOCV`, `alphas` for sklearn's `RidgeCV`). The semantics are consistent — all three return the number of alpha evaluations or EM steps — but the mechanism is fragile: any estimator that does not happen to use one of these exact names is silently unhandled. A principled replacement would be a general-purpose metric that explicitly names the field it reads (or a protocol/mixin that all compatible estimators implement), making the contract visible at experiment-definition time rather than buried in attribute-sniffing logic.

**Numerically encoded categorical variables.** Several datasets contain columns that are semantically categorical but stored as integers (e.g. `facebook`: post Type, Category; `crime`: state). `EmpiricalDataProblem` currently has no mechanism to automatically detect and one-hot-encode such columns — `x_transforms` with `OneHotEncodeCategories` requires explicit column selection at problem-definition time. A future `categorical_cols` parameter (or auto-detection from dtype/cardinality) would allow uniform treatment across datasets. This issue affects a number of datasets in the registry beyond those updated in this spec. As the number of datasets requiring non-numeric OHE grows, the module-level `_OHE` shorthand should be promoted to a public constant `ONEHOT_NON_NUMERIC = (OneHotEncodeCategories(),)` in `problems.py` and imported by notebooks rather than redefined locally.
