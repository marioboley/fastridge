# EmpiricalDataExperiment Design

## Goal

Introduce `EmpiricalDataExperiment` as a peer class to `Experiment` in `experiments/experiments.py`. It runs repeated train/test experiments on a list of `EmpiricalDataProblem` instances, stores per-run results as numpy arrays (one array per metric), and handles individual failed runs gracefully by recording NaN rather than crashing.

This design also adds two new metric classes (`PredictionR2`, `NumberOfFeatures`) and removes the `run_real_data_experiments` function, which is replaced by the new class.

---

## Background: `Experiment` Behaviour

`EmpiricalDataExperiment` must mirror `Experiment` closely so result-processing code can be eventually unified. The key behaviours to replicate:

**Constructor:**
- Accepts `problems` (list), `estimators` (list), `ns` (int or array), `reps` (int), `est_names`, `stats`, `keep_fits`, `verbose`, `seed`.
- `est_names` defaults to `[str(est) for est in estimators]`.
- `ns` is broadcast to shape `(n_problems, n_ns)` via `np.atleast_2d`.
- `self.rng = np.random.default_rng(seed)`.

**`run()` method:**
- Iterates over `reps`, then `problems`, then `ns[i]`, then `estimators`.
- For each (rep, problem, n, estimator): samples train data via `prob.rvs(n, rng=self.rng)` and a fixed 10 000-point test set via `prob.rvs(self.test_size, rng=self.rng)`.
- Clones each estimator fresh per iteration: `_est = clone(est, safe=False)`.
- Attaches `_est.fitting_time_ = time.time() - fit_start_time` after `fit`.
- Calls each stat as `stat(_est, self.problems[i], x_test, y_test)` and writes the scalar result into a pre-allocated numpy array.
- Result arrays stored as `self.__dict__[str(stat) + '_']` with shape `(reps, n_problems, n_ns, n_estimators)`.
- Optionally stores fitted estimators in `self.fits[(r, i, n, j)]` if `keep_fits=True`.

**Metric protocol:**
- Each metric is a callable: `(est, prob, x, y)` where `x, y` are the test set.
- Metrics return NaN if the required estimator attribute is absent (e.g. `VarianceAbsoluteError` returns NaN when `est.sigma_square_` is missing).
- Result arrays are pre-allocated with `np.zeros`; NaN is written only on explicit failure or metric inapplicability.

---

## New Class: `EmpiricalDataExperiment`

### Constructor

```
EmpiricalDataExperiment(problems, estimators, n_iterations, test_prop=0.3,
                        seed=None, polynomial=None, stats=empirical_default_stats,
                        est_names=None, verbose=True)
```

- `problems`: list of `EmpiricalDataProblem`.
- `estimators`: list of estimator objects.
- `n_iterations`: number of repeated train/test splits per problem.
- `test_prop`: fraction of data held out for testing (default 0.3).
- `polynomial`: if not None, applies `PolynomialFeatures(degree=polynomial)` after OHE.
- `stats`: list of metric callables following the `(est, prob, x, y)` protocol.
- `est_names`: defaults to `[str(est) for est in estimators]`.
- `verbose`: prints progress to stdout (dataset name and dots per iteration).
- `self.rng = np.random.default_rng(seed)`.

### `run()` Method

**Preprocessing (per problem, once before iterations):**
1. Call `problem.get_X_y()` to obtain `X` (DataFrame), `y` (Series).
2. Detect categorical columns via `pd.api.types.is_numeric_dtype`; apply `OneHotEncoder(drop='first', sparse_output=False)`.
3. If `polynomial` is not None, apply `PolynomialFeatures(degree=polynomial, include_bias=False)`. If the resulting matrix exceeds 35 000 000 elements, apply the existing size-cap logic (drop linear terms, randomly subsample interaction columns).
4. Derive train size: `n_train = int(X.shape[0] * (1 - test_prop))`. Store in `self.ns` as shape `(n_problems, 1)` — one entry per problem, matching the `n_ns=1` structure.

**Per-iteration loop:**
5. `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=self.rng.integers(2**31))`.
6. Drop zero-variance columns: compute `std = X_train.std(); non_zero = std[std != 0].index`; apply to both train and test.
7. For each estimator:
   a. Clone: `_est = clone(est, safe=False)`.
   b. `t0 = time.time(); _est.fit(X_train, y_train); _est.fitting_time_ = time.time() - t0` — wrapped in `try/except Exception`.
   c. On exception: write NaN to all stat arrays at `(iteration, problem_idx, 0, estimator_idx)`. Continue to next estimator.
   d. On success: call `stat(_est, problem, X_test, y_test)` for each stat and write the result.

**Result storage:**
- Pre-allocate `self.__dict__[str(stat) + '_'] = np.full((n_iterations, n_problems, 1, n_estimators), np.nan)` for each stat before the loop (note: `Experiment` uses `np.zeros`; `EmpiricalDataExperiment` uses `np.nan` so that unrun cells are distinguishable from zero results).
- Shape `(n_iterations, n_problems, n_ns, n_estimators)` with `n_ns=1`.
- NaN is also written whenever a metric returns NaN for a given estimator (already the metric protocol's convention for inapplicable metrics).
- Returns `self`.

### Verbose output

Mirrors current `run_real_data_experiments`: prints `problem.dataset (n=..., p=...)` then one dot per iteration, then newline.

---

## New Metric Classes

### `PredictionR2`

```python
class PredictionR2:
    @staticmethod
    def __call__(est, prob, x, y):
        return r2_score(y, est.predict(x))

    @staticmethod
    def __str__():
        return 'prediction_r2'

    @staticmethod
    def symbol():
        return r'$R^2$'
```

### `NumberOfFeatures`

```python
class NumberOfFeatures:
    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'coef_'):
            return len(est.coef_)
        return float('nan')

    @staticmethod
    def __str__():
        return 'number_of_features'

    @staticmethod
    def symbol():
        return r'$p$'
```

**Module-level singletons** (matching existing pattern):
```python
prediction_r2 = PredictionR2()
number_of_features = NumberOfFeatures()
```

**Default stats for empirical experiments:**
```python
empirical_default_stats = [
    prediction_mean_squared_error,
    prediction_r2,
    regularization_parameter,
    number_of_iterations,
    fitting_time,
    number_of_features,
]
```

---

## Metric Mapping: `run_real_data_experiments` vs New Framework

| Key in current result dict | How computed currently | Metric class in new framework |
|---|---|---|
| `mse` | `mean_squared_error(y_test, y_pred)` | `PredictionMeanSquaredError` (exists) |
| `r2` | `r2_score(y_test, y_pred)` | `PredictionR2` (new) |
| `time` | `time.time() - t0` per estimator | `FittingTime` (exists); runner attaches `fitting_time_` to clone |
| `lambda` | `estimator.alpha_` | `RegularizationParameter` (exists); returns NaN if absent |
| `iter` | `estimator.iterations_` | `NumberOfIterations` (exists); returns NaN if absent |
| `p` | `len(estimator.coef_)` | `NumberOfFeatures` (new) |
| `n_train` | `int(X_train.shape[0])` | Stored in `self.ns`, not a per-run stat |
| `CA`, `q` | classification accuracy, n classes | Out of scope |

---

## Removed

- `run_real_data_experiments` function: deleted from `experiments.py`.

---

## Notebook Updates

`real_data.ipynb` and `real_data_neurips2023.ipynb` currently call `run_real_data_experiments(...)` and work with the list-of-aggregated-dicts result. Both are updated to use `EmpiricalDataExperiment(...).run()`.

Result access in notebooks changes from `results[i]['EM']['r2']` (scalar mean) to `np.nanmean(exp.prediction_r2_[:, i, 0, j])` (mean over non-NaN iterations). Helper expressions for this are shown in the implementation plan.

---

## Files

- Modify: `experiments/experiments.py`
  - Add `PredictionR2`, `NumberOfFeatures` metric classes and singletons.
  - Add `empirical_default_stats` list.
  - Add `EmpiricalDataExperiment` class.
  - Remove `run_real_data_experiments`.
- Modify: `experiments/real_data.ipynb` — replace `run_real_data_experiments` calls.
- Modify: `experiments/real_data_neurips2023.ipynb` — replace `run_real_data_experiments` calls.
