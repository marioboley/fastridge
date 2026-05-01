# Timing Analysis

## Motivation

The speed-up ratio of EM over LOOCV did not improve as expected after adding multi-target
support and numpy array unpacking (to reduce normalisation time). To investigate, we need
to track normalization time separately from SVD time and produce tables and charts that
decompose total fitting time into its components.

## Changes to `fastridge.py`

Both `RidgeEM.fit` and `RidgeLOOCV.fit` gain a `normalization_time_` attribute, timed
the same way as `svd_time_`: bracket the normalization block (lines from
`a_x = x.mean(...)` through `y = (y - a_y) / b_y`) with `time.time()` calls.

## Changes to `experiments/experiments.py`

Add `SvdTime` and `NormalizationTime` metric classes. Both subclass `FittingTime`
(inheriting its warn_recompute and warn_retrieval timing logic) and override `__call__`,
`__str__`, and `symbol`. Add module-level instances `svd_time` and `normalization_time`.
These are NOT added to `default_stats` — they must be explicitly requested per experiment.

```python
class SvdTime(FittingTime):
    @staticmethod
    def __call__(est, prob, x, y):
        return est.svd_time_

    def __str__(self):
        return 'svd_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{svd}$ [s]'


class NormalizationTime(FittingTime):
    @staticmethod
    def __call__(est, prob, x, y):
        return est.normalization_time_

    def __str__(self):
        return 'normalization_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{norm}$ [s]'


svd_time = SvdTime()
normalization_time = NormalizationTime()
```

## New module `experiments/journal2026.py`

Extracts the problem and train-size definitions that are currently inline in `real_data.ipynb`
into a module, so they become the single source of truth and can be imported by the
timing notebook without duplication.

Structure mirrors `neurips2023.py`:

```python
JOURNAL2026_TRAIN_SIZES = NEURIPS2023_TRAIN_SIZES  # already covers all needed datasets
```

Problem collections — define the full sets first, then named subsets:

```python
JOURNAL2026_D1 = [...]          # problems_full + problems_large (all 21 d=1 datasets)
JOURNAL2026_D2 = [...]          # problems_full_d2 + problems_large_d2 (ribo excluded)
JOURNAL2026_D3 = [...]          # problems_full_d3 (no large, eye/ribo excluded)

JOURNAL2026_D1_PREVIEW  = [...]  # 9-dataset preview subset (from "Preview Experiment" cell)
JOURNAL2026_D1_REGULAR  = [...]  # problems_full (17 datasets — "small/moderate" figure group)
JOURNAL2026_D1_LARGE    = [...]  # problems_large (4 datasets — "large" figure group)
JOURNAL2026_D2_PREVIEW  = [...]  # d=2 versions of the 9 preview datasets
JOURNAL2026_D2_REGULAR  = [...]  # problems_full_d2 (16 datasets)
JOURNAL2026_D2_LARGE    = [...]  # problems_large_d2 (4 datasets)
JOURNAL2026_D3_PREVIEW  = [...]  # d=3 versions of the 9 preview datasets (minus exclusions)
JOURNAL2026_D3_REGULAR  = [...]  # problems_full_d3 (15 datasets)
```

The regular/large split exactly matches the existing figure split in `real_data.ipynb`:
`col_exps = [[exp_full, exp_large], [exp_full_d2, exp_large_d2], [exp_full_d3]]`.

Also add timing estimator constants:
```python
TIMING_ESTIMATORS = [RidgeEM(), RidgeLOOCV(alphas=101), RidgeLOOCV(alphas=11)]
TIMING_EST_NAMES  = ['EM', 'CV_glm_101', 'CV_glm_11']
```

## New notebook `experiments/timing_analysis.ipynb`

Runs `Experiment` on `JOURNAL2026_D1_REGULAR` and `JOURNAL2026_D1_LARGE` separately
with `TIMING_ESTIMATORS` and
`stats=[fitting_time, svd_time, normalization_time, number_of_features]`.
Use 30 reps and `seed=123`. Tag experiment cells `skip-execution` for CI.

### Sanity check cell

For each dataset, compute the mean and std of `normalization_time_` and `svd_time_`
across the three estimators. Print a warning for any dataset where the std across
estimators exceeds 10% of the mean (would indicate unexpected variance in preprocessing
that should be constant across estimators sharing the same X).

### Tables

One table for regular datasets, one for large. Columns:

| Column    | Definition |
|-----------|------------|
| `T_EM`    | mean fitting_time_ for EM |
| `T_CV101` | mean fitting_time_ for CV_glm_101 |
| `T_CV11`  | mean fitting_time_ for CV_glm_11 |
| `T_svd`   | mean svd_time_ averaged across the three estimators |
| `T_norm`  | mean normalization_time_ averaged across the three estimators |
| `T_prep`  | T_svd + T_norm |
| `SU`      | T_CV101 / T_EM |
| `SU_post` | (T_CV101 - T_prep) / (T_EM - T_prep) |

### Stacked bar chart

Two panels: regular datasets and large datasets (separate y-scales).
Each panel shows one grouped bar per dataset, one bar per estimator.
Each bar is stacked: T_norm (bottom), T_svd (middle), T_fit (top), where
T_fit = fitting_time_ - svd_time_ - normalization_time_.
Error bars on total height use 2.5th/97.5th percentiles across reps
(right-skewed timing distributions, so percentile-based CI rather than normal-theory).

## Refactor `experiments/real_data.ipynb`

Replace all inline problem and estimator definitions with imports from `journal2026`:

```python
from journal2026 import (
    JOURNAL2026_TRAIN_SIZES,
    JOURNAL2026_D1_PREVIEW, JOURNAL2026_D2_PREVIEW, JOURNAL2026_D3_PREVIEW,
    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE,
    JOURNAL2026_D2_REGULAR, JOURNAL2026_D2_LARGE,
    JOURNAL2026_D3_REGULAR,
    JOURNAL2026_ESTIMATORS, JOURNAL2026_EST_NAMES,
)
```

The experiment cells themselves (Experiment construction, run, and result tables) remain
in the notebook — only the problem/estimator definitions move to the module.

This also means `JOURNAL2026_ESTIMATORS` and `JOURNAL2026_EST_NAMES` should be added to
`journal2026.py` (same estimators as currently used in `real_data.ipynb`:
`[RidgeEM(), RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)), RidgeLOOCV(alphas=100)]`
with names `['EM', 'CV_fix', 'CV_glm']`).

## Scope

- `fastridge.py`: add `normalization_time_` to `RidgeEM.fit` and `RidgeLOOCV.fit`
- `experiments/experiments.py`: add `SvdTime`, `NormalizationTime`, `svd_time`, `normalization_time`
- `experiments/journal2026.py`: new module with all problem collections, train sizes, estimators, and timing estimators
- `experiments/real_data.ipynb`: refactor to import problem/estimator definitions from `journal2026`
- `experiments/timing_analysis.ipynb`: new notebook, not in CI
