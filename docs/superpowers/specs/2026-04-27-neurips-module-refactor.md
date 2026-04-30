# NeurIPS Module Refactor Design

The `neurips2023.py` module currently holds only `SyntheticDataExperiment`, while the NeurIPS-specific
problem collections (`NEURIPS2023`, `NEURIPS2023_D2`, `NEURIPS2023_D3`, `NEURIPS2023_TRAIN_SIZES`),
estimator list, and the `ExperimentWithPerSeriesSeeding` class all live in `problems.py` or
`experiments.py` — far from the NeurIPS-specific context they belong to. This makes `neurips2023.py`
not self-contained and forces notebooks to import paper-specific artefacts from general-purpose modules.

A secondary motivation: recent speed-ups to `RidgeEM` and `RidgeLOOCV` (`np.asarray` in `fit()`)
have changed fitting time measurements and may render comparisons to old cached computations invalid.
This creates a need for a targeted cache-invalidation mechanism that goes beyond `force_recompute`
(which accumulates multiple computation records) and beyond `ignore_cache` (which bypasses the cache
entirely). A new `overwrite_cache` option — deleting stale entries and writing a clean replacement —
fills this gap cleanly.

The goal of this refactor is to make `neurips2023.py` the single home for all NeurIPS 2023 experiment
specifics, including the ability to re-run all experiments from the command line without opening a
notebook, and to add principled cache management to the experiment infrastructure.

## Changes

### 1. Rename `_OHE` to `onehot_non_numeric` in `problems.py`

`_OHE` is currently a private module-level constant with no docstring. Making it public with a
descriptive snake_case name (`onehot_non_numeric`) allows it to be imported and reused without
redefinition. snake_case is consistent with metric singletons (`prediction_r2`, `fitting_time`)
and will look natural alongside a future `onehot_columns([...])` factory in combined transform
tuples: `(onehot_non_numeric, onehot_columns(['state']))`.

**New definition (line 380):**
```python
onehot_non_numeric = (OneHotEncodeCategories(),)
```

`real_data.ipynb` currently defines `_OHE = (OneHotEncodeCategories(),)` locally — a single-source-of-
truth violation. That local definition must be removed and all occurrences of `_OHE` in the notebook
replaced with an import of `onehot_non_numeric` from `problems`.

### 2. Move NeurIPS 2023 constants from `problems.py` to `neurips2023.py`

Remove `NEURIPS2023`, `NEURIPS2023_D2`, `NEURIPS2023_D3`, and `NEURIPS2023_TRAIN_SIZES` from
`problems.py` (lines 382–497) and define them in `neurips2023.py`, using `onehot_non_numeric` from
`problems`.

Required imports added to `neurips2023.py`:
```python
import dataclasses
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from problems import (EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric)
from data import DATASETS
```

The constant definitions are reproduced verbatim from `problems.py` with `_OHE` replaced by
`onehot_non_numeric`.

### 3. Add estimator constants to `neurips2023.py`

Define the NeurIPS 2023 estimators as module-level constants — parallel to the problem set constants
and satisfying the single-source-of-truth principle:

```python
NEURIPS2023_ESTIMATORS = [
    RidgeEM(t2=False),
    RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
    RidgeLOOCV(alphas=100),
]

NEURIPS2023_EST_NAMES = ['EM', 'CV_fix', 'CV_glm']
```

These match the estimator list currently defined inline in `real_data_neurips2023.ipynb`
(cell comment: `# estimator indices: 0=EM, 1=CV_fix, 2=CV_glm`).

### 4. Move `ExperimentWithPerSeriesSeeding` from `experiments.py` to `neurips2023.py`

`ExperimentWithPerSeriesSeeding` is a legacy class created specifically to reproduce NeurIPS 2023
results with per-series MT19937 seeding; it is not a general-purpose experiment runner. It belongs
in `neurips2023.py` alongside the problem sets and estimators it was designed to run.

**Design consideration:** the class uses several private names from `experiments.py`
(`_cache_key`, `_make_run_id`, `_RUN_FILE_STATE`, `empirical_default_stats`). Moving the class
requires either (a) promoting these to public exports of `experiments.py`, or (b) keeping them
private and importing them by name. Option (a) is preferred — if they are needed outside
`experiments.py`, they should be public.

**Circular import:** the current `neurips2023.py` already imports `from experiments import
default_stats`, and `experiments.py` does not import from `neurips2023.py`. There is therefore no
circular import risk, and no conditional or deferred import is needed.

**`EmpiricalDataExperiment` deletion:** `EmpiricalDataExperiment` is currently used in
`tests/test_experiments.py` both as the primary fixture and as the reference implementation in the
`ExperimentWithPerSeriesSeeding` equivalence test (line 253). Deletion requires updating those tests:
the fixture must be rewritten to use `ExperimentWithPerSeriesSeeding` directly, and the equivalence
test replaced with a standalone reproduction check. This is an explicit step in the plan.

### 5. Add `overwrite_cache` option to `Experiment` and `ExperimentWithPerSeriesSeeding`

Three cache-control options with a clear precedence hierarchy:

| Option | Reads cache | Writes cache | Deletes stale entries |
|---|---|---|---|
| default | yes | yes | no |
| `force_recompute=True` | no | yes (appends) | no |
| `overwrite_cache=True` | no | yes (clean) | yes, per combination |
| `ignore_cache=True` | no | no | no |

`overwrite_cache` implies recomputation (stronger than `force_recompute`) but still writes to
cache (weaker than `ignore_cache`, which leaves the cache unchanged). When `ignore_cache=True`,
it takes precedence over both other flags regardless of their values.

**Deletion timing:** deletion happens per (problem, n_train, estimator) combination immediately
before recomputing it, not upfront for the whole experiment. An interrupted run therefore leaves
already-processed combinations with fresh cached data rather than deleting everything first.

**Signature change:**
```python
def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
```

### 6. Add `__main__` runner with CLI options to `neurips2023.py`

The runner re-runs all three experiment tiers. Cache-control options are exposed as command-line
flags matching the `run()` API:

```python
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    args = parser.parse_args()

    for problems in [NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3]:
        problems_sorted = sorted(problems, key=lambda p: DATASETS[p.dataset]['n'])
        ExperimentWithPerSeriesSeeding(
            problems=problems_sorted,
            estimators=NEURIPS2023_ESTIMATORS,
            reps=100,
            ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_sorted],
            seed=123,
            est_names=NEURIPS2023_EST_NAMES,
        ).run(
            force_recompute=args.force_recompute,
            ignore_cache=args.ignore_cache,
            overwrite_cache=args.overwrite_cache,
        )
```

Since `ExperimentWithPerSeriesSeeding` will move into `neurips2023.py` (change 4), no import is
needed inside `__main__`.

### 7. Update notebook imports

**`real_data_neurips2023.ipynb`:**
- Split cell `a003` into two cells:
  - New cell inserted between the intro markdown (`a001`) and the "No Interaction Variables"
    markdown (`a002`): imports `NEURIPS2023_ESTIMATORS` and `NEURIPS2023_EST_NAMES` from
    `neurips2023`, with `NEURIPS2023_ESTIMATORS` as the final expression so it renders as output.
    This cell is the single place in the notebook where estimators are defined/imported.
  - Remaining content of `a003` (problem imports, `problems_d1`, `exp_d1` construction and run):
    remove the inline `estimators` / `est_names` definitions, which are now in the new cell above.
- Import `NEURIPS2023, NEURIPS2023_TRAIN_SIZES` from `neurips2023` (not `problems`)
- Import `NEURIPS2023_D2` from `neurips2023`
- Import `NEURIPS2023_D3` from `neurips2023`

**`real_data.ipynb`:**
- Import `NEURIPS2023_TRAIN_SIZES` from `neurips2023` (not `problems`)
- Remove the local `_OHE` definition; import `onehot_non_numeric` from `problems`
- Replace all `_OHE` references with `onehot_non_numeric`

## Out of Scope

- `journal2026.py` and its notebook — deferred to a later iteration.
- Changes to `SyntheticDataExperiment` (already uses `progress_bar`).
- Any change to cache format or serialization beyond the `overwrite_cache` deletion semantics.

## Future Work

### Progress bar in experiment runners

`Experiment` and `ExperimentWithPerSeriesSeeding` currently use `print('.')` per trial as their
verbose feedback mechanism. `SyntheticDataExperiment` already uses `fastprogress.progress_bar`
(one bar over `reps`). The natural target is one bar per problem spanning all
`(n_train, estimator, rep)` trials, matching the style of `SyntheticDataExperiment`.

This is deferred because the key design question — whether and how to support a non-verbose code
path — has not been resolved. Introducing a separate branch for verbose vs. non-verbose is a
non-trivial entropy increase (every branch is a maintenance surface) and must be discussed
explicitly before implementation rather than written in silently.
