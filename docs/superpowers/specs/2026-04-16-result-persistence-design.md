# Result Persistence Design

## Prerequisite

This spec requires the [preprocessing pipeline design](2026-04-17-preprocessing-pipeline-design.md),
the [seeding refactor design](2026-04-18-empirical-experiment-seeding-refactor-design.md),
and the [parameter-based identity design](2026-04-21-parameter-based-identity-design.md)
to be implemented first. The seeding refactor establishes `n_train`, `generator`,
`seed_scope`, and `seed_progression` as first-class parameters on
`EmpiricalDataExperiment`. The parameter-based identity design establishes
`joblib.hash()` as the cache key mechanism used here.
No changes to `problems.py` are made by this spec.

---

## Goal

Add result caching to `EmpiricalDataExperiment` so that re-running with a modified
estimator list only recomputes affected units, reusing stored results for everything
else. Eliminates the need to rerun full experiments (currently >1 h) when iterating
on a subset of estimators.

---

## Terminology

**Trial** — the atomic unit of work: one (problem, estimator, n, seed) combination
producing one scalar per metric.

**Series** — all reps for one (problem, estimator, n, base\_seed, reps)
combination, stored as a 1-D array of length `reps` per metric.

---

## Seeding and Cache Paths

Seeding parameters (`generator`, `seed_scope`, `seed_progression`) are defined in
the [seeding refactor design](2026-04-18-empirical-experiment-seeding-refactor-design.md).
This section covers only how those parameters determine cache directory paths.

The top-level folder is determined by `seed_scope`: `'series'` -> `series/`,
`'trial'` -> `trial/`. A `series/` entry is fully keyed by
`(problem, estimator, n_train, reps, generator, seed)` — `seed_progression` plays
no role in the path. A `trial/` entry uses `seed + rep_idx` as the trial seed:

| `seed_scope` | Cache path | Cache granularity |
|---|---|---|
| `'series'` | `series/.../` | all reps together per estimator |
| `'trial'` | `trial/.../` | one rep at a time per estimator |
| `'experiment'` | `experiment/` | whole result array (future, see Future) |

`generator` appears as a `<generator>/` directory level — the same seed with a
different generator produces different splits and must be stored separately.

---

## Caching

### Global cache directory

`CACHE_DIR` is a module-level constant in `experiments.py`, defaulting to the
`results/` directory at the repository root. Every `run()` call reads from and
writes to `CACHE_DIR` unless `ignore_cache=True`.

### `force_recompute` and `ignore_cache` parameters on `run()`

`run(force_recompute=False)` — when `True`, recomputes all trials/series and
appends results to the existing cache files rather than reading from them. Useful
when an estimator implementation has changed without its `get_params()` output
changing. The full computation history is retained.

`run(ignore_cache=False)` — when `True`, disables both reads and writes entirely.
The run executes as if no cache exists and produces no cached output. Intended for
unit tests and preview cells where cache pollution would be undesirable.

### Behaviour during `run()`

For each (problem, estimator) pair:

1. Derive the cache key (see below).
2. If `ignore_cache=True`: fit, store result in `self.result_`, skip all file I/O.
3. Check whether the corresponding metric file exists under `CACHE_DIR`.
4. **Hit** (and `force_recompute=False`): load stored result; place into
   `self.result_`. Skip fitting. Record retrieval in the metric file.
5. **Miss** (or `force_recompute=True`): fit the estimator, record results,
   append to metric file.

---

## File Structure

```
results/
  series/
    <problem_key>/
      <n_train>/
        <estimator_key>/
          <reps>/
            <generator>/
              <base_seed>/
                <metric_name>.json  <- one file per metric
                predictions/        <- (future) one .npy per rep
                estimators/         <- (future) one .pkl per rep
  trial/
    <problem_key>/
      <n_train>/
        <estimator_key>/
          <generator>/
            <trial_seed>/
              <metric_name>.json    <- one file per metric
              predictions.npy       <- (future) y_pred array for this trial
              estimator.pkl         <- (future) fitted estimator for this trial
  runs/
    <run_id>.json                   <- one file per experiment run
```

The top-level subfolder is determined by `seed_scope` alone: `'series'` ->
`series/`, `'trial'` -> `trial/`. `<n_train>/` captures the actual training
set size — two experiments on the same problem with different `n_train` values
produce separate paths. `<reps>/` means series of different lengths coexist without
conflict. `<generator>/` separates results from different BitGenerators since the
same seed produces different numerical outcomes across generators. Using a directory
per trial/series allows future artifact types to be added without changing the key
structure. One JSON file per metric (rather than one combined archive) means adding
or removing a metric from `stats` only affects that metric's file; all others are
unmodified.

### Storage format: per-metric JSON

Each `<metric_name>.json` file stores the full computation and retrieval history for
that metric at a single (problem, estimator, n_train, ...) path.

**`trial/` example** (scalar value per computation):

```json
{
  "computations": [
    {"value": 0.851, "run_id": "20260421-143045-a3f7"},
    {"value": 0.849, "run_id": "20260423-091200-b2c1"}
  ],
  "retrievals": [
    {"value": 0.850, "run_id": "20260422-110305-c9d0"}
  ]
}
```

**`series/` example** (array of length `reps` per computation):

```json
{
  "computations": [
    {"value": [0.851, 0.843, 0.867, 0.859, 0.872], "run_id": "20260421-143045-a3f7"}
  ],
  "retrievals": [
    {"value": [0.851, 0.843, 0.867, 0.859, 0.872], "run_id": "20260422-110305-c9d0"}
  ]
}
```

- `computations`: ordered list of all past computed results, each with the value and
  the run that produced it. New entries are appended; existing entries are never
  modified.
- `retrievals`: list of cache-hit records. Each entry stores the run ID and the mean
  value that was returned to that run (which may differ across retrievals as more
  computations are appended). Added on cache hit.
- The value used when populating `self.result_` is the **mean across all
  computations** — a scalar for `trial/`, a per-rep mean array for `series/`.

Files are written atomically (write to a temp path, then `os.replace`) to avoid
corruption on interrupted runs. JSON requires no extra dependency and is human-
readable; the storage overhead for typical metric arrays is negligible relative to
fitting time.

---

## Metric Warning Behavior

When a result is recomputed or retrieved from cache, the runner checks it against
stored history and issues warnings if the value is unexpectedly inconsistent. Each
stat class defines its own tolerance for what counts as inconsistent. All existing
stats are already class instances. This design introduces a `Metric`
base class in `experiments.py` that all stat classes will inherit from. The runner
parameter stays named `stats`. Each `Metric` subclass may override:

```python
def warn_recompute(self, existing, new_value):
    """Return a warning message if new_value is surprising given existing
    computations, or None. Called only when len(existing) >= 1."""

def warn_retrieval(self, computations):
    """Return a warning message about the reliability of the cached mean,
    or None. Called on every cache hit regardless of len(computations)."""
```

`existing` and `computations` are lists of scalar values (for `trial/`) or lists of
arrays of length `reps` (for `series/`). For `series/`, both methods apply checks
element-wise and warn if any rep triggers the condition.

### Default behavior (`Metric` base class)

`warn_recompute` warns if the new value is not close to the mean of existing values
(`np.allclose` with default tolerances). `warn_retrieval` warns if stored computation
values are not all close to their mean. This avoids false positives from negligible
floating-point differences due to minor library changes, while catching real
discrepancies. Both are element-wise for `series/` values.

This is the right default for deterministic stats like `PredictionR2` — any
discrepancy indicates non-determinism.

### `FittingTime`

Fitting time is inherently variable; the default exact-equality checks would fire
constantly. `FittingTime` overrides both methods with CI-based checks:

- `warn_recompute`: warn if the new value falls outside the 95% CI of existing
  values (mean +/- 1.96 * SE, SE = std / sqrt(n)).
- `warn_retrieval`:
  - If only one computation is stored: warn that reliability cannot be assessed and
    recommend re-running with `force_recompute=True`.
  - If two or more: warn if 1.96 * SE > 1 s across computation means.

---

## Cache Key Format

Both `problem_key` and `estimator_key` share the same format:

```
{ClassName}__{joblib_hash}
```

The hash is derived by the persistence layer using `joblib.hash()`, which hashes
the full instance state via pickle. This captures all stored attributes including
default parameter values, so that a change in any default automatically invalidates
existing results. See the
[parameter-based identity design](2026-04-21-parameter-based-identity-design.md)
for the rationale.

```python
import joblib
problem_key   = f'{type(problem).__name__}__{joblib.hash(problem)}'
estimator_key = f'{type(est).__name__}__{joblib.hash(est)}'
```

For estimators, only the unfitted state is hashed — fitted attributes (`coef_`,
etc.) are not present before `fit()` is called, so the key reflects constructor
parameters only.

Example: `EmpiricalDataProblem__8db1480a05fa91f6a3a86c57bb4f6af7`

---

## Changes to `EmpiricalDataExperiment`

Seeding parameters (`generator`, `seed_scope`, `seed_progression`) are added by the
prerequisite seeding refactor. This spec adds:

### New `run()` parameters

```python
run(force_recompute=False, ignore_cache=False)
```

### New module constant

```python
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
```

### `Metric` base class

All existing stat classes inherit from a new `Metric` base class (see Metric Warning
Behavior). The runner parameter stays named `stats`.

---

## Run File

Each call to `run()` writes a file to `results/runs/<run_id>.json`. The run ID is a
timestamp with a short random suffix to avoid collisions: `20260421-143045-a3f7`.

```json
{
  "run_id": "20260421-143045-a3f7",
  "timestamp": "2026-04-21T14:30:45",
  "environment": {
    "python": "3.11.5",
    "platform": "macOS-15.3.0-arm64"
  },
  "experiment_spec": {
    "problems": ["EmpiricalDataProblem__8db1480a", "..."],
    "estimators": ["RidgeEM__3f2a91c0", "RidgeLOOCV__7d4b2e11"],
    "ns": [50, 100, 200],
    "reps": 20,
    "seed": 0,
    "generator": "MT19937",
    "seed_scope": "series"
  },
  "summary": {
    "trials_computed": 42,
    "trials_retrieved": 18
  }
}
```

`environment` uses only stdlib (`sys`, `platform`) — no extra dependency. Summary
counts are always at the **trial level** — for `seed_scope='series'`, a series of
`reps` reps contributes `reps` to each counter. Counts are mutually exclusive and
exhaustive: every trial is either computed or retrieved. Failed trials (exception
during fitting) record NaN in the metric file; this is visible there but not
separately counted in the summary.

The `experiment_spec` records the full parameter set, including cache keys, so it is
possible to reconstruct which metric files belong to this run without traversing
the directory tree. The run file itself is informational — it is not read during
`run()` and plays no role in cache lookup.

---

## Scope

This spec covers `EmpiricalDataExperiment` only. `Experiment` (synthetic problems)
uses a single experiment-scoped RNG and is not modified here. The file structure is
designed to accommodate `Experiment` caching in a future spec without breaking
changes.

---

## Conventions

- `results/` lives at the repository root (sibling of `experiments/`). Notebooks
  reference it as `'../results'` relative to `experiments/`, but use `CACHE_DIR`
  rather than hardcoding the path.
- Metric JSON files are written atomically (write to a temp path, then
  `os.replace`) to avoid corrupted entries if a run is interrupted mid-write.
- Cache keys change automatically when constructor parameters or their defaults
  change (via `joblib.hash()`). Changing an estimator implementation without
  changing its parameters leaves stale cache entries that will silently be reused;
  use `force_recompute=True` as a workaround until systematic cache invalidation
  is addressed (see Future). Orphaned directories with unrecognised keys are
  visible and harmless.
- `results/` is tracked in git via a `results/.gitignore` file containing `*` and
  `!.gitignore` — the standard pattern that ignores all content while keeping the
  directory itself. Committing specific result files is possible with `git add -f`
  but expected to be rare (e.g. small reference results for CI).

---

## Files

- Modify: `experiments/experiments.py`:
  - Add `Metric` base class; update all existing stat classes to inherit from it
  - Add `CACHE_DIR` module constant
  - Add `force_recompute` and `ignore_cache` parameters to `run()`
  - Implement caching loop in `run()`: key derivation, file read/write, warning calls
- Create: `results/.gitignore` — containing `*` and `!.gitignore`

---

## Future

### Random estimators

The per-trial file structure accommodates randomised estimators without structural
changes. Possible approaches include threading the trial RNG through to `fit()`, or
deriving a separate integer seed per estimator from the trial seed. A loop order of
problems -> trials -> estimators (which matches the current runner) is particularly
natural: the outer trial seed fixes the data split while estimator seeds can be
varied independently in the innermost loop without affecting the split or other
estimators.

If independent control over estimator randomness is needed — for example to average
over estimator randomness at a fixed data split — a nested `<est_seed>/` level
inside `<trial_seed>/` is the natural extension. This leaves all outer path levels
unchanged and is fully backwards compatible with existing results.

### Extending to `Experiment` (synthetic problems)

`Experiment` uses a single experiment-scoped RNG and produces a result array of
shape `(len(problems), len(ns), len(estimators), reps, len(stats))`. Caching it
whole is simpler than per-trial caching: the entire array maps to one key derived
from the full experiment spec. Proposed file structure:

```
results/
  experiment/
    <experiment_key>/
      <metric_name>.npy   <- full result array, shape (problems, ns, estimators, reps)
```

`.npy` (rather than JSON) is appropriate here since the arrays are large and
numerical. The `computations`/`retrievals` history structure from the per-trial
format does not apply — whole-experiment caching treats the run as atomic.

### Cache invalidation

When an estimator's implementation changes without its constructor parameters
changing, existing cache entries for that estimator are stale but indistinguishable
from valid ones by key alone. A future spec should address systematic invalidation —
for example a `cache clear <estimator>` utility, or storing a source hash alongside
the parameter hash in the cache key.
