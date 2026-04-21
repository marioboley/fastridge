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

Add optional result caching to `EmpiricalDataExperiment` so that re-running with a
modified estimator list only recomputes affected units, reusing stored results for
everything else. Eliminates the need to rerun full experiments (currently >1 h) when
iterating on a subset of estimators.

---

## Terminology

**Trial** — the atomic unit of work: one (problem, estimator, n, seed) combination
producing one scalar per metric.

**Series** — all reps for one (problem, estimator, n, base\_seed, reps)
combination, stored as a 1-D array of length `reps` per metric.

---

## Seeding

Seeding behaviour is controlled by three orthogonal parameters on
`EmpiricalDataExperiment`:

### `generator`

*Which* numpy BitGenerator is used:

- `'PCG64'` — `np.random.Generator(np.random.PCG64(seed))`. Default.
- `'MT19937'` — `np.random.RandomState(seed)`. Numerically equivalent to the
  legacy `np.random.seed` + `np.random.choice` path for datasets without
  polynomial subsampling.

### `seed_scope`

*When* the seed is applied:

- `'series'` — seed is reset once per problem, before that problem's rep loop.
  Default.
- `'trial'` — seed is reset once per rep, inside the rep loop, before the split.
- `'experiment'` — seed is set once and a single RNG object advances through
  everything.

### `seed_progression`

*What value* is used at each scope boundary:

- `'fixed'` — always use `seed`. Default (gives the same split sequence for
  every problem under `seed_scope='series'`).
- `'sequential'` — use `seed + unit_idx`, where `unit_idx` is the rep index for
  `seed_scope='trial'` or the problem index for `seed_scope='series'`.

### Defaults and legacy equivalence

Default: `generator='MT19937', seed_scope='series', seed_progression='fixed'`.
This is numerically identical to the current implementation (for datasets without
polynomial subsampling) — no existing notebook needs to change.

### Legacy vs. forward-looking scopes

`seed_scope='trial'` is the forward-looking mode. It offers the finest cache
granularity: individual reps can be added or recomputed independently, and
estimators can be added without disturbing existing results. It also extends
naturally to randomised estimators — since each estimator has its own
`<estimator_key>/<trial_seed>` path, passing `trial_seed` directly as an integer
`random_state` gives each estimator independent, reproducible randomness with no
coupling to the data split. This works with sklearn estimators using either
generator.

`seed_scope='series'` is a legacy mode required to cache the NeurIPS 2023 empirical
results, which were computed with per-problem seeding. Adding estimators is still
possible (new estimator key = new directory), but extending reps requires
recomputing the whole series since all reps are stored together. Individual trial
RNG states are not independently derivable from the base seed.

`seed_scope='experiment'` is a legacy mode required to cache the NeurIPS 2023
synthetic results (`Experiment` runner), which use a single advancing RNG across
the whole run. Only whole-experiment caching is possible. This is still valuable
beyond legacy reproduction: caching the full result array means plotting, table
generation, and other analysis code can be iterated freely without rerunning
expensive computations.

### Caching support by scope

The cache folder is determined by `seed_scope`. Within that, the seed value used in
the path is determined by `seed_progression`:

| `seed_scope` | `seed_progression` | File seed value | Cache granularity |
|---|---|---|---|
| `'series'` | `'fixed'` | `seed` | all reps together per estimator (legacy) |
| `'series'` | `'sequential'` | `seed + prob_idx` | all reps together per estimator (legacy) |
| `'trial'` | `'sequential'` | `seed + rep_idx` | one rep at a time per estimator |

`generator` affects both the path (via a `<generator>/` directory level) and the
numerical results — two runs with the same seed but different generators produce
different splits and must be stored separately.

---

## Caching

### Global cache directory

`CACHE_DIR` is a module-level constant in `experiments.py`, defaulting to the
`results/` directory at the repository root. Notebooks can override it:

```python
import experiments
experiments.CACHE_DIR = '/path/to/custom/cache'
```

Caching is always active — there is no opt-in flag. When `CACHE_DIR` is set (which
it always is by default), every `run()` call reads from and writes to the cache.

### `overwrite` parameter on `run()`

`run(overwrite=False)` — when `True`, recomputes all trials/series and overwrites
existing cache files. Useful when an estimator implementation has changed without
its `get_params()` output changing.

### Behaviour during `run()`

For each (problem, estimator) pair:

1. Derive the cache key (see below).
2. Check whether the corresponding file exists under `CACHE_DIR`.
3. **Hit**: load the stored result and place it into `self.result_`. Skip fitting.
4. **Miss** (or `overwrite=True`): fit the estimator, record results, write file.

---

## File Structure

```
results/
  per_series/
    <problem_key>/
      <n_train>/
        <estimator_key>/
          <reps>/
            <generator>/
              <base_seed>/
                metrics.npz       <- one key per metric, arrays of shape (reps,)
                predictions/      <- (future) one .npy per rep
                estimators/       <- (future) one .pkl per rep
                meta.json         <- (future) per-series metadata
  per_trial/
    <problem_key>/
      <n_train>/
        <estimator_key>/
          <generator>/
            <trial_seed>/
              metrics.npz         <- one key per metric, scalar values
              predictions.npy     <- (future) y_pred array for this trial
              estimator.pkl       <- (future) fitted estimator for this trial
              meta.json           <- (future) per-trial metadata
  runs/                           <- (future) one file per experiment run
```

The top-level subfolder is determined by `seed_scope` alone: `'series'` ->
`per_series/`, `'trial'` -> `per_trial/`. `<n_train>/` captures the actual training
set size — two experiments on the same problem with different `n_train` values
produce separate paths. `<reps>/` means series of different lengths coexist without
conflict. `<generator>/` separates results from different BitGenerators since the
same seed produces different numerical outcomes across generators. Using a directory
per trial/series rather than a flat file allows future artifact types to be added
without changing the key structure.

### Storage format: `.npz`

`.npz` is NumPy's compressed archive format. Internally it is a ZIP file containing
one `.npy` binary file per named array, written with `np.savez_compressed` and read
back with `np.load`. It requires no extra dependency (NumPy is already required) and
is efficient for numerical arrays of any shape. Limitations: no partial read/write
(the whole file is loaded at once), and non-array data (strings, dicts) require
wrapping. For comparison: HDF5 (`h5py`) supports hierarchical storage and partial
reads but adds a dependency; JSON is suitable for metadata but not arrays; pickle is
flexible but Python-only and not safe for untrusted input. For metric arrays,
`.npz` is the right default.

`metrics.npz` stores one key per metric (e.g. `prediction_r2`, `fitting_time`):
- `per_series/` files: array of shape `(reps,)` — one value per rep
- `per_trial/` files: scalar — the single rep result

Storing metrics as separate keys means adding or removing a metric from `stats` only
invalidates that metric's values; all other metrics are reused. `n_train` is encoded
in the directory path, not inside the file.

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

### New constructor parameters

```python
EmpiricalDataExperiment(
    problems, estimators, reps, ns,
    seed=None,
    generator='PCG64',        # new: 'PCG64' | 'MT19937'
    seed_scope='series',      # new: 'series' | 'trial' | 'experiment'
    seed_progression='fixed', # new: 'fixed' | 'sequential'
    stats=None, est_names=None, verbose=True
)
```

See the [seeding refactor design](2026-04-18-empirical-experiment-seeding-refactor-design.md)
for the full loop structure and `_make_rng` implementation.

---

## Scope

This spec covers `EmpiricalDataExperiment` only. `Experiment` (synthetic problems)
uses a single experiment-scoped RNG and is not modified here. The file structure and
parameter names are designed to accommodate `Experiment` caching in a future spec
without breaking changes.

---

## Conventions

- `results/` lives at the repository root (sibling of `experiments/`). Notebooks
  reference it as `'../results'` relative to `experiments/`, but use `CACHE_DIR`
  rather than hardcoding the path.
- `metrics.npz` files are written atomically (write to a temp path, then
  `os.replace`) to avoid corrupted entries if a run is interrupted mid-write.
- Cache keys change automatically when constructor parameters or their defaults
  change (via `joblib.hash()`). Changing an estimator implementation without
  changing its parameters requires manual deletion or `overwrite=True`. Orphaned
  directories with unrecognised keys are visible and harmless.
- `results/` is tracked in git via a `results/.gitignore` file containing `*` and
  `!.gitignore` — the standard pattern that ignores all content while keeping the
  directory itself. Committing specific result files is possible with `git add -f`
  but expected to be rare (e.g. small reference results for CI).

---

## Files

- Modify: `experiments/experiments.py` — add `generator`, `seed_scope`,
  `seed_progression` to `EmpiricalDataExperiment.__init__`; update `run()`;
  add `CACHE_DIR` module constant
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
