# Result Persistence Design

## Prerequisite

This spec requires the [preprocessing pipeline design](2026-04-17-preprocessing-pipeline-design.md)
and the [seeding refactor design](2026-04-18-empirical-experiment-seeding-refactor-design.md)
to be implemented first. The preprocessing design establishes `EmpiricalDataProblem.__repr__`
as the canonical, session-stable string identity used here to derive cache keys.
The seeding refactor establishes `n_train`, `generator`, `seed_scope`, and
`seed_progression` as first-class parameters on `EmpiricalDataExperiment`.
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

### Caching support by scope

The cache folder (`per_series/` or `per_trial/`) is determined by `seed_scope`.
Within that, the seed value used in the path is determined by `seed_progression`:

| `seed_scope` | `seed_progression` | File seed value | Cache granularity |
|---|---|---|---|
| `'series'` | `'fixed'` | `seed` | all reps together per estimator |
| `'series'` | `'sequential'` | `seed + prob_idx` | all reps together per estimator |
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

### `problem_key`

```
{ClassName}__{short_hash}
```

`EmpiricalDataProblem` has a stable `__repr__` that fully encodes its identity,
including any `x_transforms` (e.g. `PolynomialExpansion`). The `problem_key` is
derived from it by the persistence layer:

```python
import hashlib
problem_key = f'{type(problem).__name__}__{hashlib.md5(repr(problem).encode()).hexdigest()}'
```

No `cache_key()` method is added to `EmpiricalDataProblem` — the persistence
layer computes the key directly from `repr()`. `__hash__` is `hash(repr(self))`
for consistent in-memory identity (sets, dict keys); the persistence layer uses
`repr()` directly for cross-session stable filesystem paths.

Example: `EmpiricalDataProblem__a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0`

### `estimator_key`

Mirrors the `problem_key` format:

```
{ClassName}__{short_hash}
```

`short_hash` is the first 8 hex chars of `md5(str(sorted(est.get_params().items())))`.
Estimators are treated as value objects whose semantic identity is their class and
constructor parameters — the mutable fitted state (`coef_`, etc.) is not part of
the key. `get_params()` is the standard sklearn method returning the constructor
parameters. The class name is not included in the hash — it appears as the directory
prefix and already disambiguates classes.

Example: `RidgeEM__7d3a1f2e`

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
- No automatic cache invalidation. The caller deletes stale entries when needed
  (e.g. after changing a problem definition). Orphaned directories with unrecognised
  keys are visible and harmless.
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

Some estimators (e.g. those using stochastic optimisation or random initialisation)
carry their own internal randomness. Under `generator='MT19937'` with
`seed_scope='series'`, an estimator's effective seed depends on how much generator
state was consumed by the split before `fit()` is called — this is deterministic
but fragile and hard to reason about.

The clean solution is `generator='PCG64'` with `seed_scope='trial'`: each trial
gets a parent Generator, from which child generators are spawned deterministically
— one for the split, one passed to the estimator via `set_params(random_state=child_rng)`.
This makes the trial fully self-contained: the trial seed alone determines both the
data split and every estimator's internal randomness. The `estimator_key` remains
unchanged (it captures class and deterministic constructor params), and the
`<trial_seed>` in the file path captures the rest.

This requires estimators to accept a `random_state` parameter (the sklearn convention).
Estimators without internal randomness ignore it.
