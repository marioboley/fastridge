# Result Persistence Design

## Goal

Add optional result caching to `EmpiricalDataExperiment` so that re-running with a
modified estimator list only recomputes affected units, reusing stored results for
everything else. Eliminates the need to rerun full experiments (currently >1 h) when
iterating on a subset of estimators.

---

## Terminology

**Trial** — the atomic unit of work: one (problem, estimator, n, seed) combination
producing one scalar per metric.

**Series** — all reps for one (problem, estimator, n, base\_seed, n\_iterations)
combination, stored as a 1-D array of length `n_iterations` per metric.

---

## Seeding

Seeding behaviour is controlled by three orthogonal parameters on
`EmpiricalDataExperiment`:

### `seed_mode`

*How* randomness is managed:

- `'legacy'` — uses `np.random.seed()` (global numpy state). Current behaviour.
- `'modern'` — uses `np.random.default_rng()` (explicit Generator object). Not yet
  implemented; reserved for future use.

### `seed_scope`

*When* the seed is applied:

- `'series'` — seed is reset once per problem, before that problem's rep loop.
  Current `EmpiricalDataExperiment` behaviour.
- `'trial'` — seed is reset once per rep, inside the rep loop, before the split.
- `'experiment'` — seed is set once at construction time and a single RNG object
  advances through everything. Current `Experiment` behaviour; not applicable to
  `EmpiricalDataExperiment` in this spec.

### `seed_progression`

*What value* is used at each scope boundary:

- `'fixed'` — always use `base_seed`. Current behaviour (`np.random.seed(seed)` gives
  the same split sequence for every problem).
- `'sequential'` — use `base_seed + unit_idx`, where `unit_idx` is the rep index for
  `seed_scope='trial'` or the problem index for `seed_scope='series'`.
- `'random'` — derive seed from `hash(base_seed, unit_idx)`. Future; not implemented
  in this spec.

### Defaults and legacy equivalence

Default: `seed_mode='legacy', seed_scope='series', seed_progression='fixed'`.
This is numerically identical to the current implementation — no existing notebook
needs to change.

### Caching support by mode

The cache folder (`per_series/` or `per_trial/`) is determined by `seed_scope`.
Within that, the seed value used as the filename is determined by `seed_progression`:

| `seed_scope` | `seed_progression` | File seed value | Cache granularity |
|---|---|---|---|
| `'series'` | `'fixed'` | `base_seed` | all reps together per estimator |
| `'series'` | `'sequential'` | `base_seed + prob_idx` | all reps together per estimator |
| `'trial'` | `'sequential'` | `base_seed + rep_idx` | one rep at a time per estimator |

`seed_mode` (`'legacy'` / `'modern'`) does not affect the path or cache lookup — it
only determines which numpy API performs the seeding.

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
          <n_iterations>/
            <base_seed>/
              metrics.npz       ← one key per metric, arrays of shape (n_iterations,)
              predictions/      ← (future) one .npy per rep
              estimators/       ← (future) one .pkl per rep
              meta.json         ← (future) per-series metadata
  per_trial/
    <problem_key>/
      <n_train>/
        <estimator_key>/
          <trial_seed>/
            metrics.npz         ← one key per metric, scalar values
            predictions.npy     ← (future) y_pred array for this trial
            estimator.pkl       ← (future) fitted estimator for this trial
            meta.json           ← (future) per-trial metadata
  runs/                         ← (future) one file per experiment run
```

The top-level subfolder is determined by `seed_scope` alone: `'series'` →
`per_series/`, `'trial'` → `per_trial/`. `<n_train>/` captures the actual training
set size — two experiments on the same problem with different `test_prop` values
produce different n_train values and therefore separate paths, with no need to
encode `test_prop` in any hash. `<n_iterations>/` means series of different lengths
coexist without conflict. Using a directory per trial/series rather than a flat file
allows future artifact types to be added without changing the key structure.

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
- `per_series/` files: array of shape `(n_iterations,)` — one value per rep
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

`polynomial` is moved from `EmpiricalDataExperiment` into `EmpiricalDataProblem`
as a constructor parameter (default `None`). It defines the feature space and belongs
to the problem's identity, not the experiment runner. In this spec the move is
identity-only: `run()` still applies the polynomial expansion itself, reading
`problem.polynomial` instead of `self.polynomial`. The natural next step — moving
the expansion into `EmpiricalDataProblem.get_X_y()` so the problem is fully
self-contained — is left for a future refactoring. The existing problem sets that
currently pass `polynomial` to the experiment (e.g. for d=2 and d=3 problems)
are updated to pass it to the problem constructor instead.

`EmpiricalDataProblem` gains a `cache_key()` method returning the first 8 hex chars
of `md5(str((self.dataset, self.target, self.drop, self.nan_policy,
str(self.transforms), self.polynomial)))`. The class name is NOT included in the
hash — it is already present as the directory prefix and including it in the hash
would be redundant. This treats the problem as a value object whose identity is
fully determined by its constructor arguments. `__hash__` is updated to return
`hash(self.cache_key())` so in-memory identity (dict keys, sets) and filesystem
identity share the same canonical basis. Python's `hash(identity_tuple)` is not
reused because Python randomises string hashing across sessions (`PYTHONHASHSEED`).

`test_prop` is not part of the problem key — it is captured by `<n_train>/` in the
path, since different `test_prop` values on the same problem yield different n_train
values and thus different directories.

The `problem_key` is `f'{type(problem).__name__}__{problem.cache_key()}'`.

Example: `EmpiricalDataProblem__a3f2b1c4`

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
    problems, estimators, n_iterations,
    test_prop=0.3, seed=None,
    seed_mode='legacy',         # new
    seed_scope='series',        # new
    seed_progression='fixed',   # new
    cache=False,                # new
    polynomial=None, stats=None, est_names=None, verbose=True
)
```

### `run(overwrite=False)` loop change for `seed_scope='trial'`

```python
for iter_idx in range(self.n_iterations):
    if self.seed_scope == 'trial':
        seed_val = (self.seed or 0) + iter_idx  # 'sequential'
        np.random.seed(seed_val)                 # 'legacy' mode
    X_train, X_test, y_train, y_test = train_test_split(...)
```

`seed_scope='series'` keeps the existing reset before the problem's rep loop.

---

## Scope

This spec covers `EmpiricalDataExperiment` only. `Experiment` (synthetic problems)
uses `seed_scope='experiment'` and is not modified here. The file structure and
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

- Modify: `experiments/experiments.py` — add `seed_mode`, `seed_scope`,
  `seed_progression` to `EmpiricalDataExperiment.__init__`; update `run()`;
  add `CACHE_DIR` module constant
- Modify: `experiments/problems.py` — add `polynomial` parameter to
  `EmpiricalDataProblem.__init__`; add `cache_key()` method; update `__hash__` to
  derive from `cache_key()`; update `NEURIPS2023_D2`, `NEURIPS2023_D3` problem sets
  to pass `polynomial` to the problem rather than the experiment
- Create: `results/.gitignore` — containing `*` and `!.gitignore`

---

## Future

### Polynomial feature subsampling

When `polynomial` is set and the expanded feature matrix exceeds 35 million entries,
`EmpiricalDataExperiment.run()` currently applies random column subsampling
(`np.random.choice`) to reduce dimensionality. This sampling happens **before** the
per-problem seed reset, meaning it consumes from whatever numpy state was current at
that point — it is not deterministic in a clearly reasoned way, and it is not
captured in the trial or series cache key.

Two paths forward:

1. **Move into the problem**: make polynomial expansion (including subsampling) a
   responsibility of `EmpiricalDataProblem` rather than the experiment. The problem
   would apply a fixed subsampling derived deterministically from its own hash. The
   result is that the feature space is fully determined by the problem identity and
   is already captured in the `problem_key`. No experiment-level randomness for
   feature construction.

2. **Move into the estimator**: treat the polynomial transformer as part of a
   pipeline estimator. The subsampling seed would then be part of `get_params()` and
   would be captured in the `estimator_key`. This keeps the feature construction
   visible as a modelling choice rather than a data preprocessing step.

Either path removes an uncontrolled source of randomness from the experiment loop.
Path 1 is more natural given the current architecture; path 2 aligns better with the
long-term goal of estimators carrying all modelling decisions.

### Random estimators

Some estimators (e.g. those using stochastic optimisation or random initialisation)
carry their own internal randomness. Under the current `seed_mode='legacy'` design,
an estimator's effective seed depends on how much global numpy state was consumed by
the split before `fit()` is called — this is deterministic but fragile and hard to
reason about.

The clean solution is `seed_mode='modern'` with `seed_scope='trial'`: each trial
gets a parent `default_rng` Generator, from which child generators are spawned
deterministically — one for the split, one passed to the estimator via
`set_params(random_state=child_rng)`. This makes the trial fully self-contained: the
trial seed alone determines both the data split and every estimator's internal
randomness. The `estimator_key` remains unchanged (it captures class and
deterministic constructor params), and the `trial_seed` in the file path captures
the rest.

This design requires that estimators accept a `random_state` parameter (the sklearn
convention). Estimators without internal randomness ignore it. This is the primary
motivation for eventually deprecating `seed_mode='legacy'` in favour of `'modern'`.
