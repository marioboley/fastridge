# Empirical Experiment Seeding Refactor Design

## Goal

Refactor `EmpiricalDataProblem.get_X_y` and `EmpiricalDataExperiment` to:

1. Unify the call site — `get_X_y(n_train, rng=None)` is always invoked inside the
   rep loop, paralleling `linear_problem.rvs(n, rng=rng)`.
2. Thread randomness explicitly — the `rng` argument controls both stochastic
   x_transforms (e.g. `PolynomialExpansion` column subsampling) and the train/test
   split, replacing uncontrolled global numpy state.
3. Support legacy and modern numpy seeding — `rng=None` falls back to global state
   (controlled via `np.random.seed`); an explicit `Generator` uses the modern API.
4. Support all seed scopes from the persistence spec — trial, series, experiment —
   by varying only which `rng` is passed to `get_X_y`, not when it is called.
5. Align `EmpiricalDataExperiment.ns` with `Experiment.ns` — integer matrix,
   no implicit fraction arithmetic inside the runner — as a step toward eventual
   unification of the two experiment classes.

---

## Relationship to Other Specs

**Builds on**: the
[preprocessing pipeline design](2026-04-17-preprocessing-pipeline-design.md),
which established `x_transforms` / `y_transforms` on `EmpiricalDataProblem` and
introduced `PolynomialExpansion` and `OneHotEncodeCategories`. That design is
already implemented; this spec addresses the polynomial subsampling seeding issue
it identified but left open (the `PolynomialExpansion` internal seeding was never
implemented — current code still uses `np.random.choice` against global state).

**Prerequisite for**: the
[result persistence design](2026-04-16-result-persistence-design.md), which
requires `n_train` as a first-class integer known before the trial loop and
`seed_mode` / `seed_scope` / `seed_progression` parameters on
`EmpiricalDataExperiment`.

---

## `EmpiricalDataProblem.get_X_y`

### New signature

```python
def get_X_y(self, n_train=None, rng=None):
```

### Behaviour when `n_train` is given (primary path)

Returns `(X_train, X_test, y_train, y_test)`.

Internal pipeline:

1. Load full dataset via `get_dataset` (already cached at the data layer).
2. Apply column drops, NaN policy, index reset — same as today.
3. Split X / y.
4. Apply y_transforms (deterministic; `rng` not forwarded).
5. Apply each x_transform as `fn(X, rng=rng)` — see x_transform protocol below.
6. Call `sklearn.model_selection.train_test_split(X, y, train_size=n_train,
   random_state=rng)` — `random_state` accepts a `Generator` or `None`.
7. Compute `std = X_train.std(); non_zero = std[std != 0].index` and restrict
   both `X_train` and `X_test` to `non_zero` columns (zero-variance filter,
   currently in the experiment runner).

### Behaviour when `n_train` is `None` (backward-compat path)

Returns `(X, y)` — full data, no split, no zero-variance filter. Transforms are
still applied with `rng=rng`. All existing doctests and notebook cells that call
`get_X_y()` without arguments continue to work.

### rng semantics

`rng=None` passes `None` to both transforms and `train_test_split`, meaning numpy
global state governs all randomness. When `np.random.seed(seed)` has been called
before `get_X_y`, the output is fully deterministic — this is the legacy path.

An explicit `np.random.Generator` is consumed sequentially: first by any stochastic
x_transforms in list order, then by `train_test_split`. The caller is responsible
for creating and advancing the generator.

---

## x_transform Protocol Extension

The documented contract for x_transforms changes from `fn(X)` to `fn(X, rng=None)`.

All built-in transforms are updated:

- `PolynomialExpansion.__call__(self, X, rng=None)`: uses `rng.choice` when `rng`
  is provided, else `np.random.choice` (legacy global state).
- `OneHotEncodeCategories.__call__(self, X, rng=None)`: accepts and ignores `rng`
  (deterministic).

User-supplied x_transforms must also accept `rng=None`. Transforms that are purely
deterministic can declare `def __call__(self, X, rng=None)` and ignore the argument.

y_transforms are unchanged — they remain single-argument callables.

---

## `n_train_from_proportion` Helper

A module-level helper in `problems.py`:

```python
def n_train_from_proportion(problems, prop=0.7):
    """Return per-problem n_train ints from a train proportion and the dataset registry.

    Raises KeyError if any problem's dataset lacks an 'n' entry in DATASETS.
    The registry count may differ from the actual loaded row count when nan_policy
    drops rows; a warning is issued by the data layer in that case.
    """
    return np.array([int(DATASETS[p.dataset]['n'] * prop) for p in problems])
```

Notebooks that previously passed `test_prop=0.3` migrate to:

```python
ns = n_train_from_proportion(problems, 0.7)
exp = EmpiricalDataExperiment(problems, estimators, n_iterations, ns=ns, ...)
```

This makes the per-problem n values explicit and visible at experiment-definition
time rather than computed implicitly inside the runner.

---

## `EmpiricalDataExperiment` Changes

### Constructor

`test_prop` is removed. `ns` (int or array-like, analogous to `Experiment.ns`)
replaces it:

```python
def __init__(self, problems, estimators, n_iterations, ns,
             seed=None,
             seed_mode='legacy',       # 'legacy' | 'modern'
             seed_scope='series',      # 'series' | 'trial' | 'experiment'
             seed_progression='fixed', # 'fixed' | 'sequential'
             stats=None, est_names=None, verbose=True):
```

`ns` is normalised to a 2-D int array of shape `(n_problems, n_sizes)` using the
same broadcasting logic as `Experiment`:

```python
self.ns = np.atleast_2d(ns)
if len(self.ns) != len(self.problems):
    self.ns = self.ns.repeat(len(self.problems), axis=0)
```

Typically `n_sizes=1` for empirical experiments; multiple n values per problem
support learning-curve experiments.

### `run()` loop structure

The key architectural change: `get_X_y` is always called inside the rep loop with
`n_train`. The seeding scope determines which `rng` is passed; the call site is
unconditional.

```python
def run(self, overwrite=False):
    ...
    rng = None

    # experiment scope: seed once before all loops
    if self.seed_scope == 'experiment':
        rng = _make_rng(self.seed, self.seed_mode)

    for prob_idx, problem in enumerate(self.problems):
        for n_idx, n_train in enumerate(self.ns[prob_idx]):

            # series scope: reseed once per (problem, n) pair
            if self.seed_scope == 'series':
                seed_val = _series_seed(self.seed, prob_idx, self.seed_progression)
                rng = _make_rng(seed_val, self.seed_mode)

            for iter_idx in range(self.n_iterations):

                # trial scope: reseed once per rep
                if self.seed_scope == 'trial':
                    seed_val = _trial_seed(self.seed, iter_idx, self.seed_progression)
                    rng = _make_rng(seed_val, self.seed_mode)

                X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)

                for est_idx, est in enumerate(self.estimators):
                    _est = clone(est, safe=False)
                    try:
                        t0 = time.time()
                        _est.fit(X_train, y_train)
                        _est.fitting_time_ = time.time() - t0
                    except Exception as e:
                        warnings.warn(...)
                        continue
                    for stat in self.stats:
                        self.__dict__[str(stat) + '_'][
                            iter_idx, prob_idx, n_idx, est_idx] = stat(
                                _est, problem, X_test, y_test)
```

### Legacy equivalence

Default: `seed_mode='legacy', seed_scope='series', seed_progression='fixed'`.

Under this configuration the new loop is numerically equivalent to the current
implementation for datasets whose polynomial expansion does not trigger column
subsampling (i.e. all but Twitter at d=2 or d=3). For Twitter, the subsampling
previously consumed from an unseeded global state; it now consumes from the same
seeded global state as the split, which changes the specific columns selected but
makes the result deterministic and reproducible for the first time.

### Result array shape

`self.__dict__[str(stat) + '_']` is allocated with shape
`(n_iterations, n_problems, n_sizes, n_estimators)` — adding the `n_sizes`
dimension (previously hardcoded to 1) to match `Experiment`'s layout.

---

## Private helpers

Module-level private functions in `experiments.py`:

```python
def _make_rng(seed, mode):
    if mode == 'modern':
        return np.random.default_rng(seed)
    # legacy: apply seed to global state and return None
    if seed is not None:
        np.random.seed(seed)
    return None

def _series_seed(base_seed, prob_idx, progression):
    if base_seed is None:
        return None
    if progression == 'fixed':
        return base_seed
    if progression == 'sequential':
        return base_seed + prob_idx

def _trial_seed(base_seed, iter_idx, progression):
    if base_seed is None:
        return None
    if progression == 'fixed':
        return base_seed
    if progression == 'sequential':
        return base_seed + iter_idx
```

`_make_rng` centralises the mode dispatch so the loop body does not branch on
`seed_mode` directly.

---

## Relation to Result Persistence

This refactor is a prerequisite for result persistence. It establishes:

- `n_train` as a first-class integer known before the trial loop (required for
  the `<n_train>` path component in cache keys).
- `seed_mode`, `seed_scope`, `seed_progression` on `EmpiricalDataExperiment`
  (required by the persistence spec's seeding section).
- A single unconditional call site for `get_X_y` inside the rep loop (required
  for the per-trial and per-series cache granularities to map cleanly to the loop
  structure).

---

## Path toward unification with `Experiment`

After this refactor both classes share:

- `ns` as a 2-D int array with the same broadcasting convention.
- Result arrays of shape `(n_iterations, n_problems, n_sizes, n_estimators)`.
- `seed_mode` / `seed_scope` / `seed_progression` parameters.

The remaining structural difference is that `Experiment` calls `rvs(n, rng)` twice
per iteration — once for training data, once for a fixed test set — while
`EmpiricalDataExperiment` calls `get_X_y(n_train, rng)` once and receives a split.
Full unification would require `rvs` to also return a train/test split (or a shared
adapter), which is out of scope here. The shared `ns` layout and seeding parameters
are the necessary foundation for that future step.

---

## Files

- **Modify** `experiments/problems.py`:
  - `EmpiricalDataProblem.get_X_y` — add `n_train`, `rng` params; add split and
    zero-variance filter when `n_train` given; forward `rng` to x_transforms.
  - `PolynomialExpansion.__call__` — add `rng=None`; use `rng.choice` when provided.
  - `OneHotEncodeCategories.__call__` — add `rng=None` (ignored).
  - Add `n_train_from_proportion` helper.

- **Modify** `experiments/experiments.py`:
  - `EmpiricalDataExperiment.__init__` — replace `test_prop` with `ns`; add
    `seed_mode`, `seed_scope`, `seed_progression`.
  - `EmpiricalDataExperiment.run` — restructure loop; remove zero-variance filter
    block; add `_series_seed` / `_trial_seed` helpers.

- **Modify** `experiments/real_data.ipynb` — replace `test_prop=0.3` with
  `ns=n_train_from_proportion(problems, 0.7)`; update imports.
