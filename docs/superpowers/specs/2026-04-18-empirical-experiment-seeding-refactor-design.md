# Empirical Experiment Seeding Refactor Design

## Goal

Refactor `EmpiricalDataProblem` and `EmpiricalDataExperiment` to:

1. Rename `get_X_y` to `rvs` — unifying the problem interface with `linear_problem`
   so both return `(X_train, X_test, y_train, y_test)` via `rvs(n_train, rng=rng)`.
2. Unify the call site — `rvs(n_train, rng=rng)` is always invoked inside the
   rep loop, with no pre-computation of data outside the loop.
3. Thread randomness explicitly — the `rng` argument controls both stochastic
   x_transforms (e.g. `PolynomialExpansion` column subsampling) and the train/test
   split, replacing uncontrolled global numpy state.
4. Support legacy and modern numpy seeding — `rng=None` falls back to global state
   (controlled via `np.random.seed`); an explicit `Generator` uses the modern API.
5. Support all seed scopes — trial, series, experiment — and seed progressions —
   fixed, sequential — by varying only which `rng` is passed to `rvs`.
6. Align `EmpiricalDataExperiment` with `Experiment` — replace `test_prop` and
   `n_iterations` with `ns` (integer matrix) and `reps`, matching the existing
   parameter names — as a step toward eventual unification.

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

## `EmpiricalDataProblem.rvs`

### New signature

```python
def rvs(self, n_train=None, rng=None):
```

`get_X_y` is renamed to `rvs`. The old name is removed with no deprecation period —
usage is fully internal (experiment runners and notebooks in this repository).

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
still applied with `rng=rng`. Existing doctests and notebook cells that call
`get_X_y()` must be updated to call `rvs()`.

### rng semantics

`rng=None` passes `None` to both transforms and `train_test_split`, meaning numpy
global state governs all randomness. When `np.random.seed(seed)` has been called
before `rvs`, the output is fully deterministic — this is the legacy path.

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
exp = EmpiricalDataExperiment(problems, estimators, reps, ns=ns, ...)
```

This makes the per-problem n values explicit and visible at experiment-definition
time rather than computed implicitly inside the runner.

---

## `EmpiricalDataExperiment` Changes

### Constructor

`test_prop` and `n_iterations` are removed. `ns` and `reps` replace them:

```python
def __init__(self, problems, estimators, reps, ns,
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

### `_set_seed(self, unit_idx)` class method

A single private method mutates `self.rng` (modern mode) or global numpy state
(legacy mode). No return value — purely a state-modifying operation:

```python
def _set_seed(self, unit_idx):
    if self.seed is None:
        return
    seed_val = self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx
    if self.seed_mode == 'legacy':
        np.random.seed(seed_val)
    else:
        self.rng = np.random.default_rng(seed_val)
```

`unit_idx` is 0 for experiment scope, `prob_idx` for series scope, `iter_idx` for
trial scope. `self.rng` is not set in `__init__` — it is a run-time attribute
initialised to `None` at the start of `run()` and updated by `_set_seed` at each
scope boundary. In legacy mode `_set_seed` seeds global state and leaves `self.rng`
as `None`; the loop always passes `self.rng` to `rvs` unconditionally regardless of
mode.

### `run()` loop structure

```python
def run(self, overwrite=False):
    ...
    self.rng = None
    if self.seed_scope == 'experiment':
        self._set_seed(unit_idx=0)

    for prob_idx, problem in enumerate(self.problems):
        for n_idx, n_train in enumerate(self.ns[prob_idx]):

            if self.seed_scope == 'series':
                self._set_seed(prob_idx)

            for iter_idx in range(self.reps):

                if self.seed_scope == 'trial':
                    self._set_seed(iter_idx)

                X_train, X_test, y_train, y_test = problem.rvs(n_train, rng=self.rng)

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
`(reps, n_problems, n_sizes, n_estimators)` — adding the `n_sizes` dimension
(previously hardcoded to 1) to match `Experiment`'s layout.

---

## Relation to Result Persistence

This refactor is a prerequisite for result persistence. It establishes:

- `n_train` as a first-class integer known before the trial loop (required for
  the `<n_train>` path component in cache keys).
- `seed_mode`, `seed_scope`, `seed_progression` on `EmpiricalDataExperiment`
  (required by the persistence spec's seeding section).
- A single unconditional call site for `rvs` inside the rep loop (required
  for the per-trial and per-series cache granularities to map cleanly to the loop
  structure).

---

## Path toward unification with `Experiment`

After this refactor both classes share:

- `ns` as a 2-D int array with the same broadcasting convention.
- `reps` for the number of repetitions.
- Result arrays of shape `(reps, n_problems, n_sizes, n_estimators)`.
- `seed_mode` / `seed_scope` / `seed_progression` parameters.
- `rvs(n, rng)` as the problem interface — both `linear_problem` and
  `EmpiricalDataProblem` now expose this name.

The remaining structural difference is that `Experiment` calls `rvs(n, rng)` twice
per iteration — once for training data, once for a fixed test set — while
`EmpiricalDataExperiment` calls `rvs(n_train, rng)` once and receives a split.
Full unification would require `linear_problem.rvs` to also accept a test split
convention, which is out of scope here. The shared `ns`, `reps`, and seeding
parameters are the necessary foundation for that future step.

---

## Files

- **Modify** `experiments/problems.py`:
  - Rename `EmpiricalDataProblem.get_X_y` to `rvs`; add `n_train`, `rng` params;
    add split and zero-variance filter when `n_train` given; forward `rng` to
    x_transforms.
  - `PolynomialExpansion.__call__` — add `rng=None`; use `rng.choice` when provided.
  - `OneHotEncodeCategories.__call__` — add `rng=None` (ignored).
  - Add `n_train_from_proportion` helper.
  - Update all `get_X_y()` calls in doctests to `rvs()`.

- **Modify** `experiments/experiments.py`:
  - `EmpiricalDataExperiment.__init__` — replace `test_prop` / `n_iterations` with
    `ns` / `reps`; add `seed_mode`, `seed_scope`, `seed_progression`.
  - `EmpiricalDataExperiment.__init__` — initialise `self.rng = None`.
  - `EmpiricalDataExperiment.run` — restructure loop; move zero-variance filter
    into `rvs`; add `_set_seed` class method.
  - Update docstring example (currently calls with `n_iterations=`, `seed=`).

- **Modify** `experiments/real_data.ipynb` — replace `test_prop=0.3` /
  `n_iterations=` with `ns=n_train_from_proportion(problems, 0.7)` / `reps=`;
  replace `get_X_y()` calls with `rvs()`; update imports.
