# Empirical Experiment Seeding Refactor Design

## Goal

This refactoring enables fully reproducible empirical data experiments that are
transparently described to allow the subsequent persistence refactoring (see
[result persistence design](2026-04-16-result-persistence-design.md)) and flexible
enough to eventually unify experiment runners. As secondary effects it modernises
the code by using the modern numpy random number API with explicit generator objects
instead of the legacy procedural API, and it continues enforcing defined
responsibility boundaries by moving concrete data processing out of the experiment
runner.

To achieve this, refactor `EmpiricalDataProblem` and `EmpiricalDataExperiment` to:

1. Thread randomness explicitly by adding an `rng` argument to `get_X_y` that controls
   both stochastic x_transforms (e.g. `PolynomialExpansion` column subsampling) and 
   the train/test split, replacing uncontrolled global numpy state.
2. Support two numpy generators — `'PCG64'` (default, modern) and `'MT19937'`
   (legacy-equivalent); both use explicit generator objects rather than global state.
3. Adding parameters to support different seed scopes (trial, series, experiment) 
   and seed progressions fixed, sequential.
4. Unify the call site of `get_X_y` to be always invoked inside the
   repetition loop, with no pre-computation of data outside the loop by adding
   a mandatory `n_train` parameter to `get_X_y` and change the return type 
   to `(X_train, X_test, y_train, y_test)`. The form of the input parameter aligns
   `EmpiricalDataProblem` with problems used in synthetic runs. The output
   signature deviates from the synthetic experiment but this seems necessary
   to support the coupled train/test set generation typically required in 
   empirical data experiments and can likely also be supported by synthetic
   experiments in the future.
5. Further align `EmpiricalDataExperiment` with `Experiment` by replacing a single
   relative `test_prop` with a flexible integer matrix `ns` (support range of
   train sizes that can differ per problem) and by renaming `n_iterations`
   to `reps`, matching the existing parameter names.
6. Move zero-variance filter from `EmpiricalDataExperiment` to `EmpiricalDataProblem`
   and make it optional.

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
`generator` / `seed_scope` / `seed_progression` parameters on
`EmpiricalDataExperiment`.

---

## `EmpiricalDataProblem.get_X_y`

### New signature

```python
def get_X_y(self, n_train, rng=None):
```

`n_train` is now required. Return type changes from `(X, y)` to
`(X_train, X_test, y_train, y_test)`.

Internal pipeline:

1. Load full dataset via `get_dataset` (already cached at the data layer).
2. Apply column drops, NaN policy, index reset — same as today.
3. Split X / y.
4. Apply y_transforms (deterministic; `rng` not forwarded).
5. Apply each x_transform as `fn(X, rng=rng)` — see x_transform protocol below.
6. Call `sklearn.model_selection.train_test_split(X, y, train_size=n_train,
   random_state=rng)` — `random_state` accepts a `Generator`, `RandomState`, or
   `None`.
7. If `self.zero_variance_filter` is `True`: compute `std = X_train.std();
   non_zero = std[std != 0].index` and restrict both `X_train` and `X_test` to
   `non_zero` columns.

### `zero_variance_filter` parameter

A boolean constructor parameter on `EmpiricalDataProblem`, defaulting to `False`.
When `True`, zero-variance columns (as measured on `X_train`) are dropped from
both `X_train` and `X_test` after splitting. The default is `False` because
filtering makes the effective feature count `p` a function of the split, which
undermines the problem's identity as a stable value object — two calls to
`get_X_y` with different splits may return different column sets. Callers that
previously relied on the filter (applied unconditionally in the experiment runner)
must pass `zero_variance_filter=True` explicitly. The parameter is included in
`__repr__` only when `True`, preserving cache-key stability for the common case.

### rng semantics

The experiment runner always passes a non-None generator object (see `_make_rng`
below). The `rng=None` default exists for external callers: x_transforms substitute
`np.random.default_rng()` when `None` (non-deterministic but still modern API);
`train_test_split` with `random_state=None` uses a random split.

An explicit generator object (either `np.random.Generator` or
`np.random.RandomState`) is consumed sequentially: first by any stochastic
x_transforms in list order, then by `train_test_split`. The caller is responsible
for creating and advancing the generator.

---

## x_transform Protocol Extension

The documented contract for x_transforms changes from `fn(X)` to `fn(X, rng=None)`.

All built-in transforms are updated:

- `PolynomialExpansion.__call__(self, X, rng=None)`: uses `(rng or np.random.default_rng()).choice`
  — when `rng=None`, a fresh unseeded `Generator` is substituted, preserving the modern API
  throughout. The experiment runner always provides an explicit rng.
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

---

## `EmpiricalDataExperiment` Changes

### Constructor

`test_prop` and `n_iterations` are removed. `ns`, `reps`, and `generator` replace
them:

```python
def __init__(self, problems, estimators, reps, ns,
             seed=None,
             generator='PCG64',        # 'PCG64' | 'MT19937'
             seed_scope='series',      # 'series' | 'trial' | 'experiment'
             seed_progression='fixed', # 'fixed' | 'sequential'
             stats=None, est_names=None, verbose=True):
```

`generator` names the numpy BitGenerator to use:
- `'PCG64'` — `np.random.Generator(np.random.PCG64(seed))`. Default.
- `'MT19937'` — `np.random.RandomState(seed)`. Produces numerically identical
  results to the current legacy code for datasets without polynomial subsampling,
  because `np.random.RandomState` is the same engine as `np.random.seed` +
  `np.random.choice`.

The constructor stores a factory function derived from the `generator` parameter:

```python
_RNG_FACTORIES = {
    'PCG64':   lambda seed: np.random.Generator(np.random.PCG64(seed)),
    'MT19937': lambda seed: np.random.RandomState(seed),
}
self._rng_factory = _RNG_FACTORIES[generator]
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

### `_make_rng(self, unit_idx)` instance method

A pure function — no side effects, no global state. Returns a fresh RNG object:

```python
def _make_rng(self, unit_idx):
    seed_val = None if self.seed is None else (
        self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx
    )
    return self._rng_factory(seed_val)
```

`unit_idx` is 0 for experiment scope, `prob_idx` for series scope, `iter_idx` for
trial scope. `seed_val=None` produces an unseeded (non-deterministic) RNG.
`self.rng` is not set in `__init__` — it is a run-time attribute assigned at the
start of `run()`.

### `run()` loop structure

```python
def run(self, overwrite=False):
    ...
    if self.seed_scope == 'experiment':
        self.rng = self._make_rng(unit_idx=0)

    for prob_idx, problem in enumerate(self.problems):
        for n_idx, n_train in enumerate(self.ns[prob_idx]):

            if self.seed_scope == 'series':
                self.rng = self._make_rng(unit_idx=prob_idx)

            for iter_idx in range(self.reps):

                if self.seed_scope == 'trial':
                    self.rng = self._make_rng(unit_idx=iter_idx)

                X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=self.rng)

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

Under `generator='MT19937', seed_scope='series', seed_progression='fixed'`, the
new loop is numerically equivalent to the current implementation for datasets
whose polynomial expansion does not trigger column subsampling (i.e. all but
Twitter at d=2 or d=3). For Twitter, the subsampling previously consumed from an
unseeded global state; it now consumes from the same seeded generator as the
split, which changes the specific columns selected but makes the result
deterministic and reproducible for the first time.

### Result array shape

`self.__dict__[str(stat) + '_']` is allocated with shape
`(reps, n_problems, n_sizes, n_estimators)` — adding the `n_sizes` dimension
(previously hardcoded to 1) to match `Experiment`'s layout.

---

## Relation to Result Persistence

This refactor is a prerequisite for result persistence. It establishes:

- `n_train` as a first-class integer known before the trial loop (required for
  the `<n_train>` path component in cache keys).
- `generator`, `seed_scope`, `seed_progression` on `EmpiricalDataExperiment`
  (required by the persistence spec's seeding section).
- A single unconditional call site for `get_X_y` inside the rep loop (required
  for the per-trial and per-series cache granularities to map cleanly to the loop
  structure).

---

## Alignment with `Experiment`

After this refactor both classes share:

- `ns` as a 2-D int array with the same broadcasting convention.
- `reps` for the number of repetitions.
- Result arrays of shape `(reps, n_problems, n_sizes, n_estimators)`.

`Experiment` carries a single `self.rng` seeded at construction, equivalent to
`seed_scope='experiment'`. `EmpiricalDataExperiment` generalises this with
explicit `seed_scope` and `seed_progression` parameters. Full seeding-parameter
alignment is deferred to a future refactor of `Experiment`.

---

## Files

- **Modify** `experiments/problems.py`:
  - `EmpiricalDataProblem.get_X_y` — add `n_train`, `rng` params; add split;
    add conditional zero-variance filter; forward `rng` to x_transforms. Update doctests.
  - `EmpiricalDataProblem.__init__` — add `zero_variance_filter=False`; include in
    `__repr__` only when `True`.
  - `PolynomialExpansion.__call__` — add `rng=None`; use `(rng or np.random.default_rng()).choice`.
  - `OneHotEncodeCategories.__call__` — add `rng=None` (ignored).
  - Add `n_train_from_proportion` helper.

- **Modify** `experiments/experiments.py`:
  - `EmpiricalDataExperiment.__init__` — replace `test_prop` / `n_iterations` with
    `ns` / `reps`; add `generator`, `seed_scope`, `seed_progression`; add
    `_RNG_FACTORIES` module-level dict; add `_make_rng` instance method.
  - `EmpiricalDataExperiment.run` — restructure loop; remove zero-variance filter
    (now in `get_X_y`). Update docstring example.

- **Modify** `experiments/real_data.ipynb` — replace `test_prop=0.3` /
  `n_iterations=` with `ns=n_train_from_proportion(problems, 0.7)` / `reps=`;
  update imports.
