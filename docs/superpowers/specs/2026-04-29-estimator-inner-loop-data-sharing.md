# Estimator Inner-Loop Data Sharing Design

## Motivation

Both experiment runners (`Experiment` and `ExperimentWithPerSeriesSeeding`) currently call
`problem.get_X_y` once per (n_idx, est_idx, rep_idx) combination. For empirical data problems
this means loading and splitting the full dataset k times per repetition, where k is the number
of estimators. On large datasets (ct_slices: 53,500 rows × 385 cols; twitter: 583,250 rows × 77
cols) this dominates experiment wall time.

The root cause is the loop order: the estimator loop is *outside* the repetition loop. If
estimators are moved innermost, each repetition materialises the dataset once and all estimators
share it.

For `ExperimentWithPerSeriesSeeding`, the problem is compounded: `_run_series` is called
separately for each estimator, and each call resets `RandomState(seed)` and replays all reps
from scratch, producing identical data k times rather than once.

## Design

### `Experiment` — lazy data cache

**Loop order change** (inside the `prob_idx` block):

Current order: `n_idx → est_idx → rep_idx`
New order:     `n_idx → rep_idx → est_idx`

This ensures successive iterations with the same `(n_idx, rep_idx)` differ only in `est_idx`,
making a one-slot cache valid across all estimators within a rep.

**Cache fields** (instance, reset per `prob_idx`):

```python
self._cached_data_key = None   # (n_idx, rep_idx) or None
self._cached_data = None       # (X_train, X_test, y_train, y_test)
```

**Modified `_run_trial`** — conditional `get_X_y` replaces the unconditional call at line 420:

```python
def _run_trial(self, prob_idx, n_idx, est_idx, rep_idx):
    problem = self.problems[prob_idx]
    n_train = int(self.ns[prob_idx][n_idx])
    data_key = (n_idx, rep_idx)
    if data_key != self._cached_data_key:
        rng = np.random.Generator(np.random.PCG64(self.seed + rep_idx))
        self._cached_data = problem.get_X_y(n_train, rng=rng)
        self._cached_data_key = data_key
    X_train, X_test, y_train, y_test = self._cached_data
    _est = clone(self.estimators[est_idx], safe=False)
    try:
        t0 = time.time()
        _est.fit(X_train, y_train)
        _est.fitting_time_ = time.time() - t0
    except Exception as e:
        warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                      f" on '{problem.dataset}': {e}")
        return
    for stat in self.stats:
        val = stat(_est, problem, X_test, y_test)
        self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
```

**Modified generator in `run()`** (inside the `prob_idx` block):

```python
self._cached_data_key = None
# ...
trials = ((n_idx, rep_idx, est_idx)
          for n_idx in range(n_sizes)
          for rep_idx in range(self.reps)
          for est_idx in range(n_estimators))
for n_idx, rep_idx, est_idx in tqdm(trials, ...):
    if ... cache hit ...:
        self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
        ...
    else:
        ...
        self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
        ...
```

**Correctness**: per-rep seeding is `PCG64(self.seed + rep_idx)`, which is independent of
`est_idx`. The same RNG state therefore produces the same data regardless of loop order.
Cache invalidation is guaranteed because `n_idx` is outermost: all `(n_idx, rep_idx)` pairs
for the same `n_idx` are exhausted before `n_idx` advances, so the cache never needs to hold
more than one dataset.

**Behaviour change**: none — trial data is identical to the current implementation.

---

### `ExperimentWithPerSeriesSeeding` — `_run_series` generalised to a list of estimators

The per-series seeding contract (each (problem, n, estimator, seed) cache entry is reproducible
in isolation) is preserved: the series cache key does not change, and the same `RandomState(seed)`
replay produces identical data sequences. The change is that `_run_series` now accepts a list
of estimator indices and loops over them inside the rep loop, so the RNG is advanced once and
shared rather than replayed per estimator.

**Modified `_run_series`** — `est_idx: int` replaced by `est_indices: list[int]`; optional
`pbar` (a tqdm object) updated once per fit to expose rep-level progress to the caller:

```python
def _run_series(self, prob_idx, n_idx, est_indices, pbar=None):
    problem = self.problems[prob_idx]
    n_train = int(self.ns[prob_idx][n_idx])
    rng = np.random.RandomState(self.seed)
    for rep_idx in range(self.reps):
        X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
        for est_idx in est_indices:
            _est = clone(self.estimators[est_idx], safe=False)
            try:
                t0 = time.time()
                _est.fit(X_train, y_train)
                _est.fitting_time_ = time.time() - t0
            except Exception as e:
                warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                              f" on '{problem.dataset}': {e}")
            else:
                for stat in self.stats:
                    val = stat(_est, problem, X_test, y_test)
                    self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
            if pbar is not None:
                pbar.update(1)
```

**Modified `run()` inner body** (inside the `prob_idx` block):

```python
n_trials = n_sizes * n_estimators * self.reps
pbar = tqdm(total=n_trials, desc=dataset, position=1, leave=False)
for n_idx in range(n_sizes):
    n_train = int(self.ns[prob_idx][n_idx])
    ests_to_compute = []
    for est_idx in range(n_estimators):
        if (not force_recompute and not overwrite_cache and not ignore_cache
                and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
            self._retrieve_series(prob_idx, n_idx, est_idx)
            self.trials_retrieved_ += self.reps
            pbar.update(self.reps)
        else:
            if overwrite_cache and not ignore_cache:
                shutil.rmtree(
                    self._series_cache_dir(prob_idx, n_idx, est_idx),
                    ignore_errors=True)
            ests_to_compute.append(est_idx)
    self._run_series(prob_idx, n_idx, ests_to_compute, pbar=pbar)
    for est_idx in ests_to_compute:
        if not ignore_cache:
            self._write_series(prob_idx, n_idx, est_idx)
        self.trials_computed_ += self.reps
pbar.close()
```

The pbar is created before the n_idx loop and closed after, so both retrieved and computed
trials contribute to the same bar, with total `n_sizes * n_estimators * self.reps`. Retrieved
series advance it by `self.reps` in bulk; computed trials advance it by 1 per fit (inside
`_run_series`). This mirrors `Experiment.run()` where the tqdm covers all
(n_idx, est_idx, rep_idx) triples with the same total. The outer tqdm (over problems) and
per-problem timing/logging are unchanged.

**Correctness**: `rng = np.random.RandomState(self.seed)` is reset once at the start of
`_run_series`, shared across all `est_indices`. Each estimator receives the same
`(X_train, X_test, y_train, y_test)` sequence as it would from a separate single-estimator
`_run_series` call, because those calls each reset to the same seed and advance identically.

**Cache compatibility**: existing cached series remain valid. The cache key (problem, n_train,
estimator, reps, seed) is unchanged. A partially-cached run (some estimators cached, some not)
is handled correctly: only `new_ests` is passed to `_run_series`.

**Behaviour change**: none for fully-uncached or fully-cached runs. The trial count
(`trials_computed_`, `trials_retrieved_`) and warning logic are identical.
