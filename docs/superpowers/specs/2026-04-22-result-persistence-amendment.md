# Result Persistence Design Amendment

This amendment supersedes Task 4 of the
[result persistence plan](../plans/2026-04-21-result-persistence.md). Tasks 1–3
of that plan are already implemented (Metric base class, FittingTime override,
cache helpers). This document replaces the caching integration design and
introduces structural changes to the runner classes.

---

## Motivation

The original Task 4 design interleaved cache check, retrieve, and store logic
with the existing loop order (trials outer, estimators inner), producing a
deeply nested `run()` method with three separate code paths for the three seed
scopes. The core insight is that moving estimators to the outer loop aligns
cache granularity with loop structure — the cache unit for a given estimator is
determined before any reps are run, and the decision (retrieve or compute) is a
single conditional at the right nesting level. This yields a clean 3-method
decomposition and a `run()` body of around 10 lines.

Further, realisation of different seeding and caching logics in the same class
renders this class very complicated. A clear legacy / maintenance strategy allows
to separate out specific classes per seeding logic. While they have some overlap
and code duplication, this is actually not a maintenance problem, because only
one runner will evolve moving forward (the per trial one). This class separation
can then be exploited to simplify each class and their corresponding persistence
structures.

---

## Simplified Result File Structure

Specific runners allow hard-coding the generator in each of them and therefore
simplifying their respective file structure to remove the 'generator' level.

## Runner Architecture

Two new runner classes replace `EmpiricalDataExperiment` over time. They share
the same constructor signature and result array layout; only the seeding
strategy and cache granularity differ.

### New `Experiment` (per-trial cache)

General-purpose runner for real-data experiments. Uses PCG64 (no `generator`
parameter). Seeds are derived per-trial: trial seed = `seed + rep_idx`.

**Loop structure:**
```
for prob_idx:
  for n_idx:
    for est_idx:
      for rep_idx in range(reps):
        if not (force_recompute or ignore_cache) and _all_stats_in_trial_cache(est_idx, prob_idx, n_idx, rep_idx):
          self._retrieve_trial(est_idx, prob_idx, n_idx, rep_idx)
        else:
          self._run_trial(est_idx, prob_idx, n_idx, rep_idx)
          if not ignore_cache:
              self._write_trial(est_idx, prob_idx, n_idx, rep_idx)
```

**Instance methods (called from `run()`):**
- `_all_stats_in_trial_cache(est_idx, prob_idx, n_idx, rep_idx)` — derives
  trial seed and cache path, returns `True` iff all stats have at least one
  stored computation
- `_retrieve_trial(est_idx, prob_idx, n_idx, rep_idx)` — loads mean value per
  stat, writes to result array, appends retrieval record to each metric file,
  calls `stat.warn_retrieval`
- `_run_trial(est_idx, prob_idx, n_idx, rep_idx)` — seeds RNG from trial seed,
  calls `problem.get_X_y`, fits estimator, writes stat values to result array
- `_write_trial(est_idx, prob_idx, n_idx, rep_idx)` — appends computation entry
  to each metric file, calls `stat.warn_recompute` when prior computations exist

Cache path per trial: `_trial_dir(prob_key, n_train, est_key, trial_seed)`.

### `ExperimentWithPerSeriesSeeding` (per-series cache)

Drop-in replacement for `EmpiricalDataExperiment`. Uses MT19937 (hard-coded)
with fixed seed progression, matching `EmpiricalDataExperiment(generator=
'MT19937', seed_scope='series', seed_progression='fixed')`. The RNG resets to
`self.seed` per `(prob_idx, est_idx)` before the reps loop. No
`seed_progression` parameter.

**Loop structure:**
```
for prob_idx:
  for n_idx:
    for est_idx:
      if not (force_recompute or ignore_cache) and _all_stats_in_series_cache(est_idx, prob_idx, n_idx):
        self._retrieve_series(est_idx, prob_idx, n_idx)
      else:
        self._run_series(est_idx, prob_idx, n_idx)
        if not ignore_cache:
          self._write_series(est_idx, prob_idx, n_idx)
```

**Instance methods:**
- `_all_stats_in_series_cache(est_idx, prob_idx, n_idx)` — derives seed value
  and cache path, returns `True` iff all stats have at least one stored
  computation
- `_retrieve_series(est_idx, prob_idx, n_idx)` — loads per-stat mean arrays of
  length `reps`, writes to result array columns, appends retrieval record,
  calls `stat.warn_retrieval`
- `_run_series(est_idx, prob_idx, n_idx)` — resets RNG, loops over `reps`
  calling `problem.get_X_y` and fitting the estimator each rep, writes stat
  values to result array
- `_write_series(est_idx, prob_idx, n_idx)` — appends the full `reps`-length
  list as one computation entry to each metric file, calls `stat.warn_recompute`
  when prior computations exist

Cache path per series:
`_series_dir(prob_key, n_train, est_key, reps, seed_val)`.

---

## Module-Level Helpers (updated)

`_trial_dir` and `_series_dir` are eliminated entirely. Each runner has a
single `_trial_cache_dir` / `_series_cache_dir` instance method (3 call sites
each) that builds the path inline. One call site each would not justify a named
function.

All other Task 3 helpers (`_cache_key`, `_load_metric_file`, `_save_metric_file`,
`_make_run_id`, `_write_run_file`) are unchanged.

---

## `run()` Signature (both classes)

```python
def run(self, force_recompute=False, ignore_cache=False):
```

Both runners expose `run_id_` (trailing underscore, public, set during `run()`)
so callers can identify which run produced the current results. The three
instance methods access it as `self.run_id_`. Both call `_write_run_file` at
the end unless `ignore_cache=True`.


---

## Legacy `Experiment` → `neurips2023.py`

The existing `Experiment` class (synthetic problems, experiment-scoped RNG,
`keep_fits`, `test_size`) is renamed `SyntheticDataExperiment` and moved to a
new `experiments/neurips2023.py` module to free the name `Experiment` for the
new runner. No behavioral changes. Two CI notebooks that import `Experiment`
from `experiments` are updated to import `SyntheticDataExperiment` from
`neurips2023`.

The long-term goal is that `neurips2023.py` becomes a fully self-contained
frozen module for all NeurIPS 2023 decisions (problem sets, synthetic runner,
etc.), while the evolving modules (`experiments.py`, `problems.py`) have no
dependency on it. Moving remaining elements (`linear_problem`, `default_stats`,
problem sets) is deferred to later steps; transient imports from `experiments`
and `problems` into `neurips2023.py` are acceptable for now.

---

## `EmpiricalDataExperiment` (frozen)

`EmpiricalDataExperiment` is not modified or moved in this spec. It remains in
`experiments.py` as the production runner for legacy work. It will be moved to
`neurips2023.py` in a later step after a dedicated equivalence-testing pass
confirms that `ExperimentWithPerSeriesSeeding` produces numerically identical
results for the NeurIPS 2023 problem sets.

---

## Numerical Equivalence

`ExperimentWithPerSeriesSeeding` is designed to be numerically identical to
`EmpiricalDataExperiment` with `generator='MT19937'` and
`seed_scope='series'` at the same `seed` and `seed_progression`. The loop
reordering (estimators outer rather than inner) is compensated by resetting the
MT19937 RNG to the same seed value before each estimator's reps loop, which
produces the same data-split sequence that the legacy runner produces for that
estimator.

---

## Files

- Create: `experiments/neurips2023.py` — contains `SyntheticDataExperiment`
  (renamed from current `experiments.Experiment`); imports whatever it needs
  from `experiments` and `problems` for now; transient dependencies are
  acceptable
- Modify: `experiments/experiments.py` — remove `Experiment`; delete
  `_trial_dir` and `_series_dir`; add `Experiment` and
  `ExperimentWithPerSeriesSeeding`
- Modify: `experiments/sparse_designs.ipynb` — update import to
  `SyntheticDataExperiment` from `neurips2023`
- Modify: `experiments/double_asymptotic_trends.ipynb` — same
- Modify: `experiments/real_data.ipynb` — migrate to `Experiment`; preview
  cells use `ignore_cache=True`
- Modify: `experiments/real_data_neurips2023.ipynb` — migrate to
  `ExperimentWithPerSeriesSeeding`; preview cells use `ignore_cache=True`
- Modify: `tests/test_experiments.py` — add tests for both new runners

---

## Notebook Migration

`real_data.ipynb` uses `EmpiricalDataExperiment` for iterative real-data
experiments and is migrated to the new `Experiment` class (PCG64, per-trial
caching). `real_data_neurips2023.ipynb` uses `EmpiricalDataExperiment` with
`generator='MT19937'` and `seed_scope='series'` and is migrated to
`ExperimentWithPerSeriesSeeding`, which is numerically identical at the same
seed. Numerical equivalence is confirmed by `test_series_exp_numerical_equivalence`
before the notebook migration is committed. Any cell that runs a quick preview
(not the full experiment) passes `ignore_cache=True` to avoid polluting the
cache with partial results.

---

## Out of Scope

- Moving `EmpiricalDataExperiment` to `neurips2023.py` (separate step)
- Caching for `SyntheticDataExperiment`
- Unifying the `linear_problem.rvs()` and `EmpiricalDataProblem.get_X_y()`
  interfaces
