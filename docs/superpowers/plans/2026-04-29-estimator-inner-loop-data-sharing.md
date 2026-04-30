# Estimator Inner-Loop Data Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `get_X_y` calls from once per (n_idx, est_idx, rep_idx) to once per
(n_idx, rep_idx) in both experiment runners, eliminating redundant dataset materialisation.

**Architecture:** `Experiment` gains a one-slot data cache keyed on `(n_idx, rep_idx)` and
reorders its trial generator. `ExperimentWithPerSeriesSeeding._run_series` is generalised from
a single `est_idx` to a list `est_indices`, and `run()` is restructured to pass all estimators
needing computation in one call.

**Tech Stack:** Python, NumPy, tqdm, scikit-learn clone

**Spec:** `docs/superpowers/specs/2026-04-29-estimator-inner-loop-data-sharing.md`

---

### Task 1: `Experiment` — lazy data cache + generator reorder

**Files:**
- Modify: `experiments/experiments.py` (lines 416–500)
- Test: `tests/test_experiments.py`

---

- [ ] **Step 1: Write failing test — `get_X_y` called once per rep**

Add to `tests/test_experiments.py`:

```python
def test_new_experiment_get_x_y_called_once_per_rep(monkeypatch):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    call_count = {'n': 0}
    original = prob.get_X_y
    def counting_get_X_y(*args, **kwargs):
        call_count['n'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(prob, 'get_X_y', counting_get_X_y)
    exp = Experiment([prob], [RidgeEM(), RidgeEM()], reps=3, ns=ns,
                     seed=1, verbose=False)
    exp.run(ignore_cache=True)
    # 1 n_size * 3 reps — shared across both estimators
    assert call_count['n'] == 3
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_experiments.py::test_new_experiment_get_x_y_called_once_per_rep -v
```

Expected: FAIL — currently `call_count['n']` is 6 (2 estimators × 3 reps).

- [ ] **Step 3: Modify `_run_trial` to use a lazy data cache**

Replace `_run_trial` (lines 416–432 of `experiments/experiments.py`) with:

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

- [ ] **Step 4: Reset cache fields and reorder the trial generator in `run()`**

Inside the `for prob_idx in tqdm(...)` block, immediately before the `trials` generator
(around line 479), add `self._cached_data_key = None` and change the generator order from
`n_idx → est_idx → rep_idx` to `n_idx → rep_idx → est_idx`:

```python
                self._cached_data_key = None
                self._cached_data = None
                trials = ((n_idx, rep_idx, est_idx)
                          for n_idx in range(n_sizes)
                          for rep_idx in range(self.reps)
                          for est_idx in range(n_estimators))
                for n_idx, rep_idx, est_idx in tqdm(trials, total=n_trials,
                                                     desc=dataset, position=1, leave=False):
```

- [ ] **Step 5: Run all tests**

```
pytest
```

Expected: all pass, including `test_new_experiment_get_x_y_called_once_per_rep`.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "perf: share get_X_y data across estimators in Experiment via lazy cache"
```

---

### Task 2: `ExperimentWithPerSeriesSeeding` — generalise `_run_series` + restructure `run()`

**Files:**
- Modify: `experiments/neurips2023.py` (lines 251–342)
- Create: `tests/test_neurips2023.py`

---

- [ ] **Step 1: Count tests before the move**

```
pytest --collect-only -q 2>/dev/null | tail -1
```

Note the total number of tests collected.

- [ ] **Step 2: Create `tests/test_neurips2023.py` and move series tests**

Create `tests/test_neurips2023.py` with the following imports and move these items from
`tests/test_experiments.py` into it (delete them from `test_experiments.py`):
- `_simple_series_exp` helper (lines 194–199)
- `test_series_exp_result_shape` (lines 202–203)
- `test_series_exp_cache_hit` (lines 206–211)
- `test_series_exp_ignore_cache` (lines 214–217)
- `test_series_exp_reproducible` (lines 220–223)
- `test_series_exp_overwrite_cache` (lines 241–253)
- `test_series_exp_log_file_written` (lines 264–269)

```python
import os
import numpy as np
import pytest
import neurips2023
from fastridge import RidgeEM
from problems import EmpiricalDataProblem
from problems import n_train_from_proportion
from neurips2023 import ExperimentWithPerSeriesSeeding
```

- [ ] **Step 3: Confirm test count is unchanged**

```
pytest --collect-only -q 2>/dev/null | tail -1
```

Expected: same total as before. If the count differs, a test was lost or duplicated — fix before proceeding.

- [ ] **Step 4: Run all tests**

```
pytest
```

Expected: all pass.

- [ ] **Step 5: Write failing test — `get_X_y` called once per rep in series runner**

Add to `tests/test_neurips2023.py`:

```python
def test_series_exp_get_x_y_called_once_per_rep(monkeypatch):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    call_count = {'n': 0}
    original = prob.get_X_y
    def counting_get_X_y(*args, **kwargs):
        call_count['n'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(prob, 'get_X_y', counting_get_X_y)
    exp = ExperimentWithPerSeriesSeeding(
        [prob], [RidgeEM(), RidgeEM()], reps=3, ns=ns, seed=1, verbose=False)
    exp.run(ignore_cache=True)
    # 1 n_size * 3 reps — shared across both estimators
    assert call_count['n'] == 3
```

- [ ] **Step 6: Run test to confirm it fails**

```
pytest tests/test_neurips2023.py::test_series_exp_get_x_y_called_once_per_rep -v
```

Expected: FAIL — currently `call_count['n']` is 6 (2 separate `_run_series` calls, each replaying 3 reps).

- [ ] **Step 7: Generalise `_run_series` to accept a list of estimator indices**

Replace `_run_series` (lines 251–268 of `experiments/neurips2023.py`) with:

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

- [ ] **Step 8: Restructure the inner body of `run()`**

Replace the block from the `series = (...)` generator through the end of the `for n_idx, est_idx` loop (lines 315–332 of `experiments/neurips2023.py`) with:

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

- [ ] **Step 9: Run all tests**

```
pytest
```

Expected: all pass, including `test_series_exp_get_x_y_called_once_per_rep` and the existing
reproducibility test `test_series_exp_reproducible`.

- [ ] **Step 10: Commit**

```bash
git add experiments/neurips2023.py tests/test_experiments.py tests/test_neurips2023.py
git commit -m "perf: share get_X_y data across estimators in ExperimentWithPerSeriesSeeding"
```
