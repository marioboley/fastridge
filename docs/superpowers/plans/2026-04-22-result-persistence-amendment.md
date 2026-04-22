# Result Persistence Amendment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale Task 4 caching design with two clean runner classes (`Experiment` and `ExperimentWithPerSeriesSeeding`) that have a ~10-line `run()` and a clear 3-method cache decomposition, and move the legacy `Experiment` (synthetic) to `neurips2023.py`.

**Architecture:** Tasks 1–3 of the original plan are already implemented (Metric base class, FittingTime override, cache helpers). This plan covers the remaining work: deleting stale helpers, renaming the legacy runner, creating two new runner classes, migrating real-data notebooks, and adding `results/.gitignore`. Each runner hard-codes its generator (PCG64 for `Experiment`, MT19937 for `ExperimentWithPerSeriesSeeding`) and owns its cache-check/retrieve/compute/write methods. The neurips notebook migration (Task 7) has no CI coverage and requires user validation.

**Tech Stack:** Python, numpy, scikit-learn (clone), joblib (hash), pytest, nbmake

---

## File Map

**Modified:**
- `experiments/experiments.py` — delete `_trial_dir`/`_series_dir`; remove old `Experiment`; add new `Experiment` and `ExperimentWithPerSeriesSeeding`
- `tests/test_experiments.py` — remove stale Task 4 tests; add tests for new runners
- `experiments/sparse_designs.ipynb` — update import from `Experiment` to `SyntheticDataExperiment`
- `experiments/double_asymptotic_trends.ipynb` — same
- `experiments/real_data.ipynb` — migrate to `Experiment`; preview cells use `ignore_cache=True` (Task 6, CI-verified)
- `experiments/real_data_neurips2023.ipynb` — migrate to `ExperimentWithPerSeriesSeeding` (Task 7, user-validated only)

**Created:**
- `experiments/neurips2023.py` — contains `SyntheticDataExperiment` (renamed from `Experiment`)
- `results/.gitignore`

---

### Task 1: Drop generator from helper signatures and remove stale tests

**Files:**
- Modify: `experiments/experiments.py:27-34`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Delete `_series_dir` and `_trial_dir` from `experiments/experiments.py`**

Remove both functions entirely (lines 27–34). Their path logic will be inlined
in `_series_cache_dir` / `_trial_cache_dir` instance methods in Tasks 3 and 4.

- [ ] **Step 2: Remove stale Task 4 tests from `tests/test_experiments.py`**

Remove these functions (they test `EmpiricalDataExperiment` with `ignore_cache`/`force_recompute` that will never be implemented on that class):
- `_simple_trial_exp`
- `test_series_cache_hit_reuses_result`
- `test_series_ignore_cache_writes_no_files`
- `test_series_force_recompute_appends_computation`
- `test_run_file_written`
- `test_trial_cache_hit_reuses_result`
- `test_trial_ignore_cache_writes_no_files`

- [ ] **Step 3: Run tests to verify nothing broken**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all tests PASS (count drops because stale tests are removed)

- [ ] **Step 4: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "refactor: drop generator param from _trial_dir/_series_dir; remove stale tests"
```

---

### Task 2: Create `neurips2023.py`, remove old `Experiment`, update CI notebooks

**Files:**
- Create: `experiments/neurips2023.py`
- Modify: `experiments/experiments.py`
- Modify: `tests/test_experiments.py`
- Modify: `experiments/sparse_designs.ipynb`
- Modify: `experiments/double_asymptotic_trends.ipynb`

- [ ] **Step 1: Write a failing test**

Add to `tests/test_experiments.py` (after the existing imports):

```python
from neurips2023 import SyntheticDataExperiment


def test_synthetic_experiment_importable():
    assert SyntheticDataExperiment is not None


def test_experiment_not_in_experiments():
    assert not hasattr(experiments, 'Experiment')
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_synthetic_experiment_importable -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'neurips2023'`

- [ ] **Step 3: Create `experiments/neurips2023.py`**

```python
import time
import numpy as np
from sklearn.base import clone
from fastprogress.fastprogress import progress_bar
from experiments import default_stats


class SyntheticDataExperiment:

    def __init__(self, problems, estimators, ns, reps, est_names=None, stats=default_stats,
                 keep_fits=True, verbose=0, seed=None):
        self.problems = problems
        self.estimators = estimators
        self.ns = np.atleast_2d(ns)
        self.ns = (self.ns if len(self.ns) == len(self.problems)
                   else self.ns.repeat(len(self.problems), axis=0))
        self.reps = reps
        self.verbose = verbose
        self.est_names = [str(est) for est in estimators] if est_names is None else est_names
        self.stats = stats
        self.keep_fits = keep_fits
        self.test_size = 10000
        self.rng = np.random.default_rng(seed)

    def run(self):
        if self.keep_fits:
            self.fits = {}
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.zeros(
                shape=(self.reps, len(self.problems), len(self.ns[0]), len(self.estimators)))
        for r in progress_bar(range(self.reps)):
            for i in range(len(self.problems)):
                x_test, y_test = self.problems[i].rvs(self.test_size, rng=self.rng)
                for n_idx, n in enumerate(self.ns[i]):
                    for j, est in enumerate(self.estimators):
                        x, y = self.problems[i].rvs(n, rng=self.rng)
                        _est = clone(est, safe=False)
                        fit_start_time = time.time()
                        _est.fit(x, y)
                        _est.fitting_time_ = time.time() - fit_start_time
                        if self.keep_fits:
                            self.fits[(r, i, n, j)] = _est
                        for stat in self.stats:
                            self.__dict__[str(stat) + '_'][r, i, n_idx, j] = stat(
                                _est, self.problems[i], x_test, y_test)
        return self
```

- [ ] **Step 4: Remove `Experiment` class from `experiments/experiments.py`**

Delete the entire block from `class Experiment:` through `return self` (the `run()` method ending before `class RidgePathExperiment:`).

- [ ] **Step 5: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS including `test_synthetic_experiment_importable` and `test_experiment_not_in_experiments`

- [ ] **Step 6: Update `experiments/sparse_designs.ipynb`**

Read the notebook:
```
Read("experiments/sparse_designs.ipynb")
```
Find the cell containing `from experiments import Experiment`. Using `NotebookEdit` with that cell's `cell_id`, replace the cell source so that:
- `from neurips2023 import SyntheticDataExperiment` is added (or replaces `from experiments import Experiment, ...`)
- All remaining imports from `experiments` (`parameter_mean_squared_error`, etc.) stay in place

For example, if the cell currently reads:
```python
from experiments import Experiment, parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time
```
replace with:
```python
from neurips2023 import SyntheticDataExperiment
from experiments import parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time
```

Then find every use of `Experiment(` in the notebook and replace with `SyntheticDataExperiment(`.

- [ ] **Step 7: Update `experiments/double_asymptotic_trends.ipynb`**

Same process as Step 6 for `experiments/double_asymptotic_trends.ipynb`.

- [ ] **Step 8: Verify notebooks execute**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --nbmake experiments/sparse_designs.ipynb experiments/double_asymptotic_trends.ipynb -v
```

Expected: PASS (cells tagged `skip-execution` are skipped automatically by nbmake)

- [ ] **Step 9: Commit**

```bash
git add experiments/neurips2023.py experiments/experiments.py tests/test_experiments.py \
        experiments/sparse_designs.ipynb experiments/double_asymptotic_trends.ipynb
git commit -m "feat: move Experiment to neurips2023.py as SyntheticDataExperiment"
```

---

### Task 3: New `Experiment` class (per-trial PCG64 caching)

**Files:**
- Modify: `experiments/experiments.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py` (add `Experiment` to the existing `from experiments import` line):

```python
from experiments import Experiment


def _simple_new_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, verbose=False)
    defaults.update(kwargs)
    return Experiment([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_new_experiment_result_shape():
    assert _simple_new_exp().run(ignore_cache=True).prediction_r2_.shape == (2, 1, 1, 1)


def test_new_experiment_trial_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_new_exp().run()
    exp2 = _simple_new_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_new_experiment_ignore_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'trial'))


def test_new_experiment_force_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    _simple_new_exp().run(force_recompute=True)
    trial_dir = os.path.join(str(tmp_path), 'trial')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(trial_dir)
                  for f in fs if f.endswith('.json')]
    assert json_files
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 2


def test_new_experiment_run_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    assert len(os.listdir(os.path.join(str(tmp_path), 'runs'))) == 1
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_new_experiment_result_shape -v
```

Expected: FAIL with `ImportError: cannot import name 'Experiment'`

- [ ] **Step 3: Add `Experiment` class to `experiments/experiments.py`**

Insert after `empirical_default_stats = [...]` and before `class EmpiricalDataExperiment:`:

```python
class Experiment:
    """Run repeated train/test experiments on EmpiricalDataProblem instances.

    Uses per-trial PCG64 seeding: trial seed = seed + rep_idx. Results are
    cached per (problem, n_train, estimator, trial_seed).

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like of shape (n_problems, n_sizes)
        A 1-D input is broadcast to all problems.
    seed : int, default 0
    stats : list of Metric or None
    est_names : list of str or None
    verbose : bool, default True
    """

    def __init__(self, problems, estimators, reps, ns,
                 seed=0, stats=None, est_names=None, verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.reps = reps
        self.ns = np.atleast_2d(ns)
        if len(self.ns) != len(self.problems):
            self.ns = self.ns.repeat(len(self.problems), axis=0)
        self.seed = seed
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def _trial_cache_dir(self, prob_idx, n_idx, est_idx, rep_idx):
        return os.path.join(
            CACHE_DIR, 'trial',
            _cache_key(self.problems[prob_idx]),
            str(int(self.ns[prob_idx][n_idx])),
            _cache_key(self.estimators[est_idx]),
            str(self.seed + rep_idx),
        )

    def _all_stats_in_trial_cache(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        return all(
            _load_metric_file(os.path.join(d, str(stat) + '.json'))['computations']
            for stat in self.stats
        )

    def _retrieve_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            mean_val = float(np.mean([c['value'] for c in data['computations']]))
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = mean_val
            data['retrievals'].append({'value': mean_val, 'run_id': self.run_id_})
            _save_metric_file(path, data)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(msg)

    def _run_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.Generator(np.random.PCG64(self.seed + rep_idx))
        X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
        _est = clone(self.estimators[est_idx], safe=False)
        try:
            t0 = time.time()
            _est.fit(X_train, y_train)
            _est.fitting_time_ = time.time() - t0
        except Exception as e:
            warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                          f" on '{problem.dataset}': {e}")
            self._last_trial_values = None
            return
        self._last_trial_values = {}
        for stat in self.stats:
            val = stat(_est, problem, X_test, y_test)
            self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
            self._last_trial_values[str(stat)] = (
                val.tolist() if hasattr(val, 'tolist') else float(val))

    def _write_trial(self, prob_idx, n_idx, est_idx, rep_idx):
        if self._last_trial_values is None:
            return
        d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            new_val = self._last_trial_values[str(stat)]
            if data['computations']:
                msg = stat.warn_recompute(
                    [c['value'] for c in data['computations']], new_val)
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': new_val, 'run_id': self.run_id_})
            _save_metric_file(path, data)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = _make_run_id()
        trials_computed = trials_retrieved = 0
        for prob_idx in range(n_problems):
            if self.verbose:
                print(self.problems[prob_idx].dataset, end=' ')
            for n_idx in range(n_sizes):
                for est_idx in range(n_estimators):
                    for rep_idx in range(self.reps):
                        if (not force_recompute and not ignore_cache
                                and self._all_stats_in_trial_cache(
                                    prob_idx, n_idx, est_idx, rep_idx)):
                            self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
                            trials_retrieved += 1
                        else:
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            trials_computed += 1
            if self.verbose:
                print()
        if not ignore_cache:
            _write_run_file(self.run_id_, {
                'problems': [_cache_key(p) for p in self.problems],
                'estimators': [_cache_key(e) for e in self.estimators],
                'ns': self.ns.tolist(),
                'reps': self.reps,
                'seed': self.seed,
            }, {'trials_computed': trials_computed, 'trials_retrieved': trials_retrieved})
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add Experiment class with per-trial PCG64 seeding and caching"
```

---

### Task 4: `ExperimentWithPerSeriesSeeding` (per-series MT19937 caching)

**Files:**
- Modify: `experiments/experiments.py`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add `ExperimentWithPerSeriesSeeding` to the existing `from experiments import` line, then add:

```python
def _simple_series_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, verbose=False)
    defaults.update(kwargs)
    return ExperimentWithPerSeriesSeeding([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_series_exp_result_shape():
    assert _simple_series_exp().run(ignore_cache=True).prediction_r2_.shape == (2, 1, 1, 1)


def test_series_exp_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_series_exp().run()
    exp2 = _simple_series_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_series_exp_ignore_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'series'))


def test_series_exp_numerical_equivalence():
    # ExperimentWithPerSeriesSeeding must reproduce EmpiricalDataExperiment
    # with generator='MT19937', seed_scope='series', seed_progression='fixed' exactly.
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    legacy = EmpiricalDataExperiment(
        [prob], [RidgeEM()], reps=2, ns=ns,
        seed=1, generator='MT19937', seed_scope='series',
        seed_progression='fixed', verbose=False)
    new = ExperimentWithPerSeriesSeeding(
        [prob], [RidgeEM()], reps=2, ns=ns,
        seed=1, verbose=False)
    legacy.run()
    new.run(ignore_cache=True)
    np.testing.assert_array_equal(legacy.prediction_r2_, new.prediction_r2_)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_series_exp_result_shape -v
```

Expected: FAIL with `ImportError: cannot import name 'ExperimentWithPerSeriesSeeding'`

- [ ] **Step 3: Add `ExperimentWithPerSeriesSeeding` to `experiments/experiments.py`**

Insert immediately after the `Experiment` class (before `class EmpiricalDataExperiment:`):

```python
class ExperimentWithPerSeriesSeeding:
    """Run repeated train/test experiments with per-series MT19937 seeding.

    Numerically identical to EmpiricalDataExperiment(generator='MT19937',
    seed_scope='series', seed_progression='fixed') at the same seed. Results
    are cached per (problem, n_train, estimator, reps, seed).

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like of shape (n_problems, n_sizes)
        A 1-D input is broadcast to all problems.
    seed : int or None
    stats : list of Metric or None
    est_names : list of str or None
    verbose : bool, default True
    """

    def __init__(self, problems, estimators, reps, ns,
                 seed=None, stats=None, est_names=None, verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.reps = reps
        self.ns = np.atleast_2d(ns)
        if len(self.ns) != len(self.problems):
            self.ns = self.ns.repeat(len(self.problems), axis=0)
        self.seed = seed
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def _series_cache_dir(self, prob_idx, n_idx, est_idx):
        return os.path.join(
            CACHE_DIR, 'series',
            _cache_key(self.problems[prob_idx]),
            str(int(self.ns[prob_idx][n_idx])),
            _cache_key(self.estimators[est_idx]),
            str(self.reps),
            str(self.seed),
        )

    def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        return all(
            _load_metric_file(os.path.join(d, str(stat) + '.json'))['computations']
            for stat in self.stats
        )

    def _retrieve_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            values = [np.asarray(c['value']) for c in data['computations']]
            mean_val = np.mean(values, axis=0)
            self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
            serialisable = mean_val.tolist() if hasattr(mean_val, 'tolist') else float(mean_val)
            data['retrievals'].append({'value': serialisable, 'run_id': self.run_id_})
            _save_metric_file(path, data)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(msg)

    def _run_series(self, prob_idx, n_idx, est_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.RandomState(self.seed)
        self._last_series_values = {str(stat): [] for stat in self.stats}
        for rep_idx in range(self.reps):
            X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
            _est = clone(self.estimators[est_idx], safe=False)
            try:
                t0 = time.time()
                _est.fit(X_train, y_train)
                _est.fitting_time_ = time.time() - t0
            except Exception as e:
                warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                              f" on '{problem.dataset}': {e}")
                for stat in self.stats:
                    self._last_series_values[str(stat)].append(float('nan'))
                continue
            for stat in self.stats:
                val = stat(_est, problem, X_test, y_test)
                self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
                self._last_series_values[str(stat)].append(
                    val.tolist() if hasattr(val, 'tolist') else float(val))

    def _write_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = _load_metric_file(path)
            new_values = self._last_series_values[str(stat)]
            if data['computations']:
                existing = [np.asarray(c['value']) for c in data['computations']]
                msg = stat.warn_recompute(existing, np.asarray(new_values))
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': new_values, 'run_id': self.run_id_})
            _save_metric_file(path, data)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = _make_run_id()
        trials_computed = trials_retrieved = 0
        for prob_idx in range(n_problems):
            if self.verbose:
                print(self.problems[prob_idx].dataset, end=' ')
            for n_idx in range(n_sizes):
                for est_idx in range(n_estimators):
                    if (not force_recompute and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        trials_retrieved += self.reps
                    else:
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        trials_computed += self.reps
            if self.verbose:
                print()
        if not ignore_cache:
            _write_run_file(self.run_id_, {
                'problems': [_cache_key(p) for p in self.problems],
                'estimators': [_cache_key(e) for e in self.estimators],
                'ns': self.ns.tolist(),
                'reps': self.reps,
                'seed': self.seed,
            }, {'trials_computed': trials_computed, 'trials_retrieved': trials_retrieved})
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS including `test_series_exp_numerical_equivalence`

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add ExperimentWithPerSeriesSeeding with MT19937 and series-level caching"
```

---

### Task 5: `results/.gitignore`

**Files:**
- Create: `results/.gitignore`

- [ ] **Step 1: Create directory and gitignore**

```bash
mkdir -p /Users/marioboley/Documents/GitHub/fastridge/results
```

Create `results/.gitignore` with contents:
```
*
!.gitignore
```

- [ ] **Step 2: Commit**

```bash
git add results/.gitignore
git commit -m "chore: add results/.gitignore to track directory without caching content"
```

---

### Task 6: Migrate real-data notebooks to new runners

**Files:**
- Modify: `experiments/real_data.ipynb`
- Modify: `experiments/real_data_neurips2023.ipynb`

**Note:** Migrating `real_data.ipynb` from `EmpiricalDataExperiment(generator='MT19937')` to
`Experiment` (PCG64) is an intentional numerical change — the evolving notebook is not
bound to MT19937 reproducibility. `real_data_neurips2023.ipynb` migrates to
`ExperimentWithPerSeriesSeeding`, which is numerically identical (confirmed by
`test_series_exp_numerical_equivalence` in Task 4).

- [ ] **Step 1: Update `real_data.ipynb` — import cell**

Read `experiments/real_data.ipynb` to verify cell IDs, then use `NotebookEdit` on
cell `76d89b51`. Replace the entire cell source with:

```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import Experiment
from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion, NEURIPS2023_TRAIN_SIZES

_OHE = (OneHotEncodeCategories(),)

problems = [
    EmpiricalDataProblem('abalone',    'Rings',                         x_transforms=_OHE, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',                            zero_variance_filter=True),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength',                    zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',   'target',                                           zero_variance_filter=True),
    EmpiricalDataProblem('eye',        'y',                                                zero_variance_filter=True),
    EmpiricalDataProblem('forest',     'area',                          x_transforms=_OHE, zero_variance_filter=True),
    EmpiricalDataProblem('student',    'G3', drop=('G1', 'G2'),         x_transforms=_OHE, zero_variance_filter=True),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',                             zero_variance_filter=True),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows', x_transforms=_OHE, zero_variance_filter=True),
]

estimators = {
    'EM':     RidgeEM(),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
    'CV_glm': RidgeLOOCV(alphas=100),
}

exp = Experiment(
    problems, list(estimators.values()),
    reps=10, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems],
    seed=123,
    est_names=list(estimators.keys())).run(ignore_cache=True)
print()
```

- [ ] **Step 2: Update `real_data.ipynb` — preview cells 5 and 8**

Cell `f6bc8769` — replace source:
```python
exp_d2 = Experiment(
    problems_d2, list(estimators.values()),
    reps=10, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d2],
    seed=123,
    est_names=list(estimators.keys())).run(ignore_cache=True)
print()
```

Cell `094956a8` — replace source:
```python
exp_d3 = Experiment(
    problems_d3, list(estimators.values()),
    reps=10, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d3],
    seed=123,
    est_names=list(estimators.keys())).run(ignore_cache=True)
print()
```

- [ ] **Step 3: Update `real_data.ipynb` — skip-execution cells**

These cells are tagged `skip-execution` and are not run in CI. For each one,
replace `EmpiricalDataExperiment(` with `Experiment(` and remove
`generator='MT19937',`. Do NOT add `ignore_cache=True` — these are full
experiment cells that should use the cache.

Cell `2bfbd407` (exp_full) — the combined problem-definition-and-run cell.
Replace the `EmpiricalDataExperiment(` call block at the end:
```python
exp_full = Experiment(
    problems_full, list(estimators_full.values()),
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_full],
    seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `28f3974b`:
```python
exp_full_d2 = Experiment(
    problems_full_d2, list(estimators_full.values()),
    reps=30, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_full_d2],
    seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `a96f98ee`:
```python
exp_full_d3 = Experiment(
    problems_full_d3, list(estimators_full.values()),
    reps=30, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_full_d3],
    seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `68c60ed3`:
```python
exp_large = Experiment(
    problems_large, list(estimators_full.values()),
    reps=30, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_large],
    seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Cell `300ca33b`:
```python
exp_large_d2 = Experiment(
    problems_large_d2, list(estimators_full.values()),
    reps=30, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_large_d2],
    seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

- [ ] **Step 4: Update `real_data_neurips2023.ipynb` — all three cells**

Read `experiments/real_data_neurips2023.ipynb` to verify cell IDs. All three
experiment cells are tagged `skip-execution`. Replace `EmpiricalDataExperiment`
with `ExperimentWithPerSeriesSeeding` and remove `generator='MT19937',`. Update
the import in cell `a003` from `from experiments import EmpiricalDataExperiment`
to `from experiments import ExperimentWithPerSeriesSeeding`.

Cell `a003` — full replacement:
```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from experiments import ExperimentWithPerSeriesSeeding
from problems import NEURIPS2023, NEURIPS2023_TRAIN_SIZES
from data import DATASETS

# estimator indices: 0=EM, 1=CV_fix, 2=CV_glm
estimators = [
    RidgeEM(t2=False),
    RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
    RidgeLOOCV(alphas=100),
]
est_names = ['EM', 'CV_fix', 'CV_glm']

problems_d1 = sorted(NEURIPS2023, key=lambda p: DATASETS[p.dataset]['n'])
exp_d1 = ExperimentWithPerSeriesSeeding(
    problems_d1, estimators,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d1],
    seed=123,
    est_names=est_names, verbose=True).run()
print()
```

Cell `a006`:
```python
from problems import NEURIPS2023_D2

problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
exp_d2 = ExperimentWithPerSeriesSeeding(
    problems_d2, estimators,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d2],
    seed=123,
    est_names=est_names, verbose=True).run()
print()
```

Cell `a009`:
```python
from problems import NEURIPS2023_D3

problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
exp_d3 = ExperimentWithPerSeriesSeeding(
    problems_d3, estimators,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d3],
    seed=123,
    est_names=est_names, verbose=True).run()
print()
```

- [ ] **Step 5: Verify CI notebook executes**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --nbmake experiments/real_data.ipynb -v
```

Expected: PASS (skip-execution cells skipped; preview cells run with `ignore_cache=True`)

- [ ] **Step 6: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: migrate real_data.ipynb to Experiment (PCG64 per-trial caching)"
```

---

### Task 7: Migrate `real_data_neurips2023.ipynb` to `ExperimentWithPerSeriesSeeding`

**Files:**
- Modify: `experiments/real_data_neurips2023.ipynb`

**Note:** All cells in this notebook are tagged `skip-execution` — nbmake provides no
coverage. Correctness relies on `test_series_exp_numerical_equivalence` (Task 4) plus
manual user review of the notebook output after running it locally.

- [ ] **Step 1: Update `real_data_neurips2023.ipynb`**

Apply the four cell edits from Task 6 Step 4 above (cells `a003`, `a006`, `a009`).

- [ ] **Step 2: Commit**

```bash
git add experiments/real_data_neurips2023.ipynb
git commit -m "feat: migrate real_data_neurips2023.ipynb to ExperimentWithPerSeriesSeeding"
```

- [ ] **Step 3: User validation**

Ask the user to run `real_data_neurips2023.ipynb` locally (e.g. reps=100 on a small
problem subset) and confirm results match expectations before merging.
