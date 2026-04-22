# Result Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add result caching to `EmpiricalDataExperiment` so that re-running with a modified estimator list only recomputes affected units, reusing stored results for everything else.

**Architecture:** A `Metric` base class adds warning hooks to the existing stat classes. Module-level helpers manage per-metric JSON file I/O. `run()` gains `force_recompute` and `ignore_cache` parameters and a caching loop that checks/writes per-stat JSON files keyed by `joblib.hash()`. Series results (all reps) are cached as a unit; trial results are cached per rep.

**Tech Stack:** Python stdlib (`json`, `os`, `tempfile`, `datetime`, `random`, `string`, `sys`, `platform`), `numpy`, `joblib` (transitive dependency of scikit-learn, always available)

---

## File Structure

- Modify: `experiments/experiments.py` — add imports, `Metric` base class, `CACHE_DIR`, helper functions, `_seed_val()` method, updated `run()`
- Modify: `tests/test_experiments.py` — add caching tests
- Create: `results/.gitignore`

---

### Task 1: `Metric` base class and stat inheritance

**Files:**
- Modify: `experiments/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py`:

```python
import os
import json
from experiments import (Metric, ParameterMeanSquaredError, PredictionMeanSquaredError,
                         RegularizationParameter, NumberOfIterations, VarianceAbsoluteError,
                         FittingTime, PredictionR2, NumberOfFeatures)


def test_stat_classes_inherit_metric():
    for cls in [ParameterMeanSquaredError, PredictionMeanSquaredError,
                RegularizationParameter, NumberOfIterations, VarianceAbsoluteError,
                FittingTime, PredictionR2, NumberOfFeatures]:
        assert issubclass(cls, Metric)


def test_warn_recompute_returns_none_in_range():
    assert Metric().warn_recompute([0.85, 0.84, 0.86], 0.85) is None


def test_warn_recompute_returns_str_on_outlier():
    result = Metric().warn_recompute([0.85, 0.85, 0.85], 0.50)
    assert isinstance(result, str)


def test_warn_recompute_series_element_wise():
    existing = [[0.85, 0.90], [0.84, 0.91], [0.86, 0.89]]
    assert Metric().warn_recompute(existing, [0.85, 0.10]) is not None
    assert Metric().warn_recompute(existing, [0.85, 0.90]) is None


def test_warn_retrieval_default_returns_none():
    assert Metric().warn_retrieval([{'value': 0.85, 'run_id': 'x'}]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_stat_classes_inherit_metric tests/test_experiments.py::test_warn_recompute_returns_none_in_range -v
```

Expected: FAIL with `ImportError: cannot import name 'Metric'`

- [ ] **Step 3: Add `Metric` base class to `experiments/experiments.py`**

Insert after the imports block, before `class ParameterMeanSquaredError`:

```python
class Metric:

    def warn_recompute(self, existing, new_value):
        if not existing:
            return None
        arr = np.array(existing)
        mean = arr.mean(axis=0)
        se = (arr.std(axis=0, ddof=1) / np.sqrt(len(existing))
              if len(existing) >= 2 else np.zeros_like(mean))
        if np.any(np.abs(np.asarray(new_value) - mean) > 1.96 * se):
            return (f'{type(self).__name__}: new value outside 95% CI of '
                    f'{len(existing)} existing computation(s)')
        return None

    def warn_retrieval(self, computations):
        return None
```

- [ ] **Step 4: Update all stat classes to inherit from `Metric`**

Change the class declaration line for all 8 stat classes:

```python
class ParameterMeanSquaredError(Metric):
class PredictionMeanSquaredError(Metric):
class RegularizationParameter(Metric):
class NumberOfIterations(Metric):
class VarianceAbsoluteError(Metric):
class FittingTime(Metric):
class PredictionR2(Metric):
class NumberOfFeatures(Metric):
```

- [ ] **Step 5: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add Metric base class with warn_recompute/warn_retrieval; update stat inheritance"
```

---

### Task 2: `FittingTime.warn_retrieval` override

**Files:**
- Modify: `experiments/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py`:

```python
def test_fitting_time_warn_retrieval_single_computation():
    ft = FittingTime()
    result = ft.warn_retrieval([{'value': 0.5, 'run_id': 'x'}])
    assert isinstance(result, str)


def test_fitting_time_warn_retrieval_narrow_ci():
    ft = FittingTime()
    comps = [{'value': v, 'run_id': 'x'} for v in [0.5, 0.51, 0.49]]
    assert ft.warn_retrieval(comps) is None


def test_fitting_time_warn_retrieval_wide_ci():
    ft = FittingTime()
    comps = [{'value': v, 'run_id': 'x'} for v in [0.1, 10.0, 0.1]]
    assert ft.warn_retrieval(comps) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_fitting_time_warn_retrieval_single_computation -v
```

Expected: FAIL — default `warn_retrieval` returns None, not a string

- [ ] **Step 3: Override `warn_retrieval` in `FittingTime`**

Replace the `FittingTime` class body in `experiments/experiments.py`:

```python
class FittingTime(Metric):

    def warn_retrieval(self, computations):
        if not computations:
            return None
        if len(computations) == 1:
            return ('FittingTime: only one computation stored; reliability unknown. '
                    'Re-run with force_recompute=True to improve estimate.')
        means = np.array([np.mean(c['value']) for c in computations])
        se = means.std(ddof=1) / np.sqrt(len(computations))
        if 1.96 * se > 1.0:
            return (f'FittingTime: cached mean unreliable '
                    f'(95% CI width {2 * 1.96 * se:.1f}s). '
                    'Re-run with force_recompute=True.')
        return None

    @staticmethod
    def __call__(est, prob, x, y):
        return est.fitting_time_

    @staticmethod
    def __str__():
        return 'fitting_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{fit}$ [s]'
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: override FittingTime.warn_retrieval with CI-based reliability check"
```

---

### Task 3: Cache helper functions

**Files:**
- Modify: `experiments/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py`:

```python
from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _make_run_id, _write_run_file)


def test_cache_key_format():
    from problems import EmpiricalDataProblem
    key = _cache_key(EmpiricalDataProblem('diabetes', 'target'))
    assert key.startswith('EmpiricalDataProblem__')
    assert len(key.split('__')[1]) > 0


def test_load_metric_file_missing(tmp_path):
    data = _load_metric_file(str(tmp_path / 'missing.json'))
    assert data == {'computations': [], 'retrievals': []}


def test_save_load_metric_file_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'm.json')
    original = {'computations': [{'value': 0.85, 'run_id': 'abc'}], 'retrievals': []}
    _save_metric_file(path, original)
    assert _load_metric_file(path) == original


def test_make_run_id_format():
    run_id = _make_run_id()
    parts = run_id.split('-')
    assert len(parts) == 3
    assert len(parts[0]) == 8   # YYYYMMDD
    assert len(parts[1]) == 6   # HHMMSS
    assert len(parts[2]) == 4   # random suffix


def test_write_run_file(tmp_path):
    run_id = _make_run_id()
    _write_run_file(str(tmp_path), run_id, {'problems': []},
                    {'trials_computed': 0, 'trials_retrieved': 0})
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data['run_id'] == run_id
    assert 'python' in data['environment']
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_cache_key_format -v
```

Expected: FAIL with `ImportError: cannot import name '_cache_key'`

- [ ] **Step 3: Add imports to `experiments/experiments.py`**

Replace the existing imports block at the top of the file with:

```python
import time
import warnings
import json
import os
import tempfile
import datetime
import random
import string
import sys
import platform
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastprogress.fastprogress import progress_bar
```

- [ ] **Step 4: Add `CACHE_DIR` and helper functions**

Insert after the imports block, before `class Metric:`:

```python
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def _cache_key(obj):
    return f'{type(obj).__name__}__{joblib.hash(obj)}'


def _series_dir(cache_dir, prob_key, n_train, est_key, reps, generator, seed):
    return os.path.join(cache_dir, 'series', prob_key, str(n_train),
                        est_key, str(reps), generator, str(seed))


def _trial_dir(cache_dir, prob_key, n_train, est_key, generator, trial_seed):
    return os.path.join(cache_dir, 'trial', prob_key, str(n_train),
                        est_key, generator, str(trial_seed))


def _load_metric_file(path):
    if not os.path.exists(path):
        return {'computations': [], 'retrievals': []}
    with open(path) as f:
        return json.load(f)


def _save_metric_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _make_run_id():
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'{ts}-{suffix}'


def _write_run_file(cache_dir, run_id, exp_spec, summary):
    data = {
        'run_id': run_id,
        'timestamp': datetime.datetime.now().isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'experiment_spec': exp_spec,
        'summary': summary,
    }
    path = os.path.join(cache_dir, 'runs', f'{run_id}.json')
    _save_metric_file(path, data)
```

- [ ] **Step 5: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add CACHE_DIR, cache key/path helpers, JSON metric I/O, run file helpers"
```

---

### Task 4: Caching integration in `run()`

**Files:**
- Modify: `experiments/experiments.py`
- Test: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py`:

```python
import experiments


def _simple_trial_exp(**kwargs):
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    ns = n_train_from_proportion([prob])
    defaults = dict(seed=1, generator='MT19937', seed_scope='trial',
                    seed_progression='sequential', verbose=False)
    defaults.update(kwargs)
    return EmpiricalDataExperiment([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


def test_series_cache_hit_reuses_result(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_exp().run()
    exp2 = _simple_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_series_ignore_cache_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'series'))


def test_series_force_recompute_appends_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_exp().run()
    _simple_exp().run(force_recompute=True)
    json_files = []
    series_dir = os.path.join(str(tmp_path), 'series')
    for root, _, files in os.walk(series_dir):
        json_files += [os.path.join(root, f) for f in files if f.endswith('.json')]
    assert json_files
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 2


def test_run_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    assert os.path.exists(runs_dir)
    assert len(os.listdir(runs_dir)) == 1


def test_trial_cache_hit_reuses_result(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    exp1 = _simple_trial_exp().run()
    exp2 = _simple_trial_exp().run()
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


def test_trial_ignore_cache_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_trial_exp().run(ignore_cache=True)
    assert not os.path.exists(os.path.join(str(tmp_path), 'trial'))
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_series_cache_hit_reuses_result -v
```

Expected: FAIL — `run()` does not accept `force_recompute` or `ignore_cache`

- [ ] **Step 3: Add `_seed_val` method to `EmpiricalDataExperiment`**

Insert after the `_make_rng` method (around line 349):

```python
def _seed_val(self, unit_idx):
    if self.seed is None:
        return None
    return self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx
```

- [ ] **Step 4: Replace `run()` with the caching version**

Replace the entire `run(self)` method with:

```python
def run(self, force_recompute=False, ignore_cache=False):
    n_problems = len(self.problems)
    n_estimators = len(self.estimators)
    n_sizes = len(self.ns[0])

    for stat in self.stats:
        self.__dict__[str(stat) + '_'] = np.full(
            (self.reps, n_problems, n_sizes, n_estimators), np.nan)

    run_id = _make_run_id()
    trials_computed = 0
    trials_retrieved = 0

    if self.seed_scope == 'experiment':
        self.rng = self._make_rng(unit_idx=0)

    for prob_idx, problem in enumerate(self.problems):
        if self.verbose:
            print(problem.dataset, end=' ')

        prob_key = _cache_key(problem)

        for n_idx, n_train in enumerate(self.ns[prob_idx]):
            n_train = int(n_train)

            if self.seed_scope == 'series':
                self.rng = self._make_rng(unit_idx=prob_idx)
                seed_val = self._seed_val(prob_idx)

                cached = {}
                if not ignore_cache and not force_recompute:
                    for est_idx, est in enumerate(self.estimators):
                        est_key = _cache_key(est)
                        d = _series_dir(CACHE_DIR, prob_key, n_train, est_key,
                                        self.reps, self.generator, seed_val)
                        hits = {}
                        for stat in self.stats:
                            path = os.path.join(d, str(stat) + '.json')
                            file_data = _load_metric_file(path)
                            if file_data['computations']:
                                hits[str(stat)] = file_data
                        if len(hits) == len(self.stats):
                            cached[est_idx] = hits

                for est_idx, hits in cached.items():
                    est_key = _cache_key(self.estimators[est_idx])
                    d = _series_dir(CACHE_DIR, prob_key, n_train, est_key,
                                    self.reps, self.generator, seed_val)
                    for stat in self.stats:
                        file_data = hits[str(stat)]
                        values = [np.asarray(c['value']) for c in file_data['computations']]
                        mean_val = np.mean(values, axis=0)
                        self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
                        serialisable = mean_val.tolist() if hasattr(mean_val, 'tolist') else float(mean_val)
                        file_data['retrievals'].append({'value': serialisable, 'run_id': run_id})
                        _save_metric_file(os.path.join(d, str(stat) + '.json'), file_data)
                        if isinstance(stat, Metric):
                            msg = stat.warn_retrieval(file_data['computations'])
                            if msg:
                                warnings.warn(msg)
                    trials_retrieved += self.reps

                uncached = [i for i in range(n_estimators) if i not in cached]
                if uncached:
                    per_est = {i: {str(s): [] for s in self.stats} for i in uncached}

                    for iter_idx in range(self.reps):
                        if self.verbose:
                            print('.', end='')
                        X_train, X_test, y_train, y_test = problem.get_X_y(
                            n_train, rng=self.rng)

                        for est_idx in uncached:
                            _est = clone(self.estimators[est_idx], safe=False)
                            try:
                                t0 = time.time()
                                _est.fit(X_train, y_train)
                                _est.fitting_time_ = time.time() - t0
                            except Exception as e:
                                warnings.warn(
                                    f"Run {iter_idx} failed for "
                                    f"'{self.est_names[est_idx]}' on "
                                    f"'{problem.dataset}': {e}")
                                for stat in self.stats:
                                    per_est[est_idx][str(stat)].append(float('nan'))
                                continue

                            for stat in self.stats:
                                val = stat(_est, problem, X_test, y_test)
                                self.__dict__[str(stat) + '_'][
                                    iter_idx, prob_idx, n_idx, est_idx] = val
                                per_est[est_idx][str(stat)].append(
                                    val.tolist() if hasattr(val, 'tolist') else float(val))

                    trials_computed += self.reps * len(uncached)

                    if not ignore_cache:
                        for est_idx in uncached:
                            est_key = _cache_key(self.estimators[est_idx])
                            d = _series_dir(CACHE_DIR, prob_key, n_train, est_key,
                                            self.reps, self.generator, seed_val)
                            for stat in self.stats:
                                path = os.path.join(d, str(stat) + '.json')
                                file_data = _load_metric_file(path)
                                new_values = per_est[est_idx][str(stat)]
                                if isinstance(stat, Metric) and file_data['computations']:
                                    existing = [np.asarray(c['value'])
                                                for c in file_data['computations']]
                                    msg = stat.warn_recompute(existing, np.asarray(new_values))
                                    if msg:
                                        warnings.warn(msg)
                                file_data['computations'].append(
                                    {'value': new_values, 'run_id': run_id})
                                _save_metric_file(path, file_data)

            elif self.seed_scope == 'trial':
                for iter_idx in range(self.reps):
                    if self.verbose:
                        print('.', end='')
                    self.rng = self._make_rng(unit_idx=iter_idx)
                    trial_seed = self._seed_val(iter_idx)
                    X_train, X_test, y_train, y_test = problem.get_X_y(
                        n_train, rng=self.rng)

                    for est_idx, est in enumerate(self.estimators):
                        est_key = _cache_key(est)
                        d = _trial_dir(CACHE_DIR, prob_key, n_train, est_key,
                                       self.generator, trial_seed)

                        trial_hits = {}
                        if not ignore_cache and not force_recompute:
                            hits = {}
                            for stat in self.stats:
                                path = os.path.join(d, str(stat) + '.json')
                                file_data = _load_metric_file(path)
                                if file_data['computations']:
                                    hits[str(stat)] = file_data
                            if len(hits) == len(self.stats):
                                trial_hits = hits

                        if trial_hits:
                            for stat in self.stats:
                                file_data = trial_hits[str(stat)]
                                values = [c['value'] for c in file_data['computations']]
                                mean_val = float(np.mean(values))
                                self.__dict__[str(stat) + '_'][
                                    iter_idx, prob_idx, n_idx, est_idx] = mean_val
                                file_data['retrievals'].append(
                                    {'value': mean_val, 'run_id': run_id})
                                _save_metric_file(os.path.join(d, str(stat) + '.json'), file_data)
                                if isinstance(stat, Metric):
                                    msg = stat.warn_retrieval(file_data['computations'])
                                    if msg:
                                        warnings.warn(msg)
                            trials_retrieved += 1
                        else:
                            _est = clone(est, safe=False)
                            try:
                                t0 = time.time()
                                _est.fit(X_train, y_train)
                                _est.fitting_time_ = time.time() - t0
                            except Exception as e:
                                warnings.warn(
                                    f"Run {iter_idx} failed for "
                                    f"'{self.est_names[est_idx]}' on "
                                    f"'{problem.dataset}': {e}")
                                continue

                            for stat in self.stats:
                                val = stat(_est, problem, X_test, y_test)
                                self.__dict__[str(stat) + '_'][
                                    iter_idx, prob_idx, n_idx, est_idx] = val
                                if not ignore_cache:
                                    path = os.path.join(d, str(stat) + '.json')
                                    file_data = _load_metric_file(path)
                                    scalar_val = (val.tolist() if hasattr(val, 'tolist')
                                                  else float(val))
                                    if isinstance(stat, Metric) and file_data['computations']:
                                        existing = [c['value']
                                                    for c in file_data['computations']]
                                        msg = stat.warn_recompute(existing, scalar_val)
                                        if msg:
                                            warnings.warn(msg)
                                    file_data['computations'].append(
                                        {'value': scalar_val, 'run_id': run_id})
                                    _save_metric_file(path, file_data)
                            trials_computed += 1

            else:  # seed_scope == 'experiment' — no per-trial/series caching
                for iter_idx in range(self.reps):
                    if self.verbose:
                        print('.', end='')
                    X_train, X_test, y_train, y_test = problem.get_X_y(
                        n_train, rng=self.rng)
                    for est_idx, est in enumerate(self.estimators):
                        _est = clone(est, safe=False)
                        try:
                            t0 = time.time()
                            _est.fit(X_train, y_train)
                            _est.fitting_time_ = time.time() - t0
                        except Exception as e:
                            warnings.warn(
                                f"Run {iter_idx} failed for "
                                f"'{self.est_names[est_idx]}' on "
                                f"'{problem.dataset}': {e}")
                            continue
                        for stat in self.stats:
                            self.__dict__[str(stat) + '_'][
                                iter_idx, prob_idx, n_idx, est_idx] = stat(
                                    _est, problem, X_test, y_test)
                    trials_computed += n_estimators

        if self.verbose:
            print()

    if not ignore_cache:
        exp_spec = {
            'problems': [_cache_key(p) for p in self.problems],
            'estimators': [_cache_key(e) for e in self.estimators],
            'ns': self.ns.tolist(),
            'reps': self.reps,
            'seed': self.seed,
            'generator': self.generator,
            'seed_scope': self.seed_scope,
        }
        _write_run_file(CACHE_DIR, run_id, exp_spec, {
            'trials_computed': trials_computed,
            'trials_retrieved': trials_retrieved,
        })

    return self
```

- [ ] **Step 5: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add caching loop to run() with force_recompute, ignore_cache, and run file"
```

---

### Task 5: `results/.gitignore`

**Files:**
- Create: `results/.gitignore`

- [ ] **Step 1: Create the results directory and gitignore**

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

## Self-Review

**Spec coverage:**
- Metric base class with warn_recompute/warn_retrieval — Task 1
- FittingTime override — Task 2
- CACHE_DIR, _cache_key, JSON I/O, run file — Task 3
- force_recompute, ignore_cache, caching loop (series + trial + experiment passthrough) — Task 4
- results/.gitignore — Task 5
- Atomic file writes (tempfile + os.replace) — in _save_metric_file, Task 3
- Trial-level summary counts — in run(), Task 4
- Per-metric JSON format with computations/retrievals — in run() + I/O helpers

**Placeholders:** None found.

**Type consistency:** `_seed_val` introduced in Task 4, used internally in `run()`. `_series_dir` / `_trial_dir` signatures defined in Task 3, called in Task 4. `_cache_key`, `_load_metric_file`, `_save_metric_file`, `_make_run_id`, `_write_run_file` all defined in Task 3 and called in Task 4. Consistent throughout.
