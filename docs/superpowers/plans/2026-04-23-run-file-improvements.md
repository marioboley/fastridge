# Run File Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make run files human-readable, attributable to a runner class, and written at both the start and end of each run so interrupted runs leave a traceable record.

**Architecture:** Three focused changes to module-level helpers in `experiments/experiments.py`, then mechanical updates to `run()` in both runner classes. Cache key format and stats file format are unchanged — those are a separate future plan.

**Tech Stack:** Python stdlib (`json`, `os`, `tempfile`, `datetime`, `random`, `string`, `sys`, `platform`, `dataclasses`), `joblib`

**Status:** Task 1 (`_make_run_id`) is complete and committed.

---

## File Map

- Modify: `experiments/experiments.py` — helpers section, `Experiment.run()`, `ExperimentWithPerSeriesSeeding.run()`
- Modify: `tests/test_experiments.py` — update existing run file test, add integration tests

---

### Task 1: `_make_run_id` class name ✅ DONE

`_make_run_id(class_name)` now returns `ClassName__YYYYMMDD-HHMMSS-xxxx`.
Both `run()` methods updated to call `_make_run_id(type(self).__name__)`.

---

### Task 2: `_write_json_atomic`, `_problem_params`, and rewritten `_write_run_file`

**Files:**
- Modify: `experiments/experiments.py:1-80`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Add `import dataclasses` to `experiments/experiments.py`**

Add after `import platform`:

```python
import dataclasses
```

- [ ] **Step 2: Add `_write_json_atomic`, `_problem_params`, and replace `_write_run_file`**

Insert immediately after the existing `_save_metric_file` function:

```python
def _write_json_atomic(path, data):
    """Write data as JSON to path atomically using a tempfile + os.replace.

    Creates parent directories as needed. On failure, cleans up the tempfile
    and emits a warning rather than raising, so a cache write failure does not
    abort an experiment run.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        warnings.warn(f'Cache write failed for {path}: {e}')


def _problem_params(problem):
    """Return a JSON-serialisable dict of field values for a dataclass problem.

    JSON-native values (str, int, float, bool, None) are kept as-is; complex
    values (tuples, objects) are converted via repr(). Callers should treat
    repr() fields as human-readable annotations, not as reconstructable values.

    >>> from problems import EmpiricalDataProblem
    >>> p = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    >>> _problem_params(p)['dataset']
    'diabetes'
    >>> _problem_params(p)['zero_variance_filter']
    True
    """
    result = {}
    for field in dataclasses.fields(problem):
        val = getattr(problem, field.name)
        result[field.name] = (val if isinstance(val, (str, int, float, bool, type(None)))
                              else repr(val))
    return result
```

Then replace the entire `_write_run_file` function with:

```python
def _write_run_file(run_id, class_name, experiment_dict, timestamp_start, summary=None):
    """Write a run file to ``results/runs/{run_id}.json``.

    Called twice per run: once at the start with summary=None (status
    'in_progress') and once at the end with the summary dict (status
    'completed'). The timestamp_start is captured once by the caller and
    passed to both calls so both writes record the same start time.
    """
    data = {
        'run_id': run_id,
        'class': class_name,
        'status': 'in_progress' if summary is None else 'completed',
        'timestamp_start': timestamp_start,
        'timestamp_end': None if summary is None else datetime.datetime.now().isoformat(),
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'experiment': experiment_dict,
        'summary': summary,
    }
    _write_json_atomic(os.path.join(CACHE_DIR, 'runs', f'{run_id}.json'), data)
```

- [ ] **Step 3: Run tests to verify nothing is broken**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all pass (existing `test_write_run_file` uses the old 3-arg signature — fix it in next step).

Actually, `test_write_run_file` will fail because `_write_run_file` now has a different signature. Fix it:

```python
def test_write_run_file(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    run_id = _make_run_id('Experiment')
    ts = datetime.datetime.now().isoformat()
    _write_run_file(run_id, 'Experiment',
                    {'reps': 2, 'seed': 1, 'ns': [[100]], 'problems': [], 'estimators': []},
                    ts, {'trials_computed': 0, 'trials_retrieved': 0})
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data['run_id'] == run_id
    assert 'python' in data['environment']
```

Also update the import line at the top of the test helper block:

```python
import datetime
from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _make_run_id, _write_run_file, _problem_params)
```

- [ ] **Step 4: Run tests**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add _write_json_atomic, _problem_params; rewrite _write_run_file with human-readable params and start/end writes"
```

---

### Task 3: Update `Experiment.run()` to use new run file helpers

**Files:**
- Modify: `experiments/experiments.py` — `Experiment.run()`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing integration test**

Add after `test_new_experiment_run_file_written`:

```python
def test_new_experiment_run_file_content(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    filenames = os.listdir(runs_dir)
    assert len(filenames) == 1
    assert filenames[0].startswith('Experiment__')
    with open(os.path.join(runs_dir, filenames[0])) as f:
        data = json.load(f)
    assert data['class'] == 'Experiment'
    assert data['status'] == 'completed'
    assert data['timestamp_start'] is not None
    assert data['timestamp_end'] is not None
    assert data['experiment']['problems'][0]['params']['dataset'] == 'diabetes'
    assert data['experiment']['estimators'][0]['class'] == 'RidgeEM'
    assert data['experiment']['reps'] == 2
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_new_experiment_run_file_content -v
```

Expected: FAIL — `Experiment.run()` still calls old `_write_run_file` with 3 args.

- [ ] **Step 3: Replace `Experiment.run()`**

```python
def run(self, force_recompute=False, ignore_cache=False):
    n_problems = len(self.problems)
    n_sizes = len(self.ns[0])
    n_estimators = len(self.estimators)
    for stat in self.stats:
        self.__dict__[str(stat) + '_'] = np.full(
            (self.reps, n_problems, n_sizes, n_estimators), np.nan)
    self.run_id_ = _make_run_id(type(self).__name__)
    timestamp_start = datetime.datetime.now().isoformat()
    experiment_dict = {
        'reps': self.reps,
        'seed': self.seed,
        'ns': self.ns.tolist(),
        'problems': [{'key': _cache_key(p), 'class': type(p).__name__,
                      'params': _problem_params(p)} for p in self.problems],
        'estimators': [{'key': _cache_key(e), 'class': type(e).__name__,
                        'params': {k: v.tolist() if hasattr(v, 'tolist') else v
                                   for k, v in e.get_params(deep=False).items()}}
                       for e in self.estimators],
    }
    if not ignore_cache:
        _write_run_file(self.run_id_, type(self).__name__, experiment_dict,
                        timestamp_start)
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
                        print('.', end='', flush=True)
        if self.verbose:
            print()
    if not ignore_cache:
        _write_run_file(self.run_id_, type(self).__name__, experiment_dict,
                        timestamp_start,
                        {'trials_computed': trials_computed,
                         'trials_retrieved': trials_retrieved})
    return self
```

- [ ] **Step 4: Run all tests**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: update Experiment.run() to write human-readable run file at start and end"
```

---

### Task 4: Update `ExperimentWithPerSeriesSeeding.run()`

**Files:**
- Modify: `experiments/experiments.py` — `ExperimentWithPerSeriesSeeding.run()`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing integration test**

Add after the existing `ExperimentWithPerSeriesSeeding` tests:

```python
def test_series_exp_run_file_content(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    filenames = os.listdir(runs_dir)
    assert len(filenames) == 1
    assert filenames[0].startswith('ExperimentWithPerSeriesSeeding__')
    with open(os.path.join(runs_dir, filenames[0])) as f:
        data = json.load(f)
    assert data['class'] == 'ExperimentWithPerSeriesSeeding'
    assert data['status'] == 'completed'
    assert data['experiment']['problems'][0]['params']['dataset'] == 'diabetes'
    assert data['experiment']['estimators'][0]['class'] == 'RidgeEM'
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_series_exp_run_file_content -v
```

Expected: FAIL.

- [ ] **Step 3: Replace `ExperimentWithPerSeriesSeeding.run()`**

```python
def run(self, force_recompute=False, ignore_cache=False):
    n_problems = len(self.problems)
    n_sizes = len(self.ns[0])
    n_estimators = len(self.estimators)
    for stat in self.stats:
        self.__dict__[str(stat) + '_'] = np.full(
            (self.reps, n_problems, n_sizes, n_estimators), np.nan)
    self.run_id_ = _make_run_id(type(self).__name__)
    timestamp_start = datetime.datetime.now().isoformat()
    experiment_dict = {
        'reps': self.reps,
        'seed': self.seed,
        'ns': self.ns.tolist(),
        'problems': [{'key': _cache_key(p), 'class': type(p).__name__,
                      'params': _problem_params(p)} for p in self.problems],
        'estimators': [{'key': _cache_key(e), 'class': type(e).__name__,
                        'params': {k: v.tolist() if hasattr(v, 'tolist') else v
                                   for k, v in e.get_params(deep=False).items()}}
                       for e in self.estimators],
    }
    if not ignore_cache:
        _write_run_file(self.run_id_, type(self).__name__, experiment_dict,
                        timestamp_start)
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
        _write_run_file(self.run_id_, type(self).__name__, experiment_dict,
                        timestamp_start,
                        {'trials_computed': trials_computed,
                         'trials_retrieved': trials_retrieved})
    return self
```

- [ ] **Step 4: Run full test suite**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: update ExperimentWithPerSeriesSeeding.run() to write human-readable run file at start and end"
```
