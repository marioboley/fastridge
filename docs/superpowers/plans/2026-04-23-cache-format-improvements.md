# Cache Format Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four cache usability and correctness issues: opaque directory names, unreadable run files, per-metric file disk waste, and orphan entries from interrupted runs.

**Architecture:** Four focused changes to module-level helpers in `experiments/experiments.py`, then mechanical updates to both runner classes (`Experiment`, `ExperimentWithPerSeriesSeeding`) to call the new helpers. Old per-metric helpers are removed once no class uses them. Tests are updated in lockstep with each change.

**Tech Stack:** Python stdlib (`json`, `os`, `tempfile`, `datetime`, `random`, `string`, `sys`, `platform`, `dataclasses`), `numpy`, `joblib`

---

## File Map

- Modify: `experiments/experiments.py` — helpers section (lines 20–65), both runner classes
- Modify: `tests/test_experiments.py` — update helper tests, add new assertions

---

### Task 1: `_cache_key` slug and `_make_run_id` class name

**Files:**
- Modify: `experiments/experiments.py:23-50`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Replace the existing `test_cache_key_format` and `test_make_run_id_format` functions in `tests/test_experiments.py` with:

```python
def test_cache_key_problem_includes_dataset_slug():
    key = _cache_key(EmpiricalDataProblem('diabetes', 'target'))
    assert key.startswith('EmpiricalDataProblem_diabetes__')
    assert len(key.split('__')[1]) > 0


def test_cache_key_estimator_no_slug():
    key = _cache_key(RidgeEM())
    assert key.startswith('RidgeEM__')
    assert len(key.split('__')[1]) > 0


def test_make_run_id_format():
    run_id = _make_run_id('Experiment')
    assert run_id.startswith('Experiment__')
    tail = run_id[len('Experiment__'):]
    parts = tail.split('-')
    assert len(parts) == 3
    assert len(parts[0]) == 8   # YYYYMMDD
    assert len(parts[1]) == 6   # HHMMSS
    assert len(parts[2]) == 4   # random suffix
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_cache_key_problem_includes_dataset_slug tests/test_experiments.py::test_make_run_id_format -v
```

Expected: FAIL — old `_cache_key` has no slug; `_make_run_id` takes no argument.

- [ ] **Step 3: Update `_cache_key` and `_make_run_id` in `experiments/experiments.py`**

Replace lines 23–24 (`_cache_key`) and lines 47–50 (`_make_run_id`) with:

```python
def _cache_key(obj):
    slug = getattr(obj, 'dataset', '')
    sep = '_' if slug else ''
    return f'{type(obj).__name__}{sep}{slug}__{joblib.hash(obj)}'


def _make_run_id(class_name):
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'{class_name}__{ts}-{suffix}'
```

- [ ] **Step 4: Run all tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS. (The two old tests were replaced in Step 1; `_make_run_id` is not yet called with a class name by the runner — that comes in Tasks 3–4.)

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add dataset slug to _cache_key; add class_name param to _make_run_id"
```

---

### Task 2: New stats file helpers, `_problem_params`, and rewritten `_write_run_file`

The old `_load_metric_file` and `_save_metric_file` are kept in place — both runner classes still use them and they are removed in Task 4. The new helpers are added alongside.

**Files:**
- Modify: `experiments/experiments.py:1-65`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Update the import line in `tests/test_experiments.py` that currently reads:
```python
from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _make_run_id, _write_run_file)
```
to:
```python
import datetime
from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _load_stats_file, _save_stats_file,
                         _make_run_id, _write_run_file, _problem_params)
```

Then add these tests after the existing `test_write_run_file`:

```python
def test_load_stats_file_missing(tmp_path):
    data = _load_stats_file(str(tmp_path))
    assert data == {}


def test_save_load_stats_file_roundtrip(tmp_path):
    original = {
        'prediction_r2': {'computations': [{'value': 0.85, 'run_id': 'Experiment__abc'}],
                          'retrievals': []},
        'fitting_time': {'computations': [{'value': 0.01, 'run_id': 'Experiment__abc'}],
                         'retrievals': []},
    }
    _save_stats_file(str(tmp_path), original)
    assert _load_stats_file(str(tmp_path)) == original


def test_save_stats_file_pretty_printed(tmp_path):
    _save_stats_file(str(tmp_path), {'prediction_r2': {'computations': [], 'retrievals': []}})
    with open(os.path.join(str(tmp_path), 'stats.json')) as f:
        content = f.read()
    assert '\n' in content


def test_problem_params_json_native_fields():
    params = _problem_params(EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True))
    assert params['dataset'] == 'diabetes'
    assert params['target'] == 'target'
    assert params['zero_variance_filter'] is True


def test_problem_params_complex_field_as_repr():
    from problems import OneHotEncodeCategories
    prob = EmpiricalDataProblem('abalone', 'Rings',
                                x_transforms=(OneHotEncodeCategories(),))
    params = _problem_params(prob)
    assert isinstance(params['x_transforms'], str)


def test_write_run_file_in_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    run_id = _make_run_id('Experiment')
    ts = datetime.datetime.now().isoformat()
    exp_dict = {'reps': 2, 'seed': 1, 'ns': [[100]], 'problems': [], 'estimators': []}
    _write_run_file(run_id, 'Experiment', exp_dict, ts)
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data['run_id'] == run_id
    assert data['class'] == 'Experiment'
    assert data['status'] == 'in_progress'
    assert data['summary'] is None
    assert 'python' in data['environment']


def test_write_run_file_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    run_id = _make_run_id('Experiment')
    ts = datetime.datetime.now().isoformat()
    exp_dict = {'reps': 2, 'seed': 1, 'ns': [[100]], 'problems': [], 'estimators': []}
    _write_run_file(run_id, 'Experiment', exp_dict, ts)
    _write_run_file(run_id, 'Experiment', exp_dict, ts,
                    {'trials_computed': 2, 'trials_retrieved': 0})
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    with open(path) as f:
        data = json.load(f)
    assert data['status'] == 'completed'
    assert data['timestamp_end'] is not None
    assert data['summary']['trials_computed'] == 2
```

Also update `test_write_run_file` (the original test) to match the new signature:

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

- [ ] **Step 2: Run tests to verify new tests fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_load_stats_file_missing tests/test_experiments.py::test_problem_params_json_native_fields tests/test_experiments.py::test_write_run_file_in_progress -v
```

Expected: FAIL with `ImportError` for new names.

- [ ] **Step 3: Add `dataclasses` import to `experiments/experiments.py`**

Add `import dataclasses` to the imports block (after `import platform`).

- [ ] **Step 4: Add `_write_json_atomic`, `_load_stats_file`, `_save_stats_file`, `_problem_params` and replace `_write_run_file` in `experiments/experiments.py`**

Insert the following immediately after the existing `_save_metric_file` function (keep `_load_metric_file` and `_save_metric_file` in place — they are removed in Task 4):

```python
def _write_json_atomic(path, data):
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


def _load_stats_file(dir_path):
    path = os.path.join(dir_path, 'stats.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_stats_file(dir_path, data):
    _write_json_atomic(os.path.join(dir_path, 'stats.json'), data)


def _problem_params(problem):
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

- [ ] **Step 5: Run all tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: add _load_stats_file/_save_stats_file, _problem_params; rewrite _write_run_file with human-readable params and start/end writes"
```

---

### Task 3: Update `Experiment` to use new helpers

**Files:**
- Modify: `experiments/experiments.py:314-418`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py` (after the existing `Experiment` tests):

```python
def test_new_experiment_stats_json_at_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    trial_dir = os.path.join(str(tmp_path), 'trial')
    stats_files = [os.path.join(r, f)
                   for r, _, fs in os.walk(trial_dir)
                   for f in fs if f == 'stats.json']
    assert stats_files
    with open(stats_files[0]) as f:
        data = json.load(f)
    assert 'prediction_r2' in data
    assert data['prediction_r2']['computations']


def test_new_experiment_run_file_class_name_in_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    filenames = os.listdir(runs_dir)
    assert len(filenames) == 1
    assert filenames[0].startswith('Experiment__')


def test_new_experiment_run_file_has_human_readable_params(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    with open(os.path.join(runs_dir, os.listdir(runs_dir)[0])) as f:
        data = json.load(f)
    assert data['experiment']['problems'][0]['params']['dataset'] == 'diabetes'
    assert data['experiment']['estimators'][0]['class'] == 'RidgeEM'
    assert data['status'] == 'completed'
    assert data['timestamp_end'] is not None
```

Also update `test_new_experiment_force_recompute` to look for `stats.json` instead of per-metric files:

```python
def test_new_experiment_force_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    _simple_new_exp().run(force_recompute=True)
    trial_dir = os.path.join(str(tmp_path), 'trial')
    stats_files = [os.path.join(r, f)
                   for r, _, fs in os.walk(trial_dir)
                   for f in fs if f == 'stats.json']
    assert stats_files
    with open(stats_files[0]) as f:
        data = json.load(f)
    first_stat = next(iter(data))
    assert len(data[first_stat]['computations']) == 2
```

Also update `test_new_experiment_run_file_written` to check the filename prefix:

```python
def test_new_experiment_run_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    assert len(os.listdir(runs_dir)) == 1
```

- [ ] **Step 2: Run tests to verify new tests fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_new_experiment_stats_json_at_leaf tests/test_experiments.py::test_new_experiment_run_file_class_name_in_filename -v
```

Expected: FAIL — `Experiment` still uses old per-metric file helpers.

- [ ] **Step 3: Replace `_all_stats_in_trial_cache`, `_retrieve_trial`, `_write_trial`, and `run()` in `Experiment`**

Replace the four methods in the `Experiment` class:

```python
def _all_stats_in_trial_cache(self, prob_idx, n_idx, est_idx, rep_idx):
    d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
    data = _load_stats_file(d)
    return all(data.get(str(stat), {}).get('computations') for stat in self.stats)

def _retrieve_trial(self, prob_idx, n_idx, est_idx, rep_idx):
    d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
    data = _load_stats_file(d)
    for stat in self.stats:
        stat_data = data[str(stat)]
        mean_val = float(np.mean([c['value'] for c in stat_data['computations']]))
        self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = mean_val
        stat_data['retrievals'].append({'value': mean_val, 'run_id': self.run_id_})
        msg = stat.warn_retrieval(stat_data['computations'])
        if msg:
            warnings.warn(msg)
    _save_stats_file(d, data)

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
    data = _load_stats_file(d)
    for stat in self.stats:
        stat_data = data.get(str(stat), {'computations': [], 'retrievals': []})
        new_val = self._last_trial_values[str(stat)]
        if stat_data['computations']:
            msg = stat.warn_recompute(
                [c['value'] for c in stat_data['computations']], new_val)
            if msg:
                warnings.warn(msg)
        stat_data['computations'].append({'value': new_val, 'run_id': self.run_id_})
        data[str(stat)] = stat_data
    _save_stats_file(d, data)

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

- [ ] **Step 4: Run all tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: update Experiment to use stats.json, slug cache keys, and human-readable run files"
```

---

### Task 4: Update `ExperimentWithPerSeriesSeeding`, remove old helpers

**Files:**
- Modify: `experiments/experiments.py:421-562`
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_experiments.py`:

```python
def test_series_exp_stats_json_at_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    series_dir = os.path.join(str(tmp_path), 'series')
    stats_files = [os.path.join(r, f)
                   for r, _, fs in os.walk(series_dir)
                   for f in fs if f == 'stats.json']
    assert stats_files
    with open(stats_files[0]) as f:
        data = json.load(f)
    assert 'prediction_r2' in data
    assert data['prediction_r2']['computations']


def test_series_exp_run_file_class_name_in_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    filenames = os.listdir(runs_dir)
    assert len(filenames) == 1
    assert filenames[0].startswith('ExperimentWithPerSeriesSeeding__')


def test_series_exp_run_file_has_human_readable_params(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    with open(os.path.join(runs_dir, os.listdir(runs_dir)[0])) as f:
        data = json.load(f)
    assert data['experiment']['problems'][0]['params']['dataset'] == 'diabetes'
    assert data['status'] == 'completed'
```

- [ ] **Step 2: Run tests to verify new tests fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py::test_series_exp_stats_json_at_leaf tests/test_experiments.py::test_series_exp_run_file_class_name_in_filename -v
```

Expected: FAIL — `ExperimentWithPerSeriesSeeding` still uses old helpers.

- [ ] **Step 3: Replace `_all_stats_in_series_cache`, `_retrieve_series`, `_write_series`, and `run()` in `ExperimentWithPerSeriesSeeding`**

```python
def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    data = _load_stats_file(d)
    return all(data.get(str(stat), {}).get('computations') for stat in self.stats)

def _retrieve_series(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    data = _load_stats_file(d)
    for stat in self.stats:
        stat_data = data[str(stat)]
        values = [np.asarray(c['value']) for c in stat_data['computations']]
        mean_val = np.mean(values, axis=0)
        self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
        serialisable = mean_val.tolist() if hasattr(mean_val, 'tolist') else float(mean_val)
        stat_data['retrievals'].append({'value': serialisable, 'run_id': self.run_id_})
        msg = stat.warn_retrieval(stat_data['computations'])
        if msg:
            warnings.warn(msg)
    _save_stats_file(d, data)
    if self.verbose:
        for _ in range(self.reps):
            print('.', end='', flush=True)

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
        else:
            for stat in self.stats:
                val = stat(_est, problem, X_test, y_test)
                self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
                self._last_series_values[str(stat)].append(
                    val.tolist() if hasattr(val, 'tolist') else float(val))
        if self.verbose:
            print('.', end='', flush=True)

def _write_series(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    data = _load_stats_file(d)
    for stat in self.stats:
        stat_data = data.get(str(stat), {'computations': [], 'retrievals': []})
        new_values = self._last_series_values[str(stat)]
        if stat_data['computations']:
            existing = [np.asarray(c['value']) for c in stat_data['computations']]
            msg = stat.warn_recompute(existing, np.asarray(new_values))
            if msg:
                warnings.warn(msg)
        stat_data['computations'].append({'value': new_values, 'run_id': self.run_id_})
        data[str(stat)] = stat_data
    _save_stats_file(d, data)

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

- [ ] **Step 4: Remove `_load_metric_file` and `_save_metric_file` from `experiments/experiments.py`**

Delete the two functions entirely (they are no longer called by any code).

- [ ] **Step 5: Update test imports to remove old helpers**

Change the import line in `tests/test_experiments.py` from:
```python
from experiments import (_cache_key, _load_metric_file, _save_metric_file,
                         _load_stats_file, _save_stats_file,
                         _make_run_id, _write_run_file, _problem_params)
```
to:
```python
from experiments import (_cache_key, _load_stats_file, _save_stats_file,
                         _make_run_id, _write_run_file, _problem_params)
```

Remove `test_load_metric_file_missing` and `test_save_load_metric_file_roundtrip` (these tested the deleted functions).

- [ ] **Step 6: Run all tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 7: Run full test suite**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "feat: update ExperimentWithPerSeriesSeeding to stats.json and human-readable run files; remove old metric file helpers"
```

---

## Self-Review

**Spec coverage:**
- Metrics file consolidation (`stats.json` per leaf, `_load_stats_file`/`_save_stats_file`) — Tasks 2, 3, 4 ✓
- Human-readable slug in cache key — Task 1 ✓
- Run file: class name in filename — Task 1 (`_make_run_id`), Tasks 3–4 (`run()`) ✓
- Run file: human-readable params (`_problem_params`, estimator `get_params`) — Task 2 ✓
- Run file: written at start and end — Tasks 3–4 (`run()` calls `_write_run_file` twice) ✓
- Pretty-printing with `indent=2` — `_write_json_atomic` in Task 2 ✓

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:**
- `_make_run_id(class_name)` — defined Task 1, called with `type(self).__name__` in Tasks 3–4 ✓
- `_load_stats_file(dir_path)` → `dict` — defined Task 2, used in Tasks 3–4 ✓
- `_save_stats_file(dir_path, data)` — defined Task 2, used in Tasks 3–4 ✓
- `_write_run_file(run_id, class_name, experiment_dict, timestamp_start, summary=None)` — defined Task 2, called with matching args in Tasks 3–4 ✓
- `_problem_params(problem)` → `dict` — defined Task 2, called in Tasks 3–4 experiment_dict build ✓
- `_write_json_atomic(path, data)` — defined Task 2, called by `_save_stats_file` and `_write_run_file` ✓
