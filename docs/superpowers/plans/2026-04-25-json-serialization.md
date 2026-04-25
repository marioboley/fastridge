# Unified JSON Serialisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `experiments/util.py` with `to_json`, `from_json`, `save_json`, and `load_json`, then wire them into `experiments.py` to replace ad-hoc JSON helpers and give run files a full serialised experiment snapshot.

**Architecture:** `util.py` provides four public functions and three private helpers with no dependency on `experiments.py`. `experiments.py` imports them, drops `_save_json`, `_load_metric_file`, and `_write_run_file`, and replaces all metric-cache and run-file I/O with the framework calls. Run-file content becomes a `to_json` snapshot of the experiment runner, written twice per run (start and end) using an explicit `_RUN_FILE_STATE` whitelist of computed attributes.

**Tech Stack:** Python stdlib (`json`, `os`, `tempfile`, `warnings`, `inspect`, `importlib`, `datetime`, `sys`, `platform`), `numpy`

---

## File Map

- **Create:** `experiments/util.py` — `to_json`, `from_json`, `save_json`, `load_json`, private helpers `_is_named_import`, `_init_params`, `_computed_keys`
- **Create:** `tests/test_util.py` — unit tests for `util.py` plus end-to-end experiment roundtrip tests
- **Modify:** `experiments/experiments.py` — drop `_save_json`, `_load_metric_file`, `_write_run_file`; add `from util import ...`; add `_METRIC_FILE_DEFAULT`, `_RUN_FILE_STATE`; add computed attributes to both `run()` methods; replace all metric cache I/O
- **Modify:** `tests/test_experiments.py` — update imports; replace stale private-function tests; add run-file content test

---

### Task 1: `experiments/util.py` — complete implementation

**Files:**
- Create: `experiments/util.py`
- Create: `tests/test_util.py`

- [ ] **Step 1: Write failing tests for `save_json` and `load_json`**

Create `tests/test_util.py`:

```python
import json
import os
import warnings
import numpy as np
import pytest
from util import to_json, from_json, save_json, load_json


# ── save_json / load_json ────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / 'test.json')
    data = {'a': 1, 'b': [2, 3]}
    save_json(path, data)
    assert load_json(path) == data


def test_load_json_missing_returns_default(tmp_path):
    assert load_json(str(tmp_path / 'missing.json'), default={'x': 1}) == {'x': 1}


def test_load_json_missing_returns_none(tmp_path):
    assert load_json(str(tmp_path / 'missing.json')) is None


def test_save_json_creates_parent_dirs(tmp_path):
    path = str(tmp_path / 'a' / 'b' / 'test.json')
    save_json(path, {'x': 1})
    assert os.path.exists(path)


def test_save_json_pretty_printed(tmp_path):
    path = str(tmp_path / 'pretty.json')
    save_json(path, {'a': 1}, indent=2)
    with open(path) as f:
        assert '\n' in f.read()


def test_save_json_compact(tmp_path):
    path = str(tmp_path / 'compact.json')
    save_json(path, {'a': 1}, indent=None)
    with open(path) as f:
        assert '\n' not in f.read()


# ── to_json dispatch ─────────────────────────────────────────────────────────

def test_to_json_primitives():
    assert to_json(None) is None
    assert to_json(True) is True
    assert to_json(42) == 42
    assert to_json(3.14) == 3.14
    assert to_json('hello') == 'hello'


def test_to_json_numpy_integer():
    result = to_json(np.int64(42))
    assert result == 42
    assert type(result) is int


def test_to_json_numpy_floating():
    result = to_json(np.float64(3.14))
    assert abs(result - 3.14) < 1e-10
    assert type(result) is float


def test_to_json_ndarray():
    result = to_json(np.array([[1, 2], [3, 4]]))
    assert result == [[1, 2], [3, 4]]
    assert isinstance(result, list)


def test_to_json_list():
    assert to_json([1, np.int64(2), 'a']) == [1, 2, 'a']


def test_to_json_tuple():
    assert to_json((1, 2)) == {'__tuple__': [1, 2]}


def test_to_json_dict():
    assert to_json({'a': np.int64(1), 'b': [np.float64(2.0)]}) == {'a': 1, 'b': [2.0]}


def test_to_json_named_import():
    assert to_json(np.log) == {'__import__': 'numpy.log'}


def test_to_json_transparent_class():
    from problems import EmpiricalDataProblem
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    result = to_json(prob)
    assert result['__class__'] == 'problems.EmpiricalDataProblem'
    assert result['dataset'] == 'diabetes'
    assert result['zero_variance_filter'] is True
    assert result['x_transforms'] == {'__tuple__': []}


def test_to_json_include_computed_false_excludes_underscore_attrs():
    from experiments import Metric
    m = Metric()
    m.computed_ = 99
    assert 'computed_' not in to_json(m, include_computed=False)


def test_to_json_include_computed_true_includes_all():
    from experiments import Metric
    m = Metric()
    m.a_ = 1
    m.b_ = 2
    result = to_json(m, include_computed=True)
    assert result['a_'] == 1
    assert result['b_'] == 2


def test_to_json_include_computed_list_selects_named():
    from experiments import Metric
    m = Metric()
    m.a_ = 1
    m.b_ = 2
    result = to_json(m, include_computed=['a_'])
    assert result['a_'] == 1
    assert 'b_' not in result


def test_to_json_warns_on_missing_attribute():
    from experiments import Metric
    m = Metric.__new__(Metric)
    # Metric.__init__ is object.__init__ (no params), so this tests VAR_ filtering;
    # instead use a class that declares a param it never stores.
    class Leaky:
        def __init__(self, x):
            pass  # never stores x
    obj = Leaky.__new__(Leaky)
    with pytest.warns(UserWarning, match='x'):
        to_json(obj)


def test_to_json_does_not_propagate_include_computed():
    from experiments import Metric
    outer = Metric()
    outer.outer_ = 42
    inner = Metric()
    inner.inner_ = 99
    outer.inner = inner
    result = to_json(outer, include_computed=True)
    assert result['outer_'] == 42
    assert 'inner_' not in result['inner']


# ── from_json dispatch ───────────────────────────────────────────────────────

def test_from_json_primitives():
    assert from_json(None) is None
    assert from_json(True) is True
    assert from_json(42) == 42
    assert from_json(3.14) == pytest.approx(3.14)
    assert from_json('hello') == 'hello'


def test_from_json_list():
    assert from_json([1, 2, 3]) == [1, 2, 3]


def test_from_json_tuple():
    assert from_json({'__tuple__': [1, 2]}) == (1, 2)


def test_from_json_named_import():
    assert from_json({'__import__': 'numpy.log'}) is np.log


def test_from_json_plain_dict():
    assert from_json({'a': 1, 'b': [2, 3]}) == {'a': 1, 'b': [2, 3]}


def test_from_json_transparent_class():
    from problems import EmpiricalDataProblem
    data = {
        '__class__': 'problems.EmpiricalDataProblem',
        'dataset': 'diabetes',
        'target': 'target',
        'drop': {'__tuple__': []},
        'nan_policy': None,
        'x_transforms': {'__tuple__': []},
        'y_transforms': {'__tuple__': []},
        'zero_variance_filter': True,
    }
    obj = from_json(data)
    assert isinstance(obj, EmpiricalDataProblem)
    assert obj.dataset == 'diabetes'
    assert obj.zero_variance_filter is True
    assert obj.x_transforms == ()


def test_from_json_restores_computed_attrs():
    # Metric is a plain (non-frozen) class with no init params — ideal for
    # testing setattr restoration without heavyweight experiment setup.
    data = {'__class__': 'experiments.Metric', 'run_id_': 'test__20260425-000000-abcd'}
    from experiments import Metric
    obj = from_json(data)
    assert isinstance(obj, Metric)
    assert obj.run_id_ == 'test__20260425-000000-abcd'
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_util.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'util'`

- [ ] **Step 3: Create `experiments/util.py`**

```python
import json
import os
import tempfile
import warnings
import inspect
import importlib

import numpy as np


def to_json(obj, include_computed=False):
    """Serialise obj to a JSON-native Python object.

    include_computed controls which trailing-underscore attributes are appended
    to Transparent Class output: False (none), True (all in obj.__dict__), or
    list[str] (named subset). Does not propagate into recursive calls.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_json(item) for item in obj]
    if isinstance(obj, tuple):
        return {'__tuple__': [to_json(item) for item in obj]}
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if _is_named_import(obj):
        return {'__import__': f'{obj.__module__}.{obj.__qualname__}'}
    cls = type(obj)
    cls_ref = f'{cls.__module__}.{cls.__qualname__}'
    params = {}
    for name, val in _init_params(obj).items():
        params[name] = to_json(val)
    computed = {k: to_json(getattr(obj, k)) for k in _computed_keys(obj, include_computed)}
    return {'__class__': cls_ref, **params, **computed}


def from_json(data):
    """Reconstruct a Python object from the output of json.loads().

    Input is always one of None, bool, int, float, str, list, or dict.
    """
    if data is None or isinstance(data, (bool, int, float, str)):
        return data
    if isinstance(data, list):
        return [from_json(item) for item in data]
    if isinstance(data, dict):
        if '__tuple__' in data:
            return tuple(from_json(item) for item in data['__tuple__'])
        if '__import__' in data:
            module, _, name = data['__import__'].rpartition('.')
            return getattr(importlib.import_module(module), name)
        if '__class__' in data:
            module, _, name = data['__class__'].rpartition('.')
            cls = getattr(importlib.import_module(module), name)
            init_kwargs = {k: from_json(v) for k, v in data.items()
                          if k != '__class__' and not k.endswith('_')}
            obj = cls(**init_kwargs)
            for k, v in data.items():
                if k.endswith('_'):
                    setattr(obj, k, from_json(v))
            return obj
        return {k: from_json(v) for k, v in data.items()}


def save_json(path, data, indent=2):
    """Write data as JSON to path atomically via tempfile + os.replace.

    Creates parent directories as needed. indent=None produces compact
    single-line output. On write failure emits UserWarning rather than raising.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        warnings.warn(f'JSON write failed for {path}: {e}')


def load_json(path, default=None):
    """Read and deserialise a JSON file via from_json; return default if missing."""
    try:
        with open(path) as f:
            return from_json(json.load(f))
    except FileNotFoundError:
        return default


def _is_named_import(obj):
    try:
        return getattr(importlib.import_module(obj.__module__), obj.__qualname__) is obj
    except Exception:
        return False


def _init_params(obj):
    cls = type(obj)
    result = {}
    VAR = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    for name, p in inspect.signature(cls.__init__).parameters.items():
        if name == 'self' or p.kind in VAR:
            continue
        try:
            result[name] = getattr(obj, name)
        except AttributeError:
            warnings.warn(
                f'{cls.__name__}.{name}: init parameter not found as attribute; '
                'omitted from serialisation')
    return result


def _computed_keys(obj, include_computed):
    all_keys = [k for k in obj.__dict__ if k.endswith('_')]
    if include_computed is False:
        return []
    if include_computed is True:
        return all_keys
    return [k for k in include_computed if k in obj.__dict__]
```

- [ ] **Step 4: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_util.py -v
```

Expected: all PASS. If `test_to_json_include_computed_false_excludes_underscore_attrs` fails because `object.__setattr__` is blocked by the frozen dataclass, replace the test body with:

```python
def test_to_json_include_computed_false_excludes_underscore_attrs():
    from experiments import Metric
    m = Metric()
    m.computed_ = 99
    assert 'computed_' not in to_json(m, include_computed=False)
```

- [ ] **Step 5: Commit**

```bash
git add experiments/util.py tests/test_util.py
git commit -m "feat: add util.py with to_json, from_json, save_json, load_json"
```

---

### Task 2: End-to-end experiment roundtrip tests

These verify the framework handles the actual object graphs from the two production notebooks before `run()` is called. Both tests check idempotency: `to_json(from_json(to_json(exp))) == to_json(exp)`.

**Files:**
- Modify: `tests/test_util.py`

- [ ] **Step 1: Add end-to-end roundtrip tests**

Append to `tests/test_util.py`:

```python
# ── end-to-end experiment roundtrips ─────────────────────────────────────────

def test_experiment_roundtrip(tmp_path):
    # Representative of real_data.ipynb: OHE transform, two estimators,
    # explicit est_names to avoid memory-address repr instability.
    from fastridge import RidgeEM, RidgeLOOCV
    from experiments import Experiment, prediction_r2, prediction_mean_squared_error
    from problems import EmpiricalDataProblem, OneHotEncodeCategories

    prob_plain = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    prob_ohe = EmpiricalDataProblem(
        'abalone', 'Rings',
        x_transforms=(OneHotEncodeCategories(),),
        zero_variance_filter=True)
    estimators = [RidgeEM(), RidgeLOOCV(alphas=100)]
    exp = Experiment(
        [prob_plain, prob_ohe], estimators,
        reps=10, ns=[[309], [3177]], seed=1, verbose=False,
        est_names=['EM', 'CV'],
        stats=[prediction_r2, prediction_mean_squared_error])

    path = str(tmp_path / 'exp.json')
    save_json(path, to_json(exp))
    reconstructed = load_json(path)
    assert to_json(reconstructed) == to_json(exp)


def test_experiment_with_per_series_seeding_roundtrip(tmp_path):
    # Representative of real_data_neurips2023.ipynb: t2=False, logspace alphas.
    from fastridge import RidgeEM, RidgeLOOCV
    from experiments import ExperimentWithPerSeriesSeeding, prediction_r2
    from problems import EmpiricalDataProblem, NEURIPS2023_TRAIN_SIZES

    problems = [EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)]
    estimators = [RidgeEM(t2=False), RidgeLOOCV(alphas=np.logspace(-10, 10, 11, base=10))]
    exp = ExperimentWithPerSeriesSeeding(
        problems, estimators, reps=100,
        ns=[[NEURIPS2023_TRAIN_SIZES['diabetes']]], seed=1, verbose=False,
        est_names=['EM', 'CV_fix'],
        stats=[prediction_r2])

    path = str(tmp_path / 'exp_series.json')
    save_json(path, to_json(exp))
    reconstructed = load_json(path)
    assert to_json(reconstructed) == to_json(exp)
```

- [ ] **Step 2: Run tests to verify they pass**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_util.py::test_experiment_roundtrip tests/test_util.py::test_experiment_with_per_series_seeding_roundtrip -v
```

Expected: both PASS. Common failure modes and fixes:
- `AttributeError` on `est_names` or `stats` in `_init_params`: the attribute name in `__init__` matches the stored attribute — verify `Experiment.__init__` stores `self.est_names` and `self.stats`.
- `FrozenInstanceError` on `EmpiricalDataProblem`: means `from_json` tried to `setattr` a `_`-suffix key that appeared in the serialised data — verify no `_` keys appear in `to_json(prob)` output.

- [ ] **Step 3: Run full test suite to check no regressions**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_util.py tests/test_experiments.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_util.py
git commit -m "test: add end-to-end JSON roundtrip tests for Experiment and ExperimentWithPerSeriesSeeding"
```

---

### Task 3: Wire up `experiments.py`

Replace the three private JSON helpers with framework calls. Add five new computed attributes to both runner classes' `run()` methods, and write the run file twice per run using `to_json`.

**Files:**
- Modify: `experiments/experiments.py`

The existing `experiments.py` has (at the top):

```python
# lines 42-46: _load_metric_file
# lines 49-59: _save_json
# lines 74-86: _write_run_file
```

- [ ] **Step 1: Add import and constants; remove old helpers**

Replace the block from line 1 imports through `_write_run_file` (line 86). The new imports section and helpers section:

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

from util import to_json, from_json, save_json, load_json


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

_METRIC_FILE_DEFAULT = {'computations': [], 'retrievals': []}

_RUN_FILE_STATE = [
    'run_id_', 'timestamp_start_', 'timestamp_end_', 'environment_',
    'problem_keys_', 'estimator_keys_',
]
```

Remove `_load_metric_file`, `_save_json`, and `_write_run_file` entirely (they are replaced by `load_json`, `save_json`+`to_json`, and inline calls respectively).

Keep `_cache_key` and `_make_run_id` unchanged.

- [ ] **Step 2: Update `Experiment._all_stats_in_trial_cache`**

Replace:
```python
_load_metric_file(os.path.join(d, str(stat) + '.json'))['computations']
```
With:
```python
load_json(os.path.join(d, str(stat) + '.json'),
          default=_METRIC_FILE_DEFAULT)['computations']
```

- [ ] **Step 3: Update `Experiment._retrieve_trial`**

Replace both `_load_metric_file` and `_save_json` calls:

```python
def _retrieve_trial(self, prob_idx, n_idx, est_idx, rep_idx):
    d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
    for stat in self.stats:
        path = os.path.join(d, str(stat) + '.json')
        data = load_json(path, default=_METRIC_FILE_DEFAULT)
        mean_val = float(np.mean([c['value'] for c in data['computations']]))
        self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = mean_val
        data['retrievals'].append({'value': mean_val, 'run_id': self.run_id_})
        save_json(path, to_json(data))
        msg = stat.warn_retrieval(data['computations'])
        if msg:
            warnings.warn(msg)
```

- [ ] **Step 4: Update `Experiment._write_trial`**

```python
def _write_trial(self, prob_idx, n_idx, est_idx, rep_idx):
    d = self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx)
    for stat in self.stats:
        path = os.path.join(d, str(stat) + '.json')
        data = load_json(path, default=_METRIC_FILE_DEFAULT)
        new_val = self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx]
        if data['computations']:
            msg = stat.warn_recompute(
                [c['value'] for c in data['computations']], new_val)
            if msg:
                warnings.warn(msg)
        data['computations'].append({'value': new_val, 'run_id': self.run_id_})
        save_json(path, to_json(data))
```

- [ ] **Step 5: Replace `Experiment.run()`**

```python
def run(self, force_recompute=False, ignore_cache=False):
    n_problems = len(self.problems)
    n_sizes = len(self.ns[0])
    n_estimators = len(self.estimators)
    for stat in self.stats:
        self.__dict__[str(stat) + '_'] = np.full(
            (self.reps, n_problems, n_sizes, n_estimators), np.nan)
    self.run_id_ = _make_run_id(type(self).__name__)
    self.timestamp_start_ = datetime.datetime.now().isoformat()
    self.environment_ = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
    }
    self.problem_keys_ = [_cache_key(p) for p in self.problems]
    self.estimator_keys_ = [_cache_key(e) for e in self.estimators]
    if not ignore_cache:
        save_json(
            os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json'),
            to_json(self, include_computed=_RUN_FILE_STATE))
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
    self.timestamp_end_ = datetime.datetime.now().isoformat()
    if not ignore_cache:
        save_json(
            os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json'),
            to_json(self, include_computed=_RUN_FILE_STATE))
    return self
```

- [ ] **Step 6: Update `ExperimentWithPerSeriesSeeding._all_stats_in_series_cache`**

Replace `_load_metric_file` with `load_json`:

```python
def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    return all(
        load_json(os.path.join(d, str(stat) + '.json'),
                  default=_METRIC_FILE_DEFAULT)['computations']
        for stat in self.stats
    )
```

- [ ] **Step 7: Update `ExperimentWithPerSeriesSeeding._retrieve_series`**

```python
def _retrieve_series(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    for stat in self.stats:
        path = os.path.join(d, str(stat) + '.json')
        data = load_json(path, default=_METRIC_FILE_DEFAULT)
        values = [np.asarray(c['value']) for c in data['computations']]
        mean_val = np.mean(values, axis=0)
        self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
        serialisable = mean_val.tolist() if hasattr(mean_val, 'tolist') else float(mean_val)
        data['retrievals'].append({'value': serialisable, 'run_id': self.run_id_})
        save_json(path, to_json(data))
        msg = stat.warn_retrieval(data['computations'])
        if msg:
            warnings.warn(msg)
    if self.verbose:
        for _ in range(self.reps):
            print('.', end='', flush=True)
```

- [ ] **Step 8: Update `ExperimentWithPerSeriesSeeding._write_series`**

```python
def _write_series(self, prob_idx, n_idx, est_idx):
    d = self._series_cache_dir(prob_idx, n_idx, est_idx)
    for stat in self.stats:
        path = os.path.join(d, str(stat) + '.json')
        data = load_json(path, default=_METRIC_FILE_DEFAULT)
        new_values = self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx]
        if data['computations']:
            msg = stat.warn_recompute(
                [c['value'] for c in data['computations']], new_values)
            if msg:
                warnings.warn(msg)
        data['computations'].append({'value': new_values.tolist(), 'run_id': self.run_id_})
        save_json(path, to_json(data))
```

- [ ] **Step 9: Replace `ExperimentWithPerSeriesSeeding.run()`**

```python
def run(self, force_recompute=False, ignore_cache=False):
    n_problems = len(self.problems)
    n_sizes = len(self.ns[0])
    n_estimators = len(self.estimators)
    for stat in self.stats:
        self.__dict__[str(stat) + '_'] = np.full(
            (self.reps, n_problems, n_sizes, n_estimators), np.nan)
    self.run_id_ = _make_run_id(type(self).__name__)
    self.timestamp_start_ = datetime.datetime.now().isoformat()
    self.environment_ = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
    }
    self.problem_keys_ = [_cache_key(p) for p in self.problems]
    self.estimator_keys_ = [_cache_key(e) for e in self.estimators]
    if not ignore_cache:
        save_json(
            os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json'),
            to_json(self, include_computed=_RUN_FILE_STATE))
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
    self.timestamp_end_ = datetime.datetime.now().isoformat()
    if not ignore_cache:
        save_json(
            os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json'),
            to_json(self, include_computed=_RUN_FILE_STATE))
    return self
```

- [ ] **Step 10: Run the experiment test suite**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py -v
```

Expected: some tests FAIL — `test_load_metric_file_missing`, `test_save_load_metric_file_roundtrip`, and `test_write_run_file` will fail because they import removed private names. Fix those in Task 4.

- [ ] **Step 11: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: wire util.py into experiments.py; replace JSON helpers with framework calls"
```

---

### Task 4: Update `tests/test_experiments.py`

**Files:**
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Update imports and stale tests**

Find the import block (currently around line 87):

```python
from experiments import (_load_metric_file, _save_json,
                         _make_run_id, _write_run_file)
```

Replace with:

```python
from experiments import _make_run_id
from util import save_json, load_json
```

- [ ] **Step 2: Replace `test_load_metric_file_missing`**

Old:
```python
def test_load_metric_file_missing(tmp_path):
    data = _load_metric_file(str(tmp_path / 'missing.json'))
    assert data == {'computations': [], 'retrievals': []}
```

New:
```python
def test_load_metric_file_missing(tmp_path):
    data = load_json(str(tmp_path / 'missing.json'),
                     default={'computations': [], 'retrievals': []})
    assert data == {'computations': [], 'retrievals': []}
```

- [ ] **Step 3: Replace `test_save_load_metric_file_roundtrip`**

Old:
```python
def test_save_load_metric_file_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'm.json')
    original = {'computations': [{'value': 0.85, 'run_id': 'abc'}], 'retrievals': []}
    _save_json(path, original)
    assert _load_metric_file(path) == original
```

New:
```python
def test_save_load_metric_file_roundtrip(tmp_path):
    path = str(tmp_path / 'sub' / 'm.json')
    original = {'computations': [{'value': 0.85, 'run_id': 'abc'}], 'retrievals': []}
    save_json(path, original)
    assert load_json(path) == original
```

- [ ] **Step 4: Replace `test_write_run_file` with a run-file content test**

Remove:
```python
def test_write_run_file(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    run_id = _make_run_id('Experiment')
    _write_run_file(run_id, {'problems': []},
                    {'trials_computed': 0, 'trials_retrieved': 0})
    path = os.path.join(str(tmp_path), 'runs', f'{run_id}.json')
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data['run_id'] == run_id
    assert 'python' in data['environment']
```

Replace with:
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
    assert data['__class__'] == 'experiments.Experiment'
    assert data['run_id_'].startswith('Experiment__')
    assert data['timestamp_start_'] is not None
    assert data['timestamp_end_'] is not None
    assert 'python' in data['environment_']
    assert data['problems'][0]['__class__'] == 'problems.EmpiricalDataProblem'
    assert data['reps'] == 2
```

- [ ] **Step 5: Run full test suite**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_experiments.py tests/test_util.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run complete suite including doctests and notebooks**

```
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest -v
```

Expected: all PASS. Notebook tests may be slow — if they time out locally, they will be covered by CI.

- [ ] **Step 7: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: update test_experiments.py for util-based JSON I/O and new run file structure"
```
