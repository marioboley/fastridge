# Experiment Progress Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual dots-and-newlines verbose output in both experiment runners with unified `master_bar` + `progress_bar` display, clean warning routing above the bars, and a per-run log file.

**Architecture:** `route_warnings_to(write_fn)` lives in `util.py` and is the only new public symbol. Both `Experiment.run()` and `ExperimentWithPerSeriesSeeding.run()` adopt an identical loop structure regardless of `verbose`; `verbose` only gates the per-dataset summary `mb.write()` call.

**Tech Stack:** `fastprogress` (already a dependency), `contextlib`, `warnings` stdlib.

**Spec:** `docs/superpowers/specs/2026-04-28-experiment-progress-display-design.md`

---

### Task 1: Add `route_warnings_to` to `util.py`

**Files:**
- Modify: `experiments/util.py:1-12`
- Test: `tests/test_util.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_util.py`:

```python
import contextlib, warnings
from util import route_warnings_to


def test_route_warnings_to_redirects():
    received = []
    with route_warnings_to(received.append):
        warnings.warn('hello', UserWarning)
    assert received == ['UserWarning: hello']


def test_route_warnings_to_restores_on_exit():
    orig = warnings.showwarning
    with route_warnings_to(lambda s: None):
        pass
    assert warnings.showwarning is orig


def test_route_warnings_to_restores_on_exception():
    orig = warnings.showwarning
    with contextlib.suppress(ValueError):
        with route_warnings_to(lambda s: None):
            raise ValueError
    assert warnings.showwarning is orig


def test_route_warnings_to_chains_non_default():
    chained = []
    sentinel = lambda msg, cat, fn, ln, file=None, line=None: chained.append(str(msg))
    # install a non-default handler before entering
    warnings.showwarning = sentinel
    try:
        received = []
        with route_warnings_to(received.append):
            warnings.warn('hi', UserWarning)
        assert received == ['UserWarning: hi']
        assert chained == ['hi']
    finally:
        warnings.showwarning = sentinel  # restore for other tests
        import warnings as _w
        from util import _default_showwarning
        _w.showwarning = _default_showwarning
```

- [ ] **Step 2: Run tests to confirm failure**

```
pytest tests/test_util.py::test_route_warnings_to_redirects \
       tests/test_util.py::test_route_warnings_to_restores_on_exit \
       tests/test_util.py::test_route_warnings_to_restores_on_exception \
       tests/test_util.py::test_route_warnings_to_chains_non_default -v
```

Expected: 4 failures (name not found).

- [ ] **Step 3: Implement in `util.py`**

At the top of `experiments/util.py`, after existing imports, add:

```python
import contextlib

_default_showwarning = warnings.showwarning


@contextlib.contextmanager
def route_warnings_to(write_fn):
    """Context manager that routes warnings.warn output through write_fn.

    Restores the original handler on exit, including on exceptions or
    KeyboardInterrupt.  If a non-default showwarning is already installed
    (e.g. pytest's capture handler), it is called after write_fn so both
    sinks receive the warning.

    >>> import warnings
    >>> received = []
    >>> with route_warnings_to(received.append):
    ...     warnings.warn('test', UserWarning)
    >>> received[0]
    'UserWarning: test'
    """
    orig = warnings.showwarning
    def _show(msg, cat, fn, ln, file=None, line=None):
        write_fn(f'{cat.__name__}: {msg}')
        if orig is not _default_showwarning:
            orig(msg, cat, fn, ln, file=file, line=line)
    warnings.showwarning = _show
    try:
        yield
    finally:
        warnings.showwarning = orig
```

- [ ] **Step 4: Run tests to confirm pass**

```
pytest tests/test_util.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/util.py tests/test_util.py
git commit -m "feat: add route_warnings_to context manager to util"
```

---

### Task 2: Update `Experiment.run()` in `experiments.py`

**Files:**
- Modify: `experiments/experiments.py` — `Experiment.run()` (lines ~450–494) and imports

- [ ] **Step 1: Add `master_bar` to the fastprogress import**

Change:
```python
from fastprogress.fastprogress import master_bar, progress_bar
```
(already done in a previous commit — verify it is present).

- [ ] **Step 2: Add `route_warnings_to` to the `util` import**

Change:
```python
from util import to_json, from_json, save_json, load_json, environment, route_warnings_to
```

- [ ] **Step 3: Rewrite `Experiment.run()`**

Replace the problem loop and surrounding verbose prints with:

```python
    def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = make_run_id(type(self).__name__)
        self.timestamp_start_ = datetime.datetime.now().isoformat()
        self.trials_computed_ = self.trials_retrieved_ = 0
        self.environment_ = environment()
        self.estimator_keys_ = [cache_key(est) for est in self.estimators]
        self.problem_keys_ = [cache_key(prob) for prob in self.problems]
        run_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json')
        log_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.log')
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        log = open(log_path, 'w') if not ignore_cache else None

        def _write(text):
            mb.write(text)
            if log:
                log.write(text + '\n')
                log.flush()

        mb = master_bar(range(n_problems))
        try:
            with route_warnings_to(_write):
                for prob_idx in mb:
                    dataset = self.problems[prob_idx].dataset
                    mb.main_bar.comment = dataset
                    t0 = time.time()
                    c0, r0 = self.trials_computed_, self.trials_retrieved_
                    trials = [(n_idx, est_idx, rep_idx)
                              for n_idx in range(n_sizes)
                              for est_idx in range(n_estimators)
                              for rep_idx in range(self.reps)]
                    for n_idx, est_idx, rep_idx in progress_bar(trials, parent=mb):
                        if (not force_recompute and not overwrite_cache and not ignore_cache
                                and self._all_stats_in_trial_cache(
                                    prob_idx, n_idx, est_idx, rep_idx)):
                            self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_retrieved_ += 1
                        else:
                            if overwrite_cache and not ignore_cache:
                                shutil.rmtree(
                                    self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx),
                                    ignore_errors=True)
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_computed_ += 1
                    elapsed = time.time() - t0
                    computed = self.trials_computed_ - c0
                    retrieved = self.trials_retrieved_ - r0
                    if self.verbose:
                        _write(f'{dataset}  —  {computed} computed, {retrieved} retrieved'
                               f'  ({elapsed:.1f}s)')
        finally:
            if log:
                log.close()
        self.timestamp_end_ = datetime.datetime.now().isoformat()
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        return self
```

- [ ] **Step 4: Run the full test suite**

```
pytest
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: replace dots output in Experiment with master_bar + progress_bar and run log"
```

---

### Task 3: Update `ExperimentWithPerSeriesSeeding.run()` in `neurips2023.py`

**Files:**
- Modify: `experiments/neurips2023.py` — imports, `_retrieve_series`, `_run_series`, `run()`

- [ ] **Step 1: Update imports**

Change the fastprogress import:
```python
from fastprogress.fastprogress import master_bar, progress_bar
```

Add `route_warnings_to` to the experiments import:
```python
from experiments import (default_stats, CACHE_DIR, cache_key, make_run_id,
                          RUN_FILE_STATE, empirical_default_stats, route_warnings_to)
```

- [ ] **Step 2: Remove per-rep dots from `_retrieve_series`**

Remove these lines from `_retrieve_series`:
```python
        if self.verbose:
            for _ in range(self.reps):
                print('.', end='', flush=True)
```

- [ ] **Step 3: Remove per-rep dot from `_run_series`**

Remove these lines from `_run_series`:
```python
            if self.verbose:
                print('.', end='', flush=True)
```

- [ ] **Step 4: Rewrite `ExperimentWithPerSeriesSeeding.run()`**

Replace the problem loop and surrounding verbose prints with:

```python
    def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = make_run_id(type(self).__name__)
        self.environment_ = environment()
        self.timestamp_start_ = datetime.datetime.now().isoformat()
        self.trials_computed_ = self.trials_retrieved_ = 0
        self.estimator_keys_ = [cache_key(est) for est in self.estimators]
        self.problem_keys_ = [cache_key(prob) for prob in self.problems]
        run_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json')
        log_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.log')
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        log = open(log_path, 'w') if not ignore_cache else None

        def _write(text):
            mb.write(text)
            if log:
                log.write(text + '\n')
                log.flush()

        mb = master_bar(range(n_problems))
        try:
            with route_warnings_to(_write):
                for prob_idx in mb:
                    dataset = self.problems[prob_idx].dataset
                    mb.main_bar.comment = dataset
                    t0 = time.time()
                    c0, r0 = self.trials_computed_, self.trials_retrieved_
                    series = [(n_idx, est_idx)
                              for n_idx in range(n_sizes)
                              for est_idx in range(n_estimators)]
                    for n_idx, est_idx in progress_bar(series, parent=mb):
                        if (not force_recompute and not overwrite_cache and not ignore_cache
                                and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                            self._retrieve_series(prob_idx, n_idx, est_idx)
                            self.trials_retrieved_ += self.reps
                        else:
                            if overwrite_cache and not ignore_cache:
                                shutil.rmtree(
                                    self._series_cache_dir(prob_idx, n_idx, est_idx),
                                    ignore_errors=True)
                            self._run_series(prob_idx, n_idx, est_idx)
                            if not ignore_cache:
                                self._write_series(prob_idx, n_idx, est_idx)
                            self.trials_computed_ += self.reps
                    elapsed = time.time() - t0
                    computed = self.trials_computed_ - c0
                    retrieved = self.trials_retrieved_ - r0
                    if self.verbose:
                        _write(f'{dataset}  —  {computed} computed, {retrieved} retrieved'
                               f'  ({elapsed:.1f}s)')
        finally:
            if log:
                log.close()
        self.timestamp_end_ = datetime.datetime.now().isoformat()
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        return self
```

- [ ] **Step 5: Run the full test suite**

```
pytest
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/neurips2023.py
git commit -m "feat: replace dots output in ExperimentWithPerSeriesSeeding with master_bar + progress_bar and run log"
```

---

### Task 4: Verify test for log file existence

**Files:**
- Modify: `tests/test_experiments.py`

The log file is a new artefact; add a minimal check that it is created alongside the run JSON.

- [ ] **Step 1: Add test**

```python
def test_new_experiment_log_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    log_files = [f for f in os.listdir(runs_dir) if f.endswith('.log')]
    assert len(log_files) == 1


def test_series_exp_log_file_written(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    runs_dir = os.path.join(str(tmp_path), 'runs')
    log_files = [f for f in os.listdir(runs_dir) if f.endswith('.log')]
    assert len(log_files) == 1
```

- [ ] **Step 2: Run full suite**

```
pytest
```

Expected: all pass including the two new tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_experiments.py
git commit -m "test: verify log file is written alongside run JSON for both experiment runners"
```
