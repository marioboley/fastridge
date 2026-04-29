# Experiment Progress Display Design

The current verbose output — dataset name, a stream of dots, then a newline — gives no
progress indication for long-running experiments and is disrupted when `warnings.warn`
fires mid-line.  A unified progress-bar display with clean warning output and a durable
per-run log addresses both problems.

---

## Decisions

### 1. Progress bars — always on

Both `Experiment` and `ExperimentWithPerSeriesSeeding` use `master_bar` (over problems)
and `progress_bar` (over inner work units per problem) unconditionally, independent of
`verbose`.  The bars disappear after each inner loop completes; only `mb.write()` text
persists.

Inner bar granularity:
- `Experiment`: `n_sizes * n_estimators * reps` steps (one step = one trial)
- `ExperimentWithPerSeriesSeeding`: `n_sizes * n_estimators` steps (one step = one series
  of `reps` trials; getting rep-level granularity would require restructuring `_run_series`
  — deferred)

### 2. `verbose` controls only per-dataset summary lines

The only `if self.verbose:` in the loop body is the per-dataset completion message:

```python
if self.verbose:
    mb.write(f'{dataset}  —  {computed} computed, {retrieved} retrieved  ({elapsed:.1f}s)')
```

All other loop logic (bars, warning routing, log writing) is identical for both values.

### 3. `route_warnings_to(write_fn)` in `util.py`

A context manager that replaces `warnings.showwarning` for the duration of the block,
routing every `warnings.warn` call through `write_fn`.  On exit (including
`KeyboardInterrupt`) the original handler is restored via `try/finally`.

To preserve `pytest.warns` compatibility, the default `showwarning` is captured at
`util.py` import time.  When a non-default handler is already installed (e.g. pytest's
capture handler), the context manager chains through it after calling `write_fn`:

```python
_default_showwarning = warnings.showwarning   # captured at import, before any test runner

@contextlib.contextmanager
def route_warnings_to(write_fn):
    orig = warnings.showwarning
    def _show(msg, cat, fn, ln, file=None, line=None):
        write_fn(f'{cat.__name__}: {msg}')
        if orig is not _default_showwarning:   # pytest or similar — chain
            orig(msg, cat, fn, ln, file=file, line=line)
    warnings.showwarning = _show
    try:
        yield
    finally:
        warnings.showwarning = orig
```

In normal interactive use `orig is _default_showwarning`, so output goes only to
`write_fn` (no duplicate on stderr).

### 4. Per-run log file

Each call to `run()` opens a plain-text log file at
`results/runs/{run_id_}.log` alongside the existing run JSON before the problem loop
and closes it after.  The `write_fn`
passed to both `route_warnings_to` and the verbose summary `mb.write()` writes to both
the bar and the log:

```python
def _write(text):
    mb.write(text)
    log_file.write(text + '\n')
    log_file.flush()
```

`verbose=False` calls suppress the `mb.write()` for summary lines but still write them
to the log, so the log is always complete regardless of `verbose`.  Warnings always go to
both bar and log.

This replaces the need to chain to stderr: the log provides a durable, searchable record
without any double-display.

---

## File changes

### `util.py`
- Add `import contextlib` and `import warnings` (or extend existing imports)
- Save `_default_showwarning = warnings.showwarning` at module level (before any
  modification)
- Add `route_warnings_to(write_fn)` context manager (public, documented, with doctest)

### `experiments.py`
- Add `from fastprogress.fastprogress import master_bar` to existing import
- Import `route_warnings_to` from `util`
- Rewrite `Experiment.run()`:
  - Open log file; define `_write(text)` combining `mb.write` + log
  - Replace the three verbose print statements (dataset name, dot, newline) with
    `master_bar` + `progress_bar` loop and a single `if self.verbose: _write(summary)`
  - Wrap the problem loop with `route_warnings_to(_write)`
  - Close log file in a `try/finally`

### `neurips2023.py`
- Add `master_bar` to the existing `fastprogress` import
- Import `route_warnings_to` from `experiments` (already imports other symbols from there)
- Remove per-rep `print('.', ...)` from `_retrieve_series` and `_run_series`
- Rewrite `ExperimentWithPerSeriesSeeding.run()` the same way as above

### Tests
No changes needed.  All experiment tests use `verbose=False`.  `pytest.warns` is
unaffected because the chaining logic preserves pytest's handler when it is installed.

---

## Output shape

During a run (terminal):
```
UserWarning: [problem=eye, n=84, est=CV_glm] FittingTime: ...
ribo    —  3 computed, 0 retrieved  (0.3s)
eye     —  0 computed, 3 retrieved  (0.1s)
                       44%|████▍     | 1/3 [00:08<00:10]
[5/22]  ▓▓▓░░░░░░░░░░░░░░░░░░░░░░░  automobile
```

After completion, bars clear and the written lines remain.  The log file at
`results/runs/{run_id_}.log` contains the same text (summaries + warnings) regardless
of terminal state.

---

## Out of scope

- Rep-level granularity for `ExperimentWithPerSeriesSeeding` (requires restructuring
  `_run_series`)
- Structured log format (JSON lines, etc.) — plain text is sufficient for now
- `SyntheticDataExperiment` — uses a different progress mechanism (`progress_bar` over
  reps); updating it is a separate concern
