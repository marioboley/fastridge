# Cache Format Improvements

## Motivation

Four issues observed during the first real-data experiment run with the new `Experiment`
runner, all addressable with focused changes to the cache helpers and `run()` methods:

1. **Run files are not invertible** — they store only hash keys, not the human-readable
   parameters needed to understand or reconstruct a run without access to the original code
2. **Cache directory names are opaque** — problem keys give no indication of dataset, making
   manual browsing of the result directory effectively useless
3. **Per-metric file granularity wastes disk** — each metric file takes at least one
   filesystem block (4 KB on APFS/HFS+) regardless of content size, amplifying 5 MB of
   actual data to ~200 MB on disk for a single notebook run
4. **Run files written at end only** — interrupted runs leave orphan cache entries that
   cannot be attributed to any run

---

## Changes

### 1. Metrics file consolidation

Replace the N per-metric JSON files at each cache leaf with a single `metrics.json`.

**Old leaf structure:**
```
<trial_dir>/
  prediction_r2.json
  prediction_mse.json
  fitting_time.json
  ...
```

**New leaf structure:**
```
<trial_dir>/
  stats.json
```

**Format:**
```json
{
  "prediction_r2": {
    "computations": [{"value": 0.851, "run_id": "Experiment__20260422-143045-a3f7"}],
    "retrievals": []
  },
  "fitting_time": {
    "computations": [{"value": 0.012, "run_id": "Experiment__20260422-143045-a3f7"}],
    "retrievals": []
  }
}
```

All JSON files (metrics and run files) are pretty-printed with `indent=2`. This is
negligible overhead relative to block size and makes files human-readable in a text editor.

**Helper changes:**

- `_load_metric_file(path)` → `_load_stats_file(dir_path)` — reads
  `<dir_path>/stats.json`; returns `{metric_name: {"computations": [], "retrievals": []},
  ...}` for all configured stats, defaulting to empty if the file or a key is absent
- `_save_metric_file(path, data)` → `_save_stats_file(dir_path, data)` — writes
  `<dir_path>/stats.json` atomically (tempfile + `os.replace`)

`_all_stats_in_*_cache` reads a single file instead of N files — simpler and faster than
before.

**Breaking change:** existing cache dirs contain per-metric files but no `stats.json`;
they are treated as cache misses. No migration is provided; orphaned directories with
old-format files are harmless.

---

### 2. Human-readable slug in cache key

The `dataset` attribute of `EmpiricalDataProblem` is included as a slug between the class
name and the hash, making cache paths browsable.

**Old format:** `EmpiricalDataProblem__8db1480a05fa91f6a3a86c57bb4f6af7`

**New format:** `EmpiricalDataProblem_diabetes__8db1480a05fa91f6a3a86c57bb4f6af7`

Updated `_cache_key`:
```python
def _cache_key(obj):
    slug = getattr(obj, 'dataset', '')
    sep = '_' if slug else ''
    return f'{type(obj).__name__}{sep}{slug}__{joblib.hash(obj)}'
```

Estimator keys are unchanged — the class name (`RidgeEM`, `RidgeLOOCV`) is already
sufficient for browsing.

**Breaking change:** existing cache dirs have old-format names and become cache misses.
This is acceptable since the metrics consolidation in Change 1 already invalidates the
existing cache.

---

### 3. Run file: class name in filename, human-readable params, written at start

#### Filename

`{ClassName}__{run_id}.json` — e.g. `Experiment__20260422-143045-a3f7.json`

The class name in the filename makes legacy runs from different runner classes
immediately distinguishable when browsing `results/runs/`.

#### Run ID

`self.run_id_` on the runner stores the full `{ClassName}__{timestamp}-{suffix}` string
(e.g. `Experiment__20260422-143045-a3f7`). This string is used in all `run_id` fields —
in the run file, and in each computation/retrieval record written to `metrics.json`. The
run file name is therefore simply `{self.run_id_}.json` with no additional construction.

`_make_run_id(class_name)` replaces `_make_run_id()`:
```python
def _make_run_id(class_name):
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f'{class_name}__{ts}-{suffix}'
```

#### Written at start and end

`run()` writes the run file **immediately** when called, with `"status": "in_progress"`,
and overwrites it at the end with `"status": "completed"` and the summary. Interrupted
runs leave a run file with `"status": "in_progress"` and `"summary": null`, making orphan
cache entries attributable.

`_write_run_file(run_id, class_name, experiment_dict, summary=None)` is called twice:

```python
# at start of run():
_write_run_file(self.run_id_, class_name, experiment_dict)

# at end of run():
_write_run_file(self.run_id_, class_name, experiment_dict, summary)
```

When `summary` is `None`: `status='in_progress'`, `timestamp_end=null`.
When `summary` is provided: `status='completed'`, `timestamp_end=<now>`.

#### Format

```json
{
  "run_id": "Experiment__20260422-143045-a3f7",
  "class": "Experiment",
  "status": "completed",
  "timestamp_start": "2026-04-22T14:30:45",
  "timestamp_end": "2026-04-22T15:05:23",
  "environment": {
    "python": "3.11.5",
    "platform": "macOS-15.3.0-arm64"
  },
  "experiment": {
    "reps": 10,
    "seed": 123,
    "ns": [[350], [500]],
    "problems": [
      {
        "key": "EmpiricalDataProblem_diabetes__8db1...",
        "class": "EmpiricalDataProblem",
        "params": {
          "dataset": "diabetes",
          "target": "target",
          "nan_policy": "raise",
          "drop": null,
          "x_transforms": "(OneHotEncodeCategories(),)",
          "zero_variance_filter": true
        }
      }
    ],
    "estimators": [
      {
        "key": "RidgeEM__3f2a...",
        "class": "RidgeEM",
        "params": {"t2": true, "epsilon": 0.001, "max_iter": 1000}
      }
    ]
  },
  "summary": {
    "trials_computed": 30,
    "trials_retrieved": 0
  }
}
```

Cache keys are included alongside params as a cross-reference safeguard — they are
re-derivable from params but having them in the run file avoids ambiguity if code changes
between runs.

#### Params extraction

**Problems** (`EmpiricalDataProblem` is a dataclass): enumerate via `dataclasses.fields()`
and include each field's value if JSON-native (`str`, `int`, `float`, `bool`, `None`),
else `repr()`. This handles `x_transforms`, `drop`, and any future complex fields without
special-casing.

```python
def _problem_params(problem):
    result = {}
    for f in dataclasses.fields(problem):
        val = getattr(problem, f.name)
        result[f.name] = val if isinstance(val, (str, int, float, bool, type(None))) else repr(val)
    return result
```

**Estimators** (sklearn): `est.get_params(deep=False)` returns a dict of constructor
params with JSON-native values. Any numpy scalars or arrays are converted via `.tolist()`.

**Runner params** (explicit): `reps`, `seed`, `ns` (`.tolist()`). Problems and estimators
are represented using the helpers above.

#### Deferred: non-frozen dataclass for runners

Making `Experiment` and `ExperimentWithPerSeriesSeeding` non-frozen dataclasses would
enable automatic field discovery via `dataclasses.fields()` and remove the hardcoded param
dicts. Deferred because `__post_init__` normalisation of `ns` (atleast_2d, broadcast) and
the mutable fitted attributes (`run_id_`, result arrays) add complexity beyond this spec's
scope. Explicit param dicts are sufficient here.

---

## Files

- Modify: `experiments/experiments.py`
  - `_cache_key` — add dataset slug
  - `_make_run_id` — add `class_name` parameter
  - `_load_metric_file` / `_save_metric_file` — replace with `_load_stats_file(dir_path)`
    / `_save_stats_file(dir_path, data)` operating on `stats.json`
  - `_write_run_file` — add `class_name` and `summary=None` parameters; write twice from
    `run()`; include human-readable experiment dict; pretty-print
  - Add `_problem_params(problem)` helper
  - `Experiment._all_stats_in_trial_cache`, `_retrieve_trial`, `_run_trial`, `_write_trial`,
    `run()` — update to use new helper signatures and `self.run_id_` full form
  - `ExperimentWithPerSeriesSeeding` — same updates
- Modify: `tests/test_experiments.py` — update cache-related tests for new file structure,
  key format, and run file format
