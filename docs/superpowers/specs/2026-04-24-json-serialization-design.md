# Unified JSON Serialization Design

## Motivation

Three object families need JSON serialization for run files, cache keys, and future fitted-model
caching: problem dataclasses, sklearn-style estimators, and experiment runners. Each family has
its own introspection API. The goal is a single recursive protocol that covers all three with
minimal footprint in the content modules (`problems.py`, `experiments.py`, `fastridge.py`).

---

## Scope

In scope:
- Spec and computed-state serialization for problems, estimators, and experiments
- Callable (function) serialization for problem transform fields
- `Computation` base class giving experiments `get_params()` via `__init__` inspection
- New `util.py` module housing the protocol and shared I/O helpers

Out of scope (planned, not implemented):
- Fitted estimator serialization (JSON or pickle)
- Metric file format changes
- Cache directory structure changes

---

## Protocol

### `to_json(obj, include_computed=False)`

Recursive function in `util.py`. Serializes the spec of any supported object. The
`include_computed` parameter controls which computed (`_`-suffix) attributes are included
alongside the spec in a flat output:

- `False` (default): spec only
- `True`: all `_`-suffix attributes from `obj.__dict__`
- `list[str]`: only the named attributes (e.g. `['run_id_', 'timestamp_start_']`)

The list form exists because not all computed attributes belong in every context: result
arrays (`prediction_r2_` etc.) replicate cache content and should be excluded from run
files; future fitted estimator attributes stored as separate files similarly should be
excluded. Callers specify exactly what they need.

**Dispatch order** (applied to `obj` for the spec portion):

| Condition | Output |
|-----------|--------|
| `None`, `bool`, `int`, `float`, `str` | `obj` (as-is) |
| `isinstance(obj, np.integer)` | `int(obj)` |
| `isinstance(obj, np.floating)` | `float(obj)` |
| `isinstance(obj, np.ndarray)` | `obj.tolist()` |
| `isinstance(obj, (list, tuple))` | `[to_json(item) for item in obj]` |
| `dataclasses.is_dataclass(obj) and not isinstance(obj, type)` | see Dataclass below |
| `hasattr(obj, 'get_params')` | see Get-params below |
| `callable(obj)` | see Callable below |
| otherwise | `TypeError` |

**Dataclass objects** (problems, transform wrappers like `PolynomialExpansion`):
```json
{
  "__class__": "problems.EmpiricalDataProblem",
  "dataset": "diabetes",
  "target": "target",
  "x_transforms": [{"__class__": "problems.PolynomialExpansion", "degree": 2}]
}
```
`__class__` is `f"{type(obj).__module__}.{type(obj).__qualname__}"`.
Fields come from `dataclasses.fields(obj)`; values are recursively serialized.
`include_computed` has no effect on dataclasses (they carry no `_`-suffix convention).

**Get-params objects** (estimators, experiments via `Computation`):

Spec-only output (`include_computed=False`):
```json
{
  "__class__": "fastridge.RidgeEM",
  "fit_intercept": true,
  "normalize": true,
  "epsilon": 1e-06
}
```

With `include_computed=['coef_', 'alpha_']`, computed attrs are merged flat:
```json
{
  "__class__": "fastridge.RidgeEM",
  "fit_intercept": true,
  "normalize": true,
  "epsilon": 1e-06,
  "coef_": [0.1, 0.2],
  "alpha_": 1.5
}
```

Params come from `obj.get_params(deep=False)`; values are recursively serialized.
Computed attrs are merged at the top level — the trailing `_` is the discriminator between
spec keys and computed keys, making a separate nesting level unnecessary.

**Callable objects** (numpy ufuncs and other importable named functions):
```json
{"__callable__": "numpy.log"}
```
Constructed as `f"{obj.__module__}.{obj.__qualname__}"`.
Lambdas and closures are explicitly unsupported: `to_json` raises `TypeError` with a message
directing the user to wrap the callable in a named dataclass instead.

### `from_json(data)`

Inverse of `to_json`. Dispatch on the shape of `data`:

| Input | Reconstruction |
|-------|----------------|
| `None`, `bool`, `int`, `float`, `str` | `data` (as-is) |
| `list` | `[from_json(item) for item in data]` |
| `dict` with `"__callable__"` | `importlib` import + `getattr` |
| `dict` with `"__class__"` | `importlib` import + `ClassName(**spec_kwargs)`, then `setattr` for `_`-suffix keys |
| plain `dict` | `{k: from_json(v) for k, v in data.items()}` |

Callable reconstruction: `"numpy.log"` →
`getattr(importlib.import_module("numpy"), "log")`.

Class reconstruction: the module and class name are split from `__class__`. The class is
imported via `importlib`. Keys not ending in `_` (excluding `__class__`) are passed to
`__init__`; keys ending in `_` are applied via `setattr` after construction. Both
dataclasses and get-params classes reconstruct via `__init__` — no `set_params()` is
needed.

Init-time normalization is idempotent: `np.atleast_2d` on an already-2D array is a no-op;
resolved defaults (e.g. `est_names` derived from estimators) round-trip correctly because
`to_json` stores the stored value, not the original argument.

---

## `Metric` and the Get-params Convention

`Metric` subclasses (`PredictionR2`, `FittingTime`, etc.) appear in the `stats` parameter of
experiment `__init__` and therefore must be serializable. They carry no parameters (all logic
is in static methods), so `get_params()` returns `{}`:

```python
class Metric:
    def get_params(self, deep=False):
        return {}
```

`to_json(prediction_r2)` → `{"__class__": "experiments.PredictionR2"}`.
`from_json({"__class__": "experiments.PredictionR2"})` → `PredictionR2()`.

Since metrics are stateless, reconstructed instances are functionally equivalent to the
module-level singletons. The `get_params` dispatch branch is reached before the `callable`
branch, so the callable path is not involved.

---

## `Computation` Base Class

A thin base class in `util.py` implementing `get_params()` via `__init__` signature
inspection, mirroring sklearn's `BaseEstimator` without inheriting from it:

```python
import inspect

class Computation:
    def get_params(self, deep=False):
        sig = inspect.signature(type(self).__init__)
        return {
            name: getattr(self, name)
            for name in sig.parameters
            if name != 'self'
        }
```

Both `Experiment` and `ExperimentWithPerSeriesSeeding` inherit from `Computation`.
No other change is required in `experiments.py` — `to_json` dispatches on `hasattr(obj,
'get_params')`, which covers both `Computation` subclasses and sklearn estimators (which
bring `get_params` from `BaseEstimator`).

The `deep=False` parameter is accepted for API compatibility but ignored: experiment params
are not themselves estimators, so deep param traversal is not applicable here.

---

## Computed State and the Run File

The `_`-suffix convention (shared with sklearn) distinguishes spec attributes (set in
`__init__`) from computed attributes (set during `run()` or `fit()`).

For experiments, `run()` sets these `_`-suffix attributes before the first run-file write:

- `run_id_` — unique identifier for this run
- `timestamp_start_` — ISO timestamp captured at the start of `run()`
- `environment_` — `{'python': sys.version.split()[0], 'platform': platform.platform()}`

Before the second write (after completion):

- `timestamp_end_` — ISO timestamp captured on completion

Result arrays (`prediction_r2_` etc.) are also `_`-suffix attrs but are deliberately
excluded from run files — they replicate cache content already stored in the metric JSON
files. `_write_run_file` uses the list form of `include_computed` to be explicit:

```python
_RUN_FILE_STATE = ['run_id_', 'timestamp_start_', 'timestamp_end_', 'environment_']

def _write_run_file(exp):
    path = os.path.join(CACHE_DIR, 'runs', f'{exp.run_id_}.json')
    _save_json(path, to_json(exp, include_computed=_RUN_FILE_STATE))
```

The `status` field (`'in_progress'` / `'completed'`) is derived from `timestamp_end_`
inside `_write_run_file` and added to the document before saving — it is not a stored
attribute.

---

## Forward Plan: Fitted Estimator Serialization

Not in scope now, but the design accommodates it. Two paths, selectable per experiment:

**JSON path**: `to_json(est, include_computed=True)` serializes all `_`-suffix attributes
flat alongside the spec. For `RidgeEM` and `RidgeLOOCV` whose fitted state is small (numpy
arrays of length `p`), this produces a fully human-readable, version-stable cache.

**Pickle path**: `save_estimator(est, path)` / `load_estimator(path)` using `pickle`.
Suitable for third-party estimators with complex fitted state (pipelines, forests, sparse
matrices).

The experiment class will expose an `estimator_cache: str` parameter (`'json'` or
`'pickle'`, defaulting to `'pickle'` for safety) to select the path per run.

---

## Module Structure

New file `experiments/util.py`:
- `Computation` — `get_params()` via `__init__` inspection
- `to_json(obj, include_computed=False)` — recursive serialization
- `from_json(data)` — recursive reconstruction
- `_save_json(path, data)` — atomic JSON write (moved from `experiments.py`)
- `_load_metric_file(path)` — metric file load with default (moved from `experiments.py`)

`experiments.py` imports from `util.py`; `problems.py` imports nothing from `util.py`
(problems are pure dataclasses; `to_json` handles them externally).

No other new modules are introduced at this stage.
