# Unified JSON Serialization Design

## Motivation

Three object families need JSON serialization for run files, cache keys, and future fitted-model
caching: problem dataclasses, sklearn-style estimators, and experiment runners. Each family has
its own introspection API. The goal is a single recursive protocol that covers all three with
minimal footprint in the content modules (`problems.py`, `experiments.py`, `fastridge.py`).

---

## Scope

In scope:
- Unfitted spec serialization for problems, estimators, and experiments
- Callable (function) serialization for problem transform fields
- `BaseExperiment` mixin giving experiments a `get_params()` implementation
- New `util.py` module housing the protocol and shared I/O helpers

Out of scope (planned, not implemented):
- Fitted estimator serialization (JSON or pickle)
- Metric file format changes
- Cache directory structure changes

---

## Protocol

### `to_json(obj)`

Recursive single-dispatch function in `util.py`. Dispatch order:

| Condition | Output |
|-----------|--------|
| `obj` is `None`, `bool`, `int`, `float`, `str` | `obj` (as-is) |
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

**Get-params objects** (estimators, experiments via `BaseExperiment`):
```json
{
  "__class__": "fastridge.RidgeEM",
  "fit_intercept": true,
  "normalize": true,
  "epsilon": 1e-06
}
```
Params come from `obj.get_params(deep=False)`; values are recursively serialized.

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
| `dict` with `"__class__"` | `importlib` import + `ClassName(**{k: from_json(v) ...})` |
| plain `dict` | `{k: from_json(v) for k, v in data.items()}` |

Callable reconstruction example: `"numpy.log"` → `getattr(importlib.import_module("numpy"), "log")`.

Class reconstruction: the module and class name are split from `__class__`. The class is imported
via `importlib` and instantiated with the remaining keys as keyword arguments. Both dataclasses and
get-params classes reconstruct via `__init__` — no `set_params()` is needed.

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

## `BaseExperiment`

A thin mixin in `util.py` implementing `get_params()` via `__init__` signature inspection,
mirroring sklearn's `BaseEstimator` without inheriting from it:

```python
import inspect

class BaseExperiment:
    def get_params(self, deep=False):
        sig = inspect.signature(type(self).__init__)
        return {
            name: getattr(self, name)
            for name in sig.parameters
            if name != 'self'
        }
```

Both `Experiment` and `ExperimentWithPerSeriesSeeding` inherit from `BaseExperiment`.
No other change is required in `experiments.py` — `to_json` dispatches on `get_params`
and the convention `self.param = param` (already followed) is all that is needed.

The `deep=False` parameter is accepted for API compatibility but ignored: experiment params
are not themselves estimators, so deep cloning is not applicable here.

---

## Spec vs State

Experiments and estimators both follow the sklearn convention that fitted attributes carry
a trailing underscore (e.g. `run_id_`, `prediction_r2_`, `coef_`). `to_json` serializes
the **spec** only (via `get_params` or dataclass fields). Callers that need to include state
(such as `_write_run_file`) gather `_`-suffix attributes separately from `obj.__dict__` and
assemble the final document themselves. This keeps `to_json` single-purpose.

Example run file structure (assembled by `_write_run_file`):
```json
{
  "run_id": "Experiment__20260424-103000-ab12",
  "class": "Experiment",
  "status": "in_progress",
  "timestamp_start": "2026-04-24T10:30:00",
  "timestamp_end": null,
  "environment": {"python": "3.11.0", "platform": "..."},
  "spec": { ... },
  "summary": null
}
```
`spec` is the output of `to_json(experiment)`. `summary` is populated on the second write.

---

## Forward Plan: Fitted Estimator Serialization

Not in scope now, but the design accommodates it cleanly. Two paths:

**JSON path**: extend a `to_json_fitted(est)` call that returns the spec dict plus a `"state"`
key containing `_`-suffix attributes serialized via `to_json`. Works for `RidgeEM` and
`RidgeLOOCV` whose fitted state is small (numpy arrays of length `p`).

**Pickle path**: `save_estimator(est, path)` / `load_estimator(path)` using `pickle`. Suitable
for third-party estimators with complex fitted state (pipelines, forests, sparse matrices).

The experiment class will expose an `estimator_cache: str` parameter (`'json'` or `'pickle'`,
defaulting to `'pickle'` for safety) to select the path per run. Both paths coexist; the
JSON path is preferred when the fitted state is known to be small and human-readable.

---

## Module Structure

New file `experiments/util.py`:
- `BaseExperiment` — `get_params()` via `__init__` inspection
- `to_json(obj)` — recursive serialization
- `from_json(data)` — recursive reconstruction
- `_save_json(path, data)` — atomic JSON write (moved from `experiments.py`)
- `_load_metric_file(path)` — metric file load with default (moved from `experiments.py`)

`experiments.py` imports from `util.py`; `problems.py` imports nothing from `util.py`
(problems are pure dataclasses; `to_json` handles them externally).

No other new modules are introduced at this stage.
