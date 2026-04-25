# Unified JSON Serialization Design

## Motivation and Goal

The current code for writing experiment run files seems brittle and has a large footprint 
in the content modules (`problems.py`, `experiments.py`, `fastridge.py`). This refactoring
replaces this code through the simple invocation of a newly introduction JSON serialisation
framework.

The goal is a single recursive protocol that covers all required object types for experiment
run files:

1. Experiment classes themselves
2. Problem definitions and their contained objects (transforms, etc.)
3. Estimators (sklearn compatible)

In particular, we are aiming for a future proof design that can flexible serialise unfitted
estimators (current use case) and unfitted estimators (future use case for result serialisation).

The JSON framework should:

1. Have minimal footprint in the content modules, i.e., be introduced in a new module
`utils.py` (or similar)
2. Be realised via recursive functions with a minimal number of cases required to cover
the above-mentioned object types with high readability (it may introduce reasonable 
conventions / restrictions to the content objects in terms of type coercions).
3. And be otherwise as general as possible (individual cases should be widened to the
simplest / most general boundaries that are still self-evidently correct).

Run file writing should then be a simple one line invocation of the serialisation
framework.

---

## Abstractions

To required object types are covered via the following abstract types:

1. **Numpy `ArrayLike`** containing all primitive Python types as well as lists and tuples.
2. **Named imports** that can be reconstructed via a named import from a Python module in
scope of the module loader. This covers `numpy.ufunc` that appear as transforms.
3. **Transparent Classes** where the parameters of the init method correspond to attributes
sufficient for reconstructing an equivalent object.

The equivalence in Case 3 excludes potentially **computed attributes** (marked with a trailing
underscore `_`) as used in sklearn estimators. However, these can optionally also be serialised
and deserialised with the framework.

---

## Protocol

### Serialisation: `to_json(obj, include_computed=False)`

Recursive function in `util.py`. The `include_computed` parameter controls which computed
(`_`-suffix) attributes are appended flat to the spec output for Transparent Class objects:

- `False` (default): spec only
- `True`: all `_`-suffix attributes present in `obj.__dict__`
- `list[str]`: only the named attributes (e.g. `['run_id_', 'timestamp_start_']`)

`include_computed` does not propagate into recursive calls — nested objects are always
serialised spec-only regardless of the top-level setting.

**Dispatch table:**

| Condition | Output |
|-----------|--------|
| `obj is None` or `isinstance(obj, (bool, int, float, str))` | `obj` |
| `isinstance(obj, np.integer)` | `int(obj)` |
| `isinstance(obj, np.floating)` | `float(obj)` |
| `isinstance(obj, np.ndarray)` | `obj.tolist()` |
| `isinstance(obj, list)` | `[to_json(item) for item in obj]` |
| `isinstance(obj, tuple)` | `{"__tuple__": [to_json(item) for item in obj]}` |
| `_is_named_import(obj)` | `{"__import__": f"{obj.__module__}.{obj.__qualname__}"}` |
| default (Transparent Class) | `{"__class__": cls_ref, **{k: to_json(v) for k, v in _init_params(obj).items()}, **{k: to_json(getattr(obj, k)) for k in _computed_keys(obj, include_computed)}}` |

If the default case cannot enumerate init params or a param is not stored as a same-named
attribute, `to_json` raises `TypeError`.

Throughout:
- `cls_ref = f"{type(obj).__module__}.{type(obj).__qualname__}"`
- `_init_params(obj)`: `{name: getattr(obj, name) for name, p in inspect.signature(type(obj).__init__).parameters.items() if name != 'self' and p.kind not in (VAR_POSITIONAL, VAR_KEYWORD)}`
- `_computed_keys(obj, include_computed)`: the selected `_`-suffix keys from `obj.__dict__`
- `_is_named_import(obj)`: `getattr(importlib.import_module(obj.__module__), obj.__qualname__) is obj`, returning `False` on any exception; verifies reconstructability rather than relying on structural heuristics

### Deserialisation: `from_json(data)`

| Input | Reconstruction |
|-------|----------------|
| `None`, `bool`, `int`, `float`, `str` | `data` |
| `list` | `[from_json(item) for item in data]` |
| `dict` with `"__tuple__"` | `tuple(from_json(item) for item in data["__tuple__"])` |
| `dict` with `"__import__"` | `getattr(importlib.import_module(module), name)` where `module, _, name = data["__import__"].rpartition(".")` |
| `dict` with `"__class__"` | `cls(**{k: from_json(v) for k in data if k != '__class__' and not k.endswith('_')})` then `setattr(obj, k, from_json(data[k]))` for `_`-suffix keys |
| plain `dict` | `{k: from_json(v) for k, v in data.items()}` |

For `__class__`: `module, _, name = data["__class__"].rpartition(".")`, `cls = getattr(importlib.import_module(module), name)`. Init-time normalisation is idempotent because `to_json` stores the already-stored attribute value, not the original constructor argument.

---

## Behaviour for Relevant Types

### Scalars and Numpy Scalars

Python primitives pass through as-is. `np.integer` and `np.floating` are coerced to Python
`int` and `float` respectively before writing.

```json
42
3.14
"diabetes"
true
null
```

### Numpy Arrays

`np.ndarray.tolist()` recursively produces nested JSON arrays of Python scalars.

```json
[[309], [354]]
```

### Lists and Tuples

Lists produce JSON arrays; tuples produce a `__tuple__` wrapper to preserve the type
distinction on reconstruction.

```json
[{"__class__": "problems.EmpiricalDataProblem", "...": "..."}, "..."]
{"__tuple__": [{"__import__": "numpy.log"}]}
```

### Named Functions and Ufuncs

Any object for which `_is_named_import` returns `True` — in practice named functions and
numpy ufuncs used as problem transforms.

```json
{"__import__": "numpy.log"}
```

### Problem Instances

`EmpiricalDataProblem` is a Transparent Class; `inspect.signature(__init__)` gives dataset,
target, and transform fields. The `x_transforms` tuple becomes a `__tuple__` wrapper.

```json
{
  "__class__": "problems.EmpiricalDataProblem",
  "dataset": "diabetes",
  "target": "target",
  "zero_variance_filter": true,
  "x_transforms": {"__tuple__": [{"__import__": "numpy.log"}]}
}
```

### Estimator Instances

`RidgeEM` and `RidgeLOOCV` are Transparent Classes; `inspect.signature(__init__)` gives
the hyperparameter spec. Computed attributes (`coef_`, `alpha_`, etc.) appear only when
`include_computed` selects them.

```json
{
  "__class__": "fastridge.RidgeEM",
  "fit_intercept": true,
  "normalize": true,
  "epsilon": 1e-06
}
```

### Metric Instances

`PredictionR2`, `FittingTime`, etc. are Transparent Classes with no `__init__` parameters.
Serialises to `__class__` only; reconstruction via `PredictionR2()` is functionally
equivalent since metrics are stateless.

```json
{"__class__": "experiments.PredictionR2"}
```

### Experiment Instances

`Experiment` and `ExperimentWithPerSeriesSeeding` are Transparent Classes that also carry
computed attributes set during `run()`. Run files are written twice — at the start and on
completion — using an explicit whitelist of computed attrs:

```python
_RUN_FILE_STATE = [
    'run_id_', 'timestamp_start_', 'timestamp_end_', 'environment_',
    'problem_keys_', 'estimator_keys_',
]

def _write_run_file(exp):
    path = os.path.join(CACHE_DIR, 'runs', f'{exp.run_id_}.json')
    _save_json(path, to_json(exp, include_computed=_RUN_FILE_STATE))
```

Result arrays (`prediction_r2_` etc.) are excluded — they replicate metric cache content.
`problem_keys_` and `estimator_keys_` are cache-directory links set at the start of `run()`.

```json
{
  "__class__": "experiments.Experiment",
  "problems": [{"__class__": "problems.EmpiricalDataProblem", "...": "..."}],
  "estimators": [{"__class__": "fastridge.RidgeEM", "...": "..."}],
  "reps": 10,
  "ns": [[309]],
  "seed": 1,
  "stats": [{"__class__": "experiments.PredictionR2"}],
  "verbose": false,
  "run_id_": "Experiment__20260424-143022-ab3f",
  "timestamp_start_": "2026-04-24T14:30:22.123456",
  "timestamp_end_": "2026-04-24T14:32:10.456789",
  "environment_": {"python": "3.11.0", "platform": "macOS-15.0-arm64"},
  "problem_keys_": ["abc123..."],
  "estimator_keys_": ["def456..."]
}
```

---

## Module Changes

New file `experiments/util.py`:
- `Computation` — base class providing `get_params()` via `__init__` inspection for sklearn compatibility; inherited by `Experiment` and `ExperimentWithPerSeriesSeeding`
- `to_json(obj, include_computed=False)` — recursive serialisation
- `from_json(data)` — recursive reconstruction
- `_save_json(path, data)` — atomic JSON write via tempfile + `os.replace` (moved from `experiments.py`)
- `_load_metric_file(path)` — metric file load with empty default (moved from `experiments.py`)

`experiments.py` imports `Computation`, `to_json`, `from_json`, `_save_json`, and
`_load_metric_file` from `util.py`. `problems.py` imports nothing from `util.py`.
