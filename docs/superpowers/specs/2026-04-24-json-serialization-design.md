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

1. **Basic Types** — Python primitives (`None`, `bool`, `int`, `float`, `str`), dicts, lists,
tuples, and their numpy scalar and array equivalents.
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
| `isinstance(obj, dict)` | `{k: to_json(v) for k, v in obj.items()}` |
| `_is_named_import(obj)` | `{"__import__": f"{obj.__module__}.{obj.__qualname__}"}` |
| default (Transparent Class) | `{"__class__": cls_ref, **{k: to_json(v) for k, v in _init_params(obj).items()}, **{k: to_json(getattr(obj, k)) for k in _computed_keys(obj, include_computed)}}` |

If a parameter from the signature is not accessible as a same-named attribute, `to_json`
emits a `UserWarning` and omits that parameter. The resulting JSON will fail on
reconstruction, which is the intended failure point.

Throughout:
- `cls_ref = f"{type(obj).__module__}.{type(obj).__qualname__}"`
- `_init_params(obj)`: `{name: getattr(obj, name) for name, p in inspect.signature(type(obj).__init__).parameters.items() if name != 'self' and p.kind not in (VAR_POSITIONAL, VAR_KEYWORD)}`
- `_computed_keys(obj, include_computed)`: the selected `_`-suffix keys from `obj.__dict__`
- `_is_named_import(obj)`: `getattr(importlib.import_module(obj.__module__), obj.__qualname__) is obj`, returning `False` on any exception; verifies reconstructability rather than relying on structural heuristics

### Deserialisation: `from_json(data)`

`from_json` operates on the Python object returned by `json.loads()`. Its input is therefore
always one of `None`, `bool`, `int`, `float`, `str`, `list`, or `dict` — the types produced
by the stdlib JSON decoder. Numpy types never appear as input.

| Input | Reconstruction |
|-------|----------------|
| `None`, `bool`, `int`, `float`, `str` | `data` (as returned by `json.loads`) |
| `list` | `[from_json(item) for item in data]` |
| `dict` with `"__tuple__"` | `tuple(from_json(item) for item in data["__tuple__"])` |
| `dict` with `"__import__"` | `getattr(importlib.import_module(module), name)` where `module, _, name = data["__import__"].rpartition(".")` |
| `dict` with `"__class__"` | `cls(**{k: from_json(v) for k in data if k != '__class__' and not k.endswith('_')})` then `setattr(obj, k, from_json(data[k]))` for `_`-suffix keys |
| plain `dict` | `{k: from_json(v) for k, v in data.items()}` |

For `__class__`: `module, _, name = data["__class__"].rpartition(".")`, `cls = getattr(importlib.import_module(module), name)`. Init-time normalisation is idempotent because `to_json` stores the already-stored attribute value. Reconstruction to numpy types (e.g. `np.ndarray` from a nested list) is the constructor's responsibility, not the framework's.

---

## Behaviour for Members of Experiment Runners

Each subsection shows the full roundtrip: Python input to `to_json`, the JSON text written
to disk, and the Python value returned by `from_json(json.loads(...))`.

### Scalars, Numpy Scalars, and Numpy Arrays

| Python in | JSON | Python out |
|-----------|------|------------|
| `42` | `42` | `42` (`int`) |
| `3.14` | `3.14` | `3.14` (`float`) |
| `"diabetes"` | `"diabetes"` | `"diabetes"` |
| `True` | `true` | `True` |
| `None` | `null` | `None` |
| `np.int64(42)` | `42` | `42` (`int`, not `np.int64`) |
| `np.float64(3.14)` | `3.14` | `3.14` (`float`, not `np.float64`) |
| `np.array([[309],[354]])` | `[[309],[354]]` | `[[309],[354]]` (`list`, not `ndarray`) |

Numpy types are coerced to Python primitives and lists on serialisation and are not
restored by the framework. Constructors that require numpy types must convert internally
(e.g. `Experiment.__init__` calls `np.atleast_2d(ns)` on its `ns` argument).

### Lists and Tuples

| Python in | JSON | Python out |
|-----------|------|------------|
| `[a, b]` | `[to_json(a), to_json(b)]` | `[from_json(a), from_json(b)]` |
| `(a, b)` | `{"__tuple__": [to_json(a), to_json(b)]}` | `(from_json(a), from_json(b))` |

### Named Functions and Ufuncs

**In:** `numpy.log`

**JSON:** `{"__import__": "numpy.log"}`

**Out:** `numpy.log` (the same object, recovered via `importlib`)

### Problem Instances

**In:** `EmpiricalDataProblem('diabetes', 'target', True, (numpy.log,))`

**JSON:**
```json
{
  "__class__": "problems.EmpiricalDataProblem",
  "dataset": "diabetes",
  "target": "target",
  "zero_variance_filter": true,
  "x_transforms": {"__tuple__": [{"__import__": "numpy.log"}]}
}
```

**Out:** `EmpiricalDataProblem('diabetes', 'target', True, (numpy.log,))`

### Estimator Instances

**In:** `RidgeEM(fit_intercept=True, normalize=True, epsilon=1e-6)`

**JSON:**
```json
{
  "__class__": "fastridge.RidgeEM",
  "fit_intercept": true,
  "normalize": true,
  "epsilon": 1e-06
}
```

**Out:** `RidgeEM(fit_intercept=True, normalize=True, epsilon=1e-6)`

### Metric Instances

**In:** `PredictionR2()`

**JSON:** `{"__class__": "experiments.PredictionR2"}`

**Out:** `PredictionR2()` (equivalent stateless instance)

## Changes to Experiment Runners

`Experiment` and `ExperimentWithPerSeriesSeeding` are already Transparent Classes. 
Non-constructor parameters relevant for run files are added as computed fields
(as is already the case for `run_id_`).
Then run files are written twice, at the start and on completion of `run` using
an explicit whitelist:

```python
_RUN_FILE_STATE = [
    'run_id_', 'timestamp_start_', 'timestamp_end_', 'environment_',
    'problem_keys_', 'estimator_keys_',
]

def _write_run_file(exp):
    path = os.path.join(CACHE_DIR, 'runs', f'{exp.run_id_}.json')
    save_json(path, to_json(exp, include_computed=_RUN_FILE_STATE))
```

Result arrays (`prediction_r2_` etc.) are excluded — they replicate metric cache content.
`problem_keys_` and `estimator_keys_` are cache-directory links set at the start of `run()`.
`ns` round-trips via `np.atleast_2d` in `__init__`.

**In:** `Experiment([prob], [est], reps=10, ns=np.array([[309]]), seed=1, stats=[prediction_r2], verbose=False)` after `run()`

**JSON:**
```json
{
  "__class__": "experiments.Experiment",
  "problems": [{"__class__": "problems.EmpiricalDataProblem", "dataset": "diabetes", "...": "..."}],
  "estimators": [{"__class__": "fastridge.RidgeEM", "fit_intercept": true, "...": "..."}],
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

**Out:** `Experiment([prob], [est], reps=10, ns=np.array([[309]]), seed=1, stats=[PredictionR2()], verbose=False)` with `run_id_`, timestamps, environment, and keys restored via `setattr`

---

## Module Changes

New file `experiments/util.py`:
- `to_json(obj, include_computed=False)` — recursive serialisation; returns a JSON-native Python object
- `from_json(data)` — recursive reconstruction
- `save_json(path, data, indent=2)` — atomic JSON write via tempfile + `os.replace` (moved from `experiments.py`); `indent=None` produces compact single-line output
- `load_json(path, default=None)` — read and deserialise a JSON file via `from_json`; return `default` if the file does not exist

`experiments.py` imports `to_json`, `from_json`, `save_json`, and
`load_json` from `util.py`. `problems.py` imports nothing from `util.py`.
