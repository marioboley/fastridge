# Parameter-Based Identity Design

## Motivation

`EmpiricalDataProblem`, `PolynomialExpansion`, and `OneHotEncodeCategories` carry
hand-written boilerplate — `__init__`, `__repr__`, `__eq__`, `__hash__`, and
`replace()` — that must be updated in multiple places whenever a parameter is added
or changed. This creates a maintenance burden and a source of subtle bugs (e.g. the
current `__repr__` omits defaults, which would silently produce stale cache keys if
a default value changed). The result-persistence design requires stable, complete
object identity for cache key derivation; this spec establishes the foundation that
makes that reliable and low-maintenance.

---

## Prerequisite

This spec must be implemented before the
[result-persistence design](2026-04-16-result-persistence-design.md), which depends
on the cache key format defined here.

---

## Two Sub-Problems

### Sub-problem 1: Cache key derivation

The cache key for any object must capture its full identity including default
parameter values, so that a change in a default automatically invalidates existing
results. `repr`-based keys are unsafe here because they omit defaults by convention.

**Solution: `joblib.hash()` uniformly for all objects** — problems, transforms, and
estimators. `joblib` is already a transitive dependency via sklearn. It hashes the
full instance state via pickle, capturing all stored attributes including defaults.
Keys are session-stable (unlike Python's built-in `hash()`). No code is required on
the classes themselves.

Cache key format (in the persistence layer):

```python
import joblib
cache_key = f'{type(obj).__name__}__{joblib.hash(obj)}'
```

Known limitation: module renames invalidate existing keys. This is accepted as a
rare event whose cost is recomputation, not data loss.

### Sub-problem 2: Object identity and maintainability

Problems and transforms are value objects — their identity is fully determined by
their constructor parameters. Estimators follow the sklearn estimator protocol.
Each class of object has an established solution.

---

## Problems and Transforms

`EmpiricalDataProblem`, `PolynomialExpansion`, and `OneHotEncodeCategories` are
converted to `@dataclass(frozen=True)`.

This auto-generates `__init__`, `__repr__`, `__eq__`, and `__hash__` from field
annotations. `dataclasses.replace()` replaces the hand-written `replace()` method.
Adding a parameter requires touching only the field list — no other methods change.

**Immutable inputs required**: callers must pass tuples, not lists, for sequence
parameters (`drop`, `x_transforms`, `y_transforms`). This eliminates any need for
`__post_init__` normalisation. Call sites in `problems.py` (`NEURIPS2023`,
`NEURIPS2023_D2`, `NEURIPS2023_D3`) and in notebooks (`real_data.ipynb`,
`real_data_neurips2023.ipynb`) are updated accordingly. This includes:
- `_OHE = (OneHotEncodeCategories(),)` instead of a list
- `drop=('G1', 'G2')` instead of `drop=['G1', 'G2']`
- `x_transforms=(PolynomialExpansion(2),)` instead of a list
- Tuple concatenation `_OHE + (PolynomialExpansion(2),)` instead of list concatenation

The default dataclass `__repr__` shows all fields including defaults:

```
EmpiricalDataProblem(dataset='diabetes', target='target', drop=(), nan_policy=None,
x_transforms=(), y_transforms=(), zero_variance_filter=False)
```

This is verbose but requires zero code and is consistent with the cache key
(both capture all parameters). The existing compact repr is removed.

`frozen=True` enforces immutability and generates `__hash__` without `unsafe_hash`.

---

## Estimators

`RidgeEM` and `RidgeLOOCV` in `fastridge.py` inherit from
`sklearn.base.BaseEstimator` and `sklearn.base.RegressorMixin`. This provides:

- `get_params(deep=True)` — constructor parameter introspection via
  `inspect.signature`, required for `sklearn.clone()`, `set_params()`, and pipeline
  integration
- `__repr__` — sklearn-style, omitting default values
- `score(X, y)` — standard R² evaluation method from `RegressorMixin`

The existing hand-written `RidgeEM.__repr__` is removed — it only exposes `epsilon`
under the wrong name `eps` and is superseded by `BaseEstimator.__repr__`.

sklearn is added as an explicit package dependency of `fastridge.py` and listed in
`setup.cfg`. It is already a dependency of the experiments module and is universally
available in the target environment.

`get_params(deep=True)` recurses into any direct parameter value that itself has
`get_params()`. Since `RidgeEM` and `RidgeLOOCV` have no nested estimator
parameters, `deep` has no observable effect today but the protocol is complete.

---

## Testing

- Remove repr doctests from `EmpiricalDataProblem` and `PolynomialExpansion` — they
  test implementation detail, not user-facing behaviour, and repr is no longer used
  for cache keys. The equality and hashability doctests (`p1 == p2`, `frozenset`)
  are user-facing and stay.
- Add a minimal `score()` doctest to `RidgeEM` documenting the new sklearn-compliant
  evaluation method.
- No changes to `ci.yml` — sklearn enters `package-test` via `install_requires` when
  the package is installed.

---

## Files

- Modify: `fastridge.py` — `RidgeEM` and `RidgeLOOCV` inherit `BaseEstimator` and
  `RegressorMixin`; remove `RidgeEM.__repr__`; add
  `from sklearn.base import BaseEstimator, RegressorMixin`
- Modify: `setup.cfg` — migrate all metadata and dependencies from `setup.py` into
  `setup.cfg` (`[metadata]` and `[options]` sections); add `scikit-learn>=1.2` to
  `install_requires`; reduce `setup.py` to `from setuptools import setup; setup()`
- Modify: `experiments/problems.py` — convert `EmpiricalDataProblem`,
  `PolynomialExpansion`, `OneHotEncodeCategories` to `@dataclass(frozen=True)`;
  remove `replace()`, `__repr__`, `__eq__`, `__hash__`; update `NEURIPS2023`,
  `NEURIPS2023_D2`, `NEURIPS2023_D3` call sites to use tuples
- Modify: `experiments/real_data.ipynb` — update all `x_transforms` and `drop`
  arguments to tuples
- Modify: `experiments/real_data_neurips2023.ipynb` — same
