# Preprocessing Pipeline Design

## Goal

Move OHE and polynomial feature expansion out of the experiment runners
(`EmpiricalDataExperiment.run()` and `run_real_data_experiments()`) and into
`EmpiricalDataProblem`, making each problem fully self-contained. Introduce
`x_transforms` and `y_transforms` as a clean, unified preprocessing interface
that replaces the existing `transforms` parameter.

---

## Motivation

Currently `EmpiricalDataProblem.get_X_y()` returns raw data; callers are
responsible for OHE and polynomial expansion. This creates several problems:

- **Result persistence ambiguity (primary).** The [result persistence design](2026-04-16-result-persistence-design.md)
  identifies the problem as a value object whose cache key determines what is
  stored on disk. If polynomial expansion is applied by the experiment runner
  rather than captured in the problem definition, two runs on the same problem
  with different `polynomial` settings would produce the same cache key and
  silently reuse incorrect cached results. Preprocessing must be part of the
  problem identity.
- **Wrong conceptual ownership.** OHE and polynomial expansion define the
  feature space — they are properties of the prediction problem, not of the
  experiment runner. Placing them in the runner is conceptually wrong regardless
  of how many callers exist.
- The existing `transforms` parameter applies column-level operations before the
  X/y split, allowing target-column transforms via `('target', fn)` tuples. This
  works but the pre-split application is an implicit contract that is easy to
  misunderstand, and it cannot express whole-matrix transforms (OHE, polynomial)
  without an awkward two-pass dispatch.

---

## Design

### `x_transforms` and `y_transforms`

Replace `transforms` with two new constructor parameters:

```python
EmpiricalDataProblem(
    dataset, target,
    drop=None, nan_policy=None,
    x_transforms=None,   # new; replaces transforms for X-side operations
    y_transforms=None,   # new; replaces transforms for target operations
)
```

Both are applied **after** the X/y split inside `get_X_y()`:

```python
X = df.drop(columns=[self.target])
y = df[self.target]
for fn in self.y_transforms:
    y = fn(y)
for fn in self.x_transforms:
    X = fn(X)
return X, y
```

**`y_transforms`**: list of callables mapping `pd.Series → pd.Series`. Plain
numpy ufuncs (`np.log`, `np.log1p`) satisfy this contract directly. Applied in
list order.

**`x_transforms`**: list of callables mapping `pd.DataFrame → pd.DataFrame`.
Applied in list order. `OneHotEncodeCategories` and `PolynomialExpansion` (see
below) satisfy this contract.

**Migration of existing `transforms`**: all current usages are target-column
transforms, e.g. `transforms=[('Residuary_resistance', np.log)]` becomes
`y_transforms=[np.log]`. The `transforms` parameter is removed with no
deprecation period — usage is fully internal.

**Per-column X-feature transforms** (the old `(col, fn)` tuple on an X column)
are not directly supported by the new interface. This can be recovered within
the new framework via a `ColumnTransform(col, fn)` value object placed in
`x_transforms`; see Future section.

### `__init__`, `__repr__`, `__eq__`, `__hash__`

`__repr__` is the canonical string identity — fully descriptive, human-readable,
and stable across sessions. It is precomputed once in `__init__` since the
object is immutable; `__eq__` and `__hash__` delegate to it directly.

```python
def __init__(self, dataset, target, drop=None, nan_policy=None,
             x_transforms=None, y_transforms=None):
    self.dataset = dataset
    self.target = target
    self.drop = tuple(drop or [])
    self.nan_policy = nan_policy
    self.x_transforms = tuple(x_transforms or [])
    self.y_transforms = tuple(y_transforms or [])
    self._repr = (
        f'EmpiricalDataProblem({self.dataset!r}, {self.target!r}'
        + (f', drop={list(self.drop)!r}' if self.drop else '')
        + (f', nan_policy={self.nan_policy!r}' if self.nan_policy else '')
        + (f', x_transforms={list(self.x_transforms)!r}' if self.x_transforms else '')
        + (f', y_transforms={list(self.y_transforms)!r}' if self.y_transforms else '')
        + ')'
    )

def __repr__(self):
    return self._repr

def __eq__(self, other):
    if not isinstance(other, EmpiricalDataProblem):
        return NotImplemented
    return self._repr == other._repr

def __hash__(self):
    return hash(self._repr)
```

`x_transforms` and `y_transforms` must contain only elements with a stable
`__repr__` — numpy ufuncs and the value objects defined below satisfy this.

The problem class carries no `cache_key()` method. The result persistence layer
in `experiments.py` derives the filesystem key from `repr(problem)` directly:
`hashlib.md5(repr(problem).encode()).hexdigest()`. `hashlib` is therefore a
top-level import in `experiments.py`, not in `problems.py`.

The `polynomial` parameter previously planned for `EmpiricalDataProblem` in the
[result persistence spec](2026-04-16-result-persistence-design.md) is
superseded by this design — polynomial degree is captured in `x_transforms` via
`PolynomialExpansion`.

---

## Value Objects

### `PolynomialExpansion`

```python
class PolynomialExpansion:
    def __init__(self, degree, max_entries=35_000_000):
        self.degree = degree
        self.max_entries = max_entries

    def __call__(self, X):
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        X_poly = pd.DataFrame(
            poly.fit_transform(X),
            columns=poly.get_feature_names_out(X.columns),
            index=X.index
        )
        n, p = X_poly.shape
        if n * p > self.max_entries:
            linear_cols = list(X.columns)
            interaction_cols = [c for c in X_poly.columns if c not in linear_cols]
            pnew = int(np.ceil(self.max_entries / n)) - len(linear_cols)
            rng = np.random.default_rng(self.degree)
            sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
            return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
        return X_poly

    def __eq__(self, other):
        return isinstance(other, PolynomialExpansion) and \
               (self.degree, self.max_entries) == (other.degree, other.max_entries)

    def __hash__(self):
        return hash((type(self).__name__, self.degree, self.max_entries))

    def __repr__(self):
        if self.max_entries == 35_000_000:
            return f'PolynomialExpansion({self.degree})'
        return f'PolynomialExpansion({self.degree}, max_entries={self.max_entries})'
```

The subsampling seed `np.random.default_rng(self.degree)` is fully deterministic
— integer arguments to `default_rng` are unaffected by `PYTHONHASHSEED`. This
fixes the existing bug where subsampling consumed from uncontrolled global numpy
state before the per-problem seed reset. Linear terms are always kept; only
interaction columns are subsampled.

### `OneHotEncodeCategories`

```python
class OneHotEncodeCategories:
    def __call__(self, X):
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if not cat_cols:
            return X
        enc = OneHotEncoder(drop='first', sparse_output=False)
        encoded = enc.fit_transform(X[cat_cols])
        return pd.concat([
            X.drop(columns=cat_cols),
            pd.DataFrame(encoded,
                         columns=enc.get_feature_names_out(cat_cols),
                         index=X.index)
        ], axis=1)

    def __eq__(self, other):
        return isinstance(other, OneHotEncodeCategories)

    def __hash__(self):
        return hash(type(self).__name__)

    def __repr__(self):
        return 'OneHotEncodeCategories()'
```

No-op when X has no categorical columns — safe to include for any problem.
Always uses `drop='first'`.

---

## Changes to Problem Sets

### `NEURIPS2023`

- Replace `transforms=[('col', fn)]` with `y_transforms=[fn]` for all problems
  that transform the target (yacht, automobile, forest).
- Add `x_transforms=[OneHotEncodeCategories()]` to all problems whose dataset
  contains categorical columns. The implementation must audit each dataset;
  `OneHotEncodeCategories` being a no-op on numeric DataFrames means it can be
  added broadly if preferred.

### `NEURIPS2023_D2` and `NEURIPS2023_D3`

These are no longer filtered views of `NEURIPS2023` — they are new problem
instances with `PolynomialExpansion` appended to `x_transforms`. A helper
function avoids repetition:

```python
def _with_polynomial(p, degree):
    return EmpiricalDataProblem(
        p.dataset, p.target, p.drop, p.nan_policy,
        x_transforms=list(p.x_transforms) + [PolynomialExpansion(degree)],
        y_transforms=list(p.y_transforms),
    )

NEURIPS2023_D2 = frozenset(
    _with_polynomial(p, 2)
    for p in NEURIPS2023
    if 'n' in DATASETS[p.dataset] and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    _with_polynomial(p, 3)
    for p in NEURIPS2023_D2
    if DATASETS[p.dataset]['p'] < 150 and DATASETS[p.dataset]['n'] < 20000
)
```

The filter criteria are unchanged. Both D2 and D3 are constructed from
`NEURIPS2023` (not from each other) so that `_with_polynomial` adds exactly the
right degree without accumulating transforms.

---

## Changes to Experiment Runners

Remove the OHE and polynomial expansion blocks from both
`EmpiricalDataExperiment.run()` and `run_real_data_experiments()`. Both callers
now receive a fully preprocessed `X` from `get_X_y()`. The `polynomial`
parameter is removed from both. Zero-variance column dropping (fold-specific,
inside the rep loop) is unchanged and remains in the experiment runners.

---

## Files

- **Modify**: `experiments/problems.py` — add `x_transforms`, `y_transforms`
  to `EmpiricalDataProblem`; update `__repr__`, `__eq__`, `__hash__`, `get_X_y()`;
  add `PolynomialExpansion` and `OneHotEncodeCategories` classes;
  update `NEURIPS2023`, `NEURIPS2023_D2`, `NEURIPS2023_D3`
- **Modify**: `experiments/experiments.py` — remove OHE and polynomial blocks
  from `EmpiricalDataExperiment.run()` and `run_real_data_experiments()`; remove
  `polynomial` parameter from both
- **Modify**: `experiments/real_data.ipynb` — add `x_transforms` to all inline
  problem definitions (OHE universally; `PolynomialExpansion` for D2/D3 lists);
  remove `polynomial=` from all `EmpiricalDataExperiment(...)` calls; update
  imports to include `OneHotEncodeCategories` and `PolynomialExpansion`
- **Modify**: `experiments/real_data_neurips2023.ipynb` — remove `polynomial=`
  from experiment cells only (problem sets are imported from `problems.py` and
  already carry the correct transforms after Tasks 3–4)

---

## Future

- **Generic sklearn wrapper**: a thin `SklearnTransform(estimator)` callable
  that calls `estimator.fit_transform(X)`, usable once sklearn transformers are
  configured with `set_output(transform='pandas')` to satisfy the DataFrame
  contract. Would allow `OneHotEncodeCategories` to be retired in favour of a
  configured `ColumnTransformer`.
- **`ColumnTransform(col, fn)`**: a value object that applies `fn` to a single
  column of X and returns the updated DataFrame, recovering the old
  `(col, fn)` tuple functionality within the new framework. Requires `fn` to be
  hashable.
