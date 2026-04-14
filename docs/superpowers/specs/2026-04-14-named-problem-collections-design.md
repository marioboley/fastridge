# Named Problem Collections Design

## Goal

Make `EmpiricalDataProblem` a hashable value object and introduce named `frozenset` collections in `problems.py`, so that problem sets can be composed, filtered by dataset dimensionality, and deduplicated without a master registry. For example:

```python
from data import DATASETS
NEURIPS2023_D3 = frozenset(
    p for p in NEURIPS2023_D2
    if DATASETS[p.dataset]['p'] < 150 and DATASETS[p.dataset]['n'] < 20000
)
```

This requires `DATASETS` entries to carry `'n'` and `'p'` metadata (raw dataset shape).

## Interface

### Hashable `EmpiricalDataProblem`

`__init__` converts `drop` and `transforms` to tuples immediately, so all stored attributes are immutable:

```python
def __init__(self, dataset, target, drop=None, nan_policy=None, transforms=None):
    self.dataset = dataset
    self.target = target
    self.drop = tuple(drop or [])
    self.nan_policy = nan_policy
    self.transforms = tuple(transforms or [])
```

`__hash__` and `__eq__` are keyed on the full definition tuple:

```python
(dataset, target, drop, nan_policy, transforms)
```

Callables in `transforms` must be hashable. Numpy ufuncs (`np.log`, `np.log1p`) satisfy this. Lambdas do not — `TypeError` is raised at hash time.

### Registry metadata in `data.py`

Each `DATASETS` entry gains optional `'n'` and `'p'` keys reflecting raw dataset shape (`df.shape[0]` and `df.shape[1]`), i.e. rows and total columns including target:

```python
'diabetes': {'sources': [...], 'n': 442, 'p': 11},
'yacht':    {'sources': [...], 'n': 308, 'p': 7},
```

`get_dataset()` warns if the actual loaded shape does not match the registered values, catching silent source drift.

Populated immediately for all NEURIPS2023 datasets. Other entries gain metadata over time.

### Named collections

Module-level `frozenset` variables in `problems.py`. No helper class or registry:

```python
import numpy as np
from problems import EmpiricalDataProblem

NEURIPS2023 = frozenset({
    EmpiricalDataProblem('diabetes', 'target'),
    EmpiricalDataProblem('yacht', 'Residuary_resistance',
                         transforms=[('Residuary_resistance', np.log)]),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows',
                         transforms=[('price', np.log)]),
    # ... remaining NEURIPS2023 datasets
})
```

Subsets are defined by filtering on `DATASETS` metadata. Datasets without `'n'`/`'p'` entries (crop, elec_devices, starlight — no source yet) are naturally excluded. `NEURIPS2023_D2` additionally excludes ribo (p=4089); `NEURIPS2023_D3` additionally excludes eye (p=201) and four large-n datasets (blog, ct_slices, tomshw, twitter):

```python
from data import DATASETS

NEURIPS2023_D2 = frozenset(
    p for p in NEURIPS2023
    if 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)
NEURIPS2023_D3 = frozenset(
    p for p in NEURIPS2023_D2
    if DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)
```

Threshold values (1000, 150, 20000) are calibrated against the populated registry during implementation. Ad-hoc subsets follow the same pattern:

```python
NEURIPS2023_SMALL = frozenset(
    p for p in NEURIPS2023 if DATASETS[p.dataset]['n'] < 1000
)
```

Future collections defined independently — identical `EmpiricalDataProblem` instances (same definition → same hash) deduplicate automatically on `|`:

```python
ALL_PROBLEMS = NEURIPS2023 | PAPER2026   # duplicates eliminated by hash
SHARED = NEURIPS2023 & PAPER2026
```

## Files

- Modify: `experiments/data.py` — add `'n'` and `'p'` to NEURIPS2023 dataset entries; add shape validation warning in `get_dataset()`
- Modify: `experiments/problems.py` — convert `drop`/`transforms` to tuple in `__init__`; add `__hash__`, `__eq__`; add `NEURIPS2023` and derived frozenset collections

## Testing

Doctests in `EmpiricalDataProblem`:
- Two independently constructed identical instances are equal and hash equal
- Two identical instances deduplicate in a `frozenset`
- A `frozenset` union of overlapping collections has correct cardinality
- Unhashable callable in `transforms` raises `TypeError` at hash time

## Future extensibility

- **Result caching:** the hash key is stable across sessions for named functions and numpy ufuncs, making it suitable as a cache key for experiment results.
- **Registry metadata growth:** `'n'` and `'p'` are the first structured metadata fields. Default target, task type, or data source provenance could follow the same pattern.
- **`PAPER2026` collections:** defined independently in a notebook or a future module; composition with `NEURIPS2023` via `|`, `&`, `-` requires no coordination.
