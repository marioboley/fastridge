# Named Problem Collections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `EmpiricalDataProblem` a hashable value object, add raw shape metadata to the dataset registry, and define named `frozenset` collections (`NEURIPS2023`, `NEURIPS2023_D2`, `NEURIPS2023_D3`) in `problems.py`.

**Architecture:** Three sequential changes — registry metadata first (data.py), then hashable `EmpiricalDataProblem` (problems.py), then named collections (problems.py). Each is independently committable. The named collections depend on both prior changes.

**Tech Stack:** `experiments/data.py`, `experiments/problems.py`, pytest doctest-modules.

---

## Files

- Modify: `experiments/data.py` — add `'n'`/`'p'` to DATASETS entries; add shape warning in `get_dataset()`
- Modify: `experiments/problems.py` — tuple conversion in `__init__`; `__hash__`/`__eq__`; `NEURIPS2023` and derived collections

---

### Task 1: Add shape metadata and validation to data.py

**Files:** `experiments/data.py`

- [ ] **Step 1: Add `'n'` and `'p'` to all currently-cached DATASETS entries**

In `experiments/data.py`, update every entry in `DATASETS` that has known dimensions (from `datasets/<name>.csv`). Entries without a source (crop, elec_devices, starlight) are left without metadata — the filter criteria in Task 3 will naturally exclude them.

Replace the `DATASETS` dict (lines 214–292) with the version below, which adds `'n'` and `'p'` to all cached datasets. Only the key–value pairs change; source callables are unchanged:

```python
DATASETS = {
    'abalone':          {'sources': [from_ucimlrepo(1)],
                         'n': 4177, 'p': 9},
    'autompg':          {'sources': [
        from_ucimlrepo(9),
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/9/auto%2Bmpg.zip',
            'auto-mpg.data',
            sep=r'\s+', header=None, na_values='?',
            names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                   'acceleration', 'model_year', 'origin', 'car_name'],
        ),
        # TODO: verify whether from_ucimlrepo includes car_name in features; if not,
        # sources are inconsistent and car_name should be dropped at source level.
        # Also: ucimlrepo has been unavailable frequently — review all ucimlrepo sources
        # and add CDN ZIP fallbacks where possible.
    ],                   'n': 398, 'p': 9},
    'automobile':       {'sources': [from_ucimlrepo(10)],
                         'n': 205, 'p': 26},
    'airfoil':          {'sources': [from_ucimlrepo(291)],
                         'n': 1503, 'p': 6},
    'boston':           {'sources': [from_url('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')],
                         'n': 506, 'p': 14},
    'crime':            {'sources': [from_ucimlrepo(183)],
                         'n': 1994, 'p': 128},
    'concrete':         {'sources': [from_ucimlrepo(165)],
                         'n': 1030, 'p': 9},
    'naval_propulsion': {'sources': [
        from_ucimlrepo(316),
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/316/'
            'condition%2Bbased%2Bmaintenance%2Bof%2Bnaval%2Bpropulsion%2Bplants.zip',
            'UCI CBM Dataset/data.txt',
            sep=r'\s+', header=None,
            names=['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2',
                   'P48', 'P1', 'P2', 'Pexh', 'TIC', 'mf',
                   'GT_compressor_decay', 'GT_turbine_decay'],
        ),
    ],                   'n': 11934, 'p': 18},
    'diabetes':         {'sources': [from_sklearn(load_diabetes)],
                         'n': 442, 'p': 11},
    'eye':              {'sources': [fetch_eye],
                         'n': 120, 'p': 201},
    'facebook':         {'sources': [from_ucimlrepo(368)],
                         'n': 500, 'p': 19},
    'forest':           {'sources': [from_ucimlrepo(162)],
                         'n': 517, 'p': 13},
    'parkinsons':       {'sources': [from_ucimlrepo(189)],
                         'n': 5875, 'p': 21},
    'real_estate':      {'sources': [from_ucimlrepo(477)],
                         'n': 414, 'p': 7},
    'student':          {'sources': [from_ucimlrepo(320)],  # Portuguese only; see docs/superpowers/issues/2026-04-11-student-dataset-structure.md
                         'n': 649, 'p': 33},
    'yacht':            {'sources': [from_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
                                              sep=r'\s+', header=None,
                                              names=['Longitudinal_position', 'Prismatic_coefficient',
                                                     'Length_displacement_ratio', 'Beam_draught_ratio',
                                                     'Length_beam_ratio', 'Froude_number',
                                                     'Residuary_resistance'])],
                         'n': 308, 'p': 7},
    'ribo':             {'sources': [fetch_riboflavin],
                         'n': 71, 'p': 4089},
    'blog':             {'sources': [
        from_ucimlrepo(304),
        from_zip(                                                    # CDN fallback (train split only)
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/304/blogfeedback.zip',
            'blogData_train.csv',
            header=None, names=[f'V{i}' for i in range(1, 282)]
        ),
    ],                   'n': 52397, 'p': 281},
    'ct_slices': {'sources': [
        from_ucimlrepo(206),
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/206/'
            'relative%2Blocation%2Bof%2Bct%2Bslices%2Bon%2Baxial%2Baxis.zip',
            'slice_localization_data.csv'
        ),
    ],           'n': 53500, 'p': 386},
    'tomshw':    {'sources': [from_zip_tar(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
        'regression.tar.gz',
        './regression/TomsHardware/TomsHardware.data',
        header=None, names=[f'V{i}' for i in range(1, 98)]
    )],          'n': 28179, 'p': 97},
    'twitter':   {'sources': [from_zip_tar(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
        'regression.tar.gz',
        './regression/Twitter/Twitter.data',
        header=None, names=[f'V{i}' for i in range(1, 79)]
    )],          'n': 583250, 'p': 78},
    'crop':             {'sources': []},  # URL TBD
    'elec_devices':     {'sources': []},  # URL TBD
    'starlight':        {'sources': []},  # URL TBD
}
```

- [ ] **Step 2: Add shape validation warning in `get_dataset()`**

At the end of `get_dataset()`, just before the final `return`, add:

```python
    meta = DATASETS[name]
    if 'n' in meta and df.shape[0] != meta['n']:
        warnings.warn(
            f"Dataset '{name}': expected n={meta['n']}, got {df.shape[0]}."
        )
    if 'p' in meta and df.shape[1] != meta['p']:
        warnings.warn(
            f"Dataset '{name}': expected p={meta['p']}, got {df.shape[1]}."
        )
    return pd.read_csv(cache_path, na_values=_EXTRA_NA_VALUES)
```

The full updated function body:

```python
def get_dataset(name):
    """Retrieve a dataset by name, using the local cache if available.

    Checks datasets/<name>.csv first. If absent, tries each source callable
    in REGISTRY[name]['sources'] in order, persists the result to
    datasets/<name>.csv on success, and returns the DataFrame.

    Raises KeyError for unknown names.
    Raises RuntimeError if all sources fail and no cache file exists.

    >>> df = get_dataset('yacht')
    >>> df.shape
    (308, 7)
    >>> df = get_dataset('diabetes')
    >>> df.shape
    (442, 11)
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list(DATASETS)}")

    cache_path = CACHE_DIR / f'{name}.csv'
    if not cache_path.exists():
        sources = DATASETS[name]['sources']
        last_exc = None
        for source in sources:
            try:
                df = source()
                CACHE_DIR.mkdir(exist_ok=True)
                df.to_csv(cache_path, index=False)
                break
            except Exception as exc:
                last_exc = exc
        else:
            raise RuntimeError(
                f"All sources failed for dataset '{name}'."
                + (f" Last error: {last_exc}" if last_exc else " No sources configured.")
            )

    df = pd.read_csv(cache_path, na_values=_EXTRA_NA_VALUES)
    meta = DATASETS[name]
    if 'n' in meta and df.shape[0] != meta['n']:
        warnings.warn(
            f"Dataset '{name}': expected n={meta['n']}, got {df.shape[0]}."
        )
    if 'p' in meta and df.shape[1] != meta['p']:
        warnings.warn(
            f"Dataset '{name}': expected p={meta['p']}, got {df.shape[1]}."
        )
    return df
```

- [ ] **Step 3: Run doctests to verify**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -20
```

Expected: all existing doctests pass (yacht shape (308,7), diabetes shape (442,11)).

- [ ] **Step 4: Run the full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/data.py
git commit -m "feat: add n/p metadata and shape validation to dataset registry"
```

---

### Task 2: Make EmpiricalDataProblem hashable

**Files:** `experiments/problems.py`

- [ ] **Step 1: Add hash/equality doctests to the class docstring**

In `experiments/problems.py`, append the following section to the end of the `EmpiricalDataProblem` class docstring (after the existing "Column transforms" section, before the closing `"""`):

```
    Value-object identity (hash and equality based on full definition):

    >>> p1 = EmpiricalDataProblem('diabetes', 'target')
    >>> p2 = EmpiricalDataProblem('diabetes', 'target')
    >>> p1 == p2
    True
    >>> p1 is p2
    False
    >>> len(frozenset({p1, p2}))
    1
    >>> p3 = EmpiricalDataProblem('yacht', 'Residuary_resistance')
    >>> len(frozenset({p1, p2, p3}))
    2
    >>> EmpiricalDataProblem('diabetes', 'target').drop
    ()
    >>> EmpiricalDataProblem('diabetes', 'target', drop=['bmi']).drop
    ('bmi',)
```

- [ ] **Step 2: Run doctests to verify they fail**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -20
```

Expected: failures on `p1 == p2` (no `__eq__`), `frozenset` deduplication, and `.drop` returning `()`.

- [ ] **Step 3: Update `__init__` to convert to tuples and add `__hash__` and `__eq__`**

Replace the `__init__` method and add `__hash__` and `__eq__` immediately after it:

```python
    def __init__(self, dataset, target, drop=None, nan_policy=None, transforms=None):
        self.dataset = dataset
        self.target = target
        self.drop = tuple(drop or [])
        self.nan_policy = nan_policy
        self.transforms = tuple(transforms or [])

    def __eq__(self, other):
        if not isinstance(other, EmpiricalDataProblem):
            return NotImplemented
        return (self.dataset, self.target, self.drop, self.nan_policy, self.transforms) == \
               (other.dataset, other.target, other.drop, other.nan_policy, other.transforms)

    def __hash__(self):
        return hash((self.dataset, self.target, self.drop, self.nan_policy, self.transforms))
```

- [ ] **Step 4: Run doctests to verify they pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -20
```

Expected: all doctests pass.

- [ ] **Step 5: Run the full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: make EmpiricalDataProblem a hashable value object"
```

---

### Task 3: Add NEURIPS2023 named collections to problems.py

**Files:** `experiments/problems.py`

- [ ] **Step 1: Add `DATASETS` import**

At the top of `experiments/problems.py`, update the import from `data`:

```python
from data import get_dataset, DATASETS
```

- [ ] **Step 2: Append named collections at the end of `problems.py`**

Add the following block at the very end of `experiments/problems.py`:

```python
import numpy as np  # noqa: F811 — already imported above; repeated for locality

NEURIPS2023 = frozenset({
    EmpiricalDataProblem('abalone',          'Rings'),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         transforms=[('price', np.log)]),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=['car_name'], nan_policy='drop_rows'),
    EmpiricalDataProblem('blog',             'V281'),
    EmpiricalDataProblem('boston',           'medv'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols'),
    EmpiricalDataProblem('ct_slices',        'reference'),
    EmpiricalDataProblem('diabetes',         'target'),
    EmpiricalDataProblem('eye',              'y'),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=['comment', 'like', 'share'],
                         nan_policy='drop_rows'),
    EmpiricalDataProblem('forest',           'area',
                         transforms=[('area', np.log1p)]),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=['GT_turbine_decay']),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=['GT_compressor_decay']),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=['total_UPDRS']),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=['motor_UPDRS']),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area'),
    EmpiricalDataProblem('ribo',             'y'),
    EmpiricalDataProblem('student',          'G3',
                         drop=['G1', 'G2']),
    EmpiricalDataProblem('tomshw',           'V97'),
    EmpiricalDataProblem('twitter',          'V78'),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         transforms=[('Residuary_resistance', np.log)]),
})

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

Note: `numpy` is already imported at the top of `problems.py` as `np` — the `# noqa: F811` comment is not needed; simply use the existing `np` without re-importing. Remove the re-import line and reference the `np` already in scope.

- [ ] **Step 3: Verify collection sizes are correct**

Run this quick check to confirm the expected cardinalities:

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "
from problems import NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3
print('NEURIPS2023:   ', len(NEURIPS2023))
print('NEURIPS2023_D2:', len(NEURIPS2023_D2))
print('NEURIPS2023_D3:', len(NEURIPS2023_D3))
print('D2 excluded:   ', {p.dataset for p in NEURIPS2023 - NEURIPS2023_D2})
print('D3 excluded:   ', {p.dataset for p in NEURIPS2023_D2 - NEURIPS2023_D3})
"
```

Expected:
- `NEURIPS2023`: 24 problems (23 with known sources + ribo)
- `NEURIPS2023_D2`: 23 problems (ribo excluded by p=4089)
- `NEURIPS2023_D3`: 18 problems (eye excluded by p=201; blog, ct_slices excluded by n; tomshw excluded by n; twitter excluded by n)
- D2 excluded: `{'ribo'}`
- D3 excluded: `{'eye', 'blog', 'ct_slices', 'tomshw', 'twitter'}`

Adjust threshold values (1000, 150, 20000) if the cardinalities differ from expected, then re-run.

- [ ] **Step 4: Run the full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add NEURIPS2023 named problem collections to problems.py"
```
