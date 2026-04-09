# Real Data Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `experiments/data.py` with a dataset registry and `get_dataset()` function, move `RealDataExperiments` into `experiments/experiments.py`, commit a yacht CSV fixture, and create a reduced `experiments/real_data.ipynb`.

**Architecture:** `get_dataset(name)` checks `datasets/<name>.csv` first (cache), then tries source callables in order, persists on success. Source factories (`from_sklearn`, `from_ucimlrepo`, `from_url`) return zero-argument callables yielding a full `pd.DataFrame`. `RealDataExperiments` moves from `Analysis/Real_Data/RealDataFunction.py` into `experiments/experiments.py` with filenames replaced by DataFrames. The notebook's full experiment cells are tagged `skip-execution`; a two-dataset preview cell runs in pytest via nbmake.

**Tech Stack:** Python, pandas, scikit-learn, ucimlrepo, pytest (doctest + nbmake)

---

### Task 1: Add ucimlrepo dependency and commit yacht fixture

**Files:**
- Modify: `requirements.txt`
- Create: `datasets/.gitignore`
- Create: `datasets/yacht.csv` (downloaded and committed)

- [ ] **Step 1: Add ucimlrepo to requirements and install**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
echo "ucimlrepo>=0.0.3" >> requirements.txt
source .venv/bin/activate && pip install ucimlrepo
```

- [ ] **Step 2: Download yacht dataset and save as CSV fixture**

```bash
source .venv/bin/activate && python3 - <<'EOF'
import pandas as pd
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=243)
df = pd.concat([ds.data.features, ds.data.targets], axis=1)
df.columns = [c.strip() for c in df.columns]
print(df.shape)   # expect (308, 7)
df.to_csv('datasets/yacht.csv', index=False)
EOF
```

Expected output: `(308, 7)`

- [ ] **Step 3: Create datasets/.gitignore to track only the fixture**

Create `datasets/.gitignore` with content:
```
*
!.gitignore
!yacht.csv
```

- [ ] **Step 4: Verify fixture is tracked**

```bash
git status datasets/
```

Expected: `datasets/.gitignore` and `datasets/yacht.csv` shown as untracked (to be added).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt datasets/.gitignore datasets/yacht.csv
git commit -m "feat: add ucimlrepo dependency and yacht CSV fixture"
```

---

### Task 2: Implement source factories in experiments/data.py

**Files:**
- Create: `experiments/data.py`

- [ ] **Step 1: Create experiments/data.py with from_sklearn**

Create `experiments/data.py`:

```python
"""
Dataset registry and retrieval for real data experiments.

>>> df = get_dataset('yacht')
>>> df.shape
(308, 7)
>>> df = get_dataset('diabetes')
>>> df.shape
(442, 11)
"""
import warnings
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / 'datasets'


def from_sklearn(loader_fn):
    """Return a source callable that loads a sklearn dataset as a full DataFrame.

    The returned callable takes no arguments and returns a pd.DataFrame
    containing all feature columns and the target column.
    """
    def source():
        bunch = loader_fn(as_frame=True)
        return bunch.frame
    return source


def from_ucimlrepo(id):
    """Return a source callable that fetches a dataset from the UCI ML Repository.

    The returned callable takes no arguments and returns a pd.DataFrame
    with features and targets concatenated column-wise.
    """
    def source():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=id)
        return pd.concat([ds.data.features, ds.data.targets], axis=1)
    return source


def from_url(url, **read_csv_kwargs):
    """Return a source callable that downloads a CSV from the given URL.

    Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        return pd.read_csv(url, **read_csv_kwargs)
    return source
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, 'experiments')
from data import from_sklearn, from_ucimlrepo, from_url
print('ok')
"
```

Expected: `ok`

---

### Task 3: Implement REGISTRY and get_dataset

**Files:**
- Modify: `experiments/data.py`

- [ ] **Step 1: Add REGISTRY and get_dataset to experiments/data.py**

Append to `experiments/data.py` after the source factories:

```python
from sklearn.datasets import load_diabetes

REGISTRY = {
    'abalone':          {'sources': [from_ucimlrepo(1)]},
    'autompg':          {'sources': [from_ucimlrepo(9)]},
    'automobile':       {'sources': [from_ucimlrepo(10)]},
    'airfoil':          {'sources': [from_ucimlrepo(291)]},
    'bh':               {'sources': [from_ucimlrepo(162)]},  # verify UCI ID for Boston Housing
    'crime':            {'sources': [from_ucimlrepo(183)]},
    'concrete':         {'sources': [from_ucimlrepo(165)]},
    'naval_propulsion': {'sources': [from_ucimlrepo(316)]},
    'diabetes':         {'sources': [from_sklearn(load_diabetes), from_ucimlrepo(529)]},
    'eye':              {'sources': []},  # URL TBD — no network source yet
    'facebook':         {'sources': [from_ucimlrepo(368)]},
    'forest':           {'sources': [from_ucimlrepo(162)]},
    'parkinsons':       {'sources': [from_ucimlrepo(189)]},
    'real_estate':      {'sources': [from_ucimlrepo(477)]},
    'student':          {'sources': [from_ucimlrepo(320)]},
    'yacht':            {'sources': [from_ucimlrepo(243)]},
    'ribo':             {'sources': []},  # URL TBD — no network source yet
    'crop':             {'sources': []},  # URL TBD — no network source yet
    'elec_devices':     {'sources': []},  # URL TBD — no network source yet
    'starlight':        {'sources': []},  # URL TBD — no network source yet
}


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
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")

    cache_path = CACHE_DIR / f'{name}.csv'
    if cache_path.exists():
        return pd.read_csv(cache_path)

    sources = REGISTRY[name]['sources']
    last_exc = None
    for source in sources:
        try:
            df = source()
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"All sources failed for dataset '{name}'."
        + (f" Last error: {last_exc}" if last_exc else " No sources configured.")
    )
```

**Note on Boston Housing (`bh`):** UCI ML Repository removed this dataset. During implementation, verify whether `from_ucimlrepo(162)` is the correct entry or if a direct URL is needed. Update the registry accordingly. The `forest` dataset also uses UCI ID 162 — these are currently duplicated in error; `bh` needs a different source.

- [ ] **Step 2: Verify get_dataset with yacht (cache hit)**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, 'experiments')
from data import get_dataset
df = get_dataset('yacht')
print(df.shape)  # expect (308, 7)
"
```

Expected: `(308, 7)`

- [ ] **Step 3: Verify get_dataset with diabetes (from_sklearn)**

```bash
source .venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, 'experiments')
from data import get_dataset
df = get_dataset('diabetes')
print(df.shape)  # expect (442, 11)
print(list(df.columns))
"
```

Expected: `(442, 11)` with columns including `target`.

- [ ] **Step 4: Run doctests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v
```

Expected: doctests in module docstring pass (yacht cache hit, diabetes via sklearn).

- [ ] **Step 5: Commit**

```bash
git add experiments/data.py
git commit -m "feat: add data.py with source factories, REGISTRY, and get_dataset"
```

---

### Task 4: Add pytest tests for get_dataset caching behaviour

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create tests directory and test file**

Create `tests/__init__.py` (empty).

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest


def test_cache_miss_calls_source_and_writes_file(tmp_path, monkeypatch):
    import experiments.data as data_module
    monkeypatch.setattr(data_module, 'CACHE_DIR', tmp_path)
    monkeypatch.setitem(
        data_module.REGISTRY, '_test',
        {'sources': [lambda: pd.DataFrame({'a': [1, 2], 'b': [3, 4]})]}
    )
    df = data_module.get_dataset('_test')
    assert df.shape == (2, 2)
    assert (tmp_path / '_test.csv').exists()


def test_cache_hit_returns_file_without_calling_source(tmp_path, monkeypatch):
    import experiments.data as data_module
    monkeypatch.setattr(data_module, 'CACHE_DIR', tmp_path)
    # Write a CSV directly — source should never be called
    cached = pd.DataFrame({'x': [9, 8, 7]})
    cached.to_csv(tmp_path / '_cached.csv', index=False)
    called = []
    def failing_source():
        called.append(True)
        raise RuntimeError("source should not be called")
    monkeypatch.setitem(
        data_module.REGISTRY, '_cached',
        {'sources': [failing_source]}
    )
    df = data_module.get_dataset('_cached')
    assert list(df['x']) == [9, 8, 7]
    assert called == []


def test_all_sources_fail_raises_runtime_error(tmp_path, monkeypatch):
    import experiments.data as data_module
    monkeypatch.setattr(data_module, 'CACHE_DIR', tmp_path)
    monkeypatch.setitem(
        data_module.REGISTRY, '_failing',
        {'sources': [lambda: (_ for _ in ()).throw(ValueError("fail"))]}
    )
    with pytest.raises(RuntimeError, match="All sources failed"):
        data_module.get_dataset('_failing')
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && pytest tests/test_data.py -v
```

Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/test_data.py
git commit -m "test: add caching behaviour tests for get_dataset"
```

---

### Task 5: Add data.py doctests to pytest.ini

**Files:**
- Modify: `pytest.ini`

- [ ] **Step 1: Add doctest-modules for experiments/data.py**

Edit `pytest.ini` to add `--doctest-modules experiments/data.py`:

```ini
[pytest]
addopts =
    --doctest-modules fastridge.py
    --doctest-modules experiments/data.py
    --codeblocks README.md
    --nbmake experiments/double_asymptotic_trends.ipynb
    --nbmake experiments/sparse_designs.ipynb
    --nbmake experiments/tutorial.ipynb
```

- [ ] **Step 2: Run full pytest to verify nothing breaks**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && pytest
```

Expected: all existing tests plus new doctests pass.

- [ ] **Step 3: Commit**

```bash
git add pytest.ini
git commit -m "test: add experiments/data.py to doctest-modules in pytest.ini"
```

---

### Task 6: Move RealDataExperiments into experiments/experiments.py

**Files:**
- Modify: `experiments/experiments.py`

- [ ] **Step 1: Read RealDataFunction.py to understand its full interface**

Read `Analysis/Real_Data/RealDataFunction.py` — already done during brainstorming; the function signature is:

```python
def RealDataExperiments(data_files, targets, estimators={}, n_iterations=100,
                        test_prop=0.3, seed=None, polynomial=None, classification=False)
```

Where `data_files` is a list of CSV filename strings and `targets` is a parallel list of target column names. The function reads each file with `pd.read_csv`.

- [ ] **Step 2: Add RealDataExperiment to experiments/experiments.py**

Append to `experiments/experiments.py` after the existing classes. The only change from the original is:
- Accept `dataframes` (list of `pd.DataFrame`) instead of `data_files` (list of filename strings)
- Accept `targets` (list of target column names) as before
- Remove the `pd.read_csv` call and `data_name` derivation from filename
- Add `import pandas as pd` at the top if not already present

Add these imports at the top of `experiments/experiments.py` (after existing imports):

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import time
import pandas as pd
```

Then append the class at the bottom of `experiments/experiments.py`:

```python
def RealDataExperiment(dataframes, targets, names, estimators={}, n_iterations=100,
                       test_prop=0.3, seed=None, polynomial=None, classification=False):
    """Run repeated train/test experiments on a list of DataFrames.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Raw datasets (full DataFrames from get_dataset).
    targets : list of str
        Target column name for each dataset.
    names : list of str
        Display name for each dataset (used as result dict key).
    estimators : dict of str -> fitted estimator
    n_iterations : int
    test_prop : float
    seed : int or None
    polynomial : int or None
    classification : bool
    """
    results = {}
    for j, df in enumerate(dataframes):
        name = names[j]
        target = targets[j]
        X = df.drop([target], axis=1)
        y = df[target]

        print(name)

        categorical_cols = [col for col in X.columns if X[col].dtype in ['object', 'category']]
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        if categorical_cols:
            encoded_X = encoder.fit_transform(X[categorical_cols])
            X = pd.concat([
                X.drop(categorical_cols, axis=1),
                pd.DataFrame(encoded_X, columns=encoder.get_feature_names_out(categorical_cols))
            ], axis=1)

        if polynomial is not None:
            poly = PolynomialFeatures(degree=polynomial, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))
            ppoly = X_poly.shape[1]
            npoly = X_poly.shape[0]
            if npoly * ppoly > 35000000:
                X_poly = X_poly.drop(X.columns, axis=1)
                pnew = int(np.ceil(35000000 / npoly))
                X_poly = X_poly.iloc[:, np.random.choice(X_poly.shape[1], size=pnew, replace=False)]
                X = pd.concat([X, X_poly], axis=1)
            else:
                X = X_poly
            print(X.shape)

        estimator_results = {est_name: {'mse': [], 'r2': [], 'time': [], 'p': [], 'lambda': [], 'iter': [], 'CA': [], 'q': []}
                             for est_name in estimators}

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_iterations):
            print(i, end='')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
            std = X_train.std()
            non_zero_std_cols = std[std != 0].index
            X_train = X_train[non_zero_std_cols]
            X_test = X_test[non_zero_std_cols]

            for est_name, estimator in estimators.items():
                t0 = time.time()
                estimator.fit(X_train, y_train)
                elapsed = time.time() - t0

                if classification:
                    score = estimator.score(X_test, y_test)
                    estimator_results[est_name]['CA'].append(score)
                    estimator_results[est_name]['p'].append(X_train.shape[1])
                    estimator_results[est_name]['q'].append(len(estimator.classes_))
                else:
                    y_pred = estimator.predict(X_test)
                    estimator_results[est_name]['mse'].append(mean_squared_error(y_test, y_pred))
                    estimator_results[est_name]['r2'].append(r2_score(y_test, y_pred))
                    estimator_results[est_name]['p'].append(len(estimator.coef_))
                    estimator_results[est_name]['lambda'].append(estimator.alpha_)

                estimator_results[est_name]['time'].append(elapsed)
                if est_name == 'EM':
                    estimator_results[est_name]['iter'].append(estimator.iterations_)

        data_results = {}
        for est_name, er in estimator_results.items():
            data_results[est_name] = {
                'mse':    np.mean(er['mse']) if er['mse'] else float('nan'),
                'r2':     np.mean(er['r2']) if er['r2'] else float('nan'),
                'time':   np.mean(er['time']),
                'p':      np.mean(er['p']),
                'n_train': int(X_train.shape[0]),
                'lambda': np.mean(er['lambda']) if er['lambda'] else float('nan'),
                'iter':   np.mean(er['iter']) if er['iter'] else 100,
                'CA':     np.mean(er['CA']) if er['CA'] else float('nan'),
                'q':      np.mean(er['q']) if er['q'] else float('nan'),
            }
        results[name] = data_results

    return results
```

- [ ] **Step 3: Verify experiments.py imports cleanly**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && python3 -c "
import sys; sys.path.insert(0, 'experiments')
from experiments import RealDataExperiment
print('ok')
"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: move RealDataExperiments into experiments.py as RealDataExperiment"
```

---

### Task 7: Create experiments/real_data.ipynb

**Files:**
- Create: `experiments/real_data.ipynb`
- Modify: `pytest.ini`

- [ ] **Step 1: Check RidgeEM signature for squareU parameter**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && python3 -c "
import inspect
from fastridge import RidgeEM
print(inspect.signature(RidgeEM.__init__))
"
```

If `squareU` is not in the signature, use `t2=True` (the default) instead. Note the result — the notebook cell must use the correct current parameter name.

- [ ] **Step 2: Create real_data.ipynb**

Create `experiments/real_data.ipynb` as a new notebook. Use the Read tool on an existing notebook (e.g. `experiments/tutorial.ipynb`) to obtain valid metadata format, then create with the following cells:

**Cell 1 — imports (cell_type: code):**
```python
import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from tabulate import tabulate
from fastridge import RidgeEM, RidgeLOOCV
from experiments import RealDataExperiment
from data import get_dataset
```

**Cell 2 — preview experiment, 2 datasets (cell_type: code):**
```python
datasets = ['yacht', 'diabetes']
targets  = ['Residuary resistance per unit weight of displacement', 'target']
names    = ['yacht', 'diabetes']

dataframes = [get_dataset(name) for name in datasets]

estimators = {
    'EM':     RidgeEM(),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
}

results = RealDataExperiment(dataframes, targets, names, estimators, n_iterations=10, seed=1)

table = {}
for data_name, data_result in results.items():
    table[data_name] = {est: data_result[est]['r2'] for est in data_result}
tdf = pd.DataFrame(table)
print(tabulate(tdf.transpose(), headers='keys', tablefmt='psql', floatfmt='.2f'))
```

**Cell 3 — full experiment (cell_type: code, metadata tags: `['skip-execution']`):**
```python
# Full experiment matching legacy notebook — skip-execution in CI
datasets_full = ['abalone', 'automobile', 'autompg', 'airfoil', 'bh',
                 'crime', 'concrete', 'naval_propulsion', 'diabetes',
                 'facebook', 'forest', 'parkinsons', 'real_estate',
                 'student', 'yacht', 'ribo']
targets_full = [
    'Rings', 'price', 'mpg', 'Scaled sound pressure level', 'MEDV',
    'ViolentCrimesPerPop', 'Concrete compressive strength(MPa, megapascals)',
    'gt_c_decay', 'target', 'Total Interactions', 'area',
    'motor_UPDRS', 'Y house price of unit area', 'G3',
    'Residuary resistance per unit weight of displacement', 'y'
]
names_full = datasets_full

dataframes_full = [get_dataset(name) for name in datasets_full]

estimators_full = {
    'EM':     RidgeEM(),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
}

results_full = RealDataExperiment(dataframes_full, targets_full, names_full,
                                  estimators_full, n_iterations=100, seed=1)

table_full = {}
for data_name, data_result in results_full.items():
    row = {est: data_result[est]['r2'] for est in data_result}
    row['p'] = data_result[list(data_result)[0]]['p']
    row['n_train'] = data_result[list(data_result)[0]]['n_train']
    table_full[data_name] = row
tdf_full = pd.DataFrame(table_full)
print(tabulate(tdf_full.transpose().sort_values('n_train', ascending=False),
               headers='keys', tablefmt='psql', floatfmt='.2f'))
```

**Note:** Target column names for the full experiment (cell 3) must be verified against actual DataFrame columns returned by `get_dataset` — they depend on the column names each UCI source provides. Update as needed during implementation.

- [ ] **Step 3: Add nbmake entry to pytest.ini**

Edit `pytest.ini`:

```ini
[pytest]
addopts =
    --doctest-modules fastridge.py
    --doctest-modules experiments/data.py
    --codeblocks README.md
    --nbmake experiments/double_asymptotic_trends.ipynb
    --nbmake experiments/sparse_designs.ipynb
    --nbmake experiments/tutorial.ipynb
    --nbmake experiments/real_data.ipynb
```

- [ ] **Step 4: Run pytest to verify notebook preview cell passes**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate && pytest experiments/real_data.ipynb -v
```

Expected: notebook runs without error. If `squareU` error appears, fix `RidgeEM()` call per Step 1 finding.

- [ ] **Step 5: Run full pytest**

```bash
source .venv/bin/activate && pytest
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/real_data.ipynb pytest.ini
git commit -m "feat: add real_data.ipynb with preview experiment and skip-execution full experiment"
```

---

### Task 8: Push dev and merge to main

- [ ] **Step 1: Push dev**

```bash
git push origin dev
```

- [ ] **Step 2: Merge to main**

```bash
git checkout main
git merge dev
git push origin main
git checkout dev
```

- [ ] **Step 3: Commit and push spec and plan docs**

```bash
git add docs/superpowers/specs/2026-04-09-real-data-module-design.md \
        docs/superpowers/plans/2026-04-09-real-data-module.md
git commit -m "docs: add real data module spec and implementation plan"
git push origin dev
git checkout main
git merge dev
git push origin main
git checkout dev
```
