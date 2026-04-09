---
name: Real Data Module
description: Design for experiments/data.py — two-tier dataset registry with local caching, programmatic retrieval, and a reduced real_data.ipynb
type: project
---

# Real Data Module Design

## Background

The legacy notebook `Analysis/Real_Data/Real data.ipynb` runs a comparative experiment across 19 regression datasets loaded from local CSV files not in the repository. This project establishes `experiments/data.py` to abstract dataset access, moves `RealDataExperiments` into `experiments/experiments.py`, and creates a reproducible `experiments/real_data.ipynb` with a small runnable preview experiment.

## Goal

Implement `experiments/data.py` with a `DATASETS` registry and a `get_dataset(name)` function. Provide `experiments/real_data.ipynb` with a reduced preview experiment (tested via nbmake) and full-experiment cells tagged `skip-execution`. Move `RealDataFunction.py` logic into `experiments.py`.

## Known Open Issue

`RidgeEM(squareU=False)` appears in the legacy notebook. This parameter is not present in the current `RidgeEM` API — it may be a renamed or removed parameter. Resolution is deferred until the notebook is assembled and run.

---

## Architecture

### Tier 1 — Data retrieval (`get_dataset`)

`get_dataset(name) -> pd.DataFrame`

Returns the full raw DataFrame for the named dataset. Pipeline:
1. Check `datasets/<name>.csv` — if present, load via `pd.read_csv` and return
2. Try each source callable in `DATASETS[name]['sources']` in order; stop on first success
3. Persist result to `datasets/<name>.csv` via `pd.DataFrame.to_csv(index=False)`
4. Return DataFrame

Raises `KeyError` for unknown names. Raises `RuntimeError` if all sources fail.

### Tier 2 — Task assembly (caller's responsibility)

Target extraction, column dropping, one-hot encoding, and train/test splitting remain the caller's responsibility — currently handled by `RealDataExperiment` (moved from `RealDataFunction.py`). This project does not refactor tier 2.

---

## Source Factories

Each factory returns a zero-argument callable that returns a `pd.DataFrame` or raises on failure.

```python
def from_sklearn(loader_fn) -> callable
```
Calls `loader_fn(as_frame=True)` and returns `bunch.frame` — the full DataFrame including the target column.

Doctest:
```python
>>> from sklearn.datasets import load_diabetes
>>> src = from_sklearn(load_diabetes)
>>> df = src()
>>> df.shape
(442, 11)
```

```python
def from_ucimlrepo(id: int) -> callable
```
Calls `ucimlrepo.fetch_ucirepo(id=id)` and returns the combined features+targets DataFrame.

```python
def from_url(url: str, **read_csv_kwargs) -> callable
```
Returns `pd.read_csv(url, **read_csv_kwargs)`.

---

## DATASETS Registry

Two-level structure:

```python
DATASETS = {
    'abalone':    {'sources': [from_ucimlrepo(1)]},
    'diabetes':   {'sources': [from_sklearn(load_diabetes)]},
    ...
}
```

Full registry entries (one per raw dataset — shared-source tasks reference the same entry):

| Key | Full name | Sources |
|---|---|---|
| `abalone` | Abalone | `from_ucimlrepo(1)` |
| `autompg` | Auto MPG | `from_ucimlrepo(9)` |
| `automobile` | Automobile | `from_ucimlrepo(10)` |
| `airfoil` | Airfoil Self-Noise | `from_ucimlrepo(291)` |
| `bh` | Boston Housing | `from_url(BH_URL, ...)` |
| `crime` | Communities and Crime | `from_ucimlrepo(183)` |
| `concrete` | Concrete Compressive Strength | `from_ucimlrepo(165)` |
| `naval_propulsion` | Naval Propulsion Plants | `from_ucimlrepo(316)` |
| `diabetes` | Diabetes | `from_sklearn(load_diabetes)`, `from_ucimlrepo(529)` |
| `eye` | Scheetz gene expression | `from_url(EYE_URL)` |
| `facebook` | Facebook Metrics | `from_ucimlrepo(368)` |
| `forest` | Forest Fires | `from_ucimlrepo(162)` |
| `parkinsons` | Parkinson's Telemonitoring | `from_ucimlrepo(189)` |
| `real_estate` | Real Estate Valuation | `from_ucimlrepo(477)` |
| `student` | Student Performance | `from_ucimlrepo(320)` |
| `yacht` | Yacht Hydrodynamics | `from_ucimlrepo(243)` |
| `ribo` | Riboflavin | `from_url(RIBO_URL)` |
| `crop` | CROP (UCR) | `from_url(CROP_URL)` |
| `elec_devices` | Electric Devices (UCR) | `from_url(ELEC_URL)` |
| `starlight` | Star Light Curves (UCR) | `from_url(STAR_URL)` |

Notes: `bh`, `eye`, `ribo` and the three UCR URLs need to be identified during implementation. `naval_propulsion` backs both `conditionCom` and `conditionTur` tasks in the notebook; `parkinsons` backs both motor and total UPDRS tasks. The exact column names, target names, and drop lists for tier-2 assembly are determined during notebook construction.

---

## `datasets/` Folder

Located at project root. Tracked in git. Contents gitignored by default via `datasets/.gitignore`:
```
*
!.gitignore
!yacht.csv
```
`yacht.csv` (308 rows, 7 columns) is committed as the example fixture — small, clean, no missing values, easily retrievable from `from_ucimlrepo(243)` for comparison.

---

## Doctests

Placed in the module docstring of `experiments/data.py`. Added to pytest via existing `--doctest-modules` coverage (no pytest.ini change needed).

The primary module doctest exercises `get_dataset` for both supported local cases — committed CSV and sklearn:

```python
>>> df = get_dataset('yacht')   # cache hit — committed datasets/yacht.csv
>>> df.shape
(308, 7)
>>> df = get_dataset('diabetes')  # from_sklearn, no network
>>> df.shape
(442, 11)
```

No doctests at the individual source factory level.

---

## `experiments/experiments.py` Changes

`RealDataFunction.py` logic moved in as `RealDataExperiment`. The interface changes minimally: instead of accepting filenames, it accepts a list of `(name, df)` pairs where `df` is the raw DataFrame from `get_dataset`. Target column, drop list, and encoding remain internal. Exact refactoring is deferred to implementation — the key constraint is that the notebook's existing call pattern is preserved as closely as possible.

---

## `experiments/real_data.ipynb`

Structure:
- **Cell 1 (imports):** `get_dataset`, `RealDataExperiment`, estimators
- **Cell 2 (preview experiment):** runs on `['yacht', 'diabetes']` only — two datasets with tested loaders, no network required after cache populated. Produces summary table.
- **Cell 3+ (full experiment, `skip-execution`):** full dataset list matching legacy notebook, all polynomial degrees

Added to `pytest.ini` nbmake list.

---

## Out of Scope

- Refactoring tier-2 task assembly (target selection, encoding, train/test split) — stays inside `RealDataExperiment`
- Figure 3 scatter plot — deferred to a future project
- Resolving `RidgeEM(squareU=False)` — deferred until notebook run reveals the issue
