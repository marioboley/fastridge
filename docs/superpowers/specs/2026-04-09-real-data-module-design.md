---
name: Real Data Module
description: Design for experiments/data.py — registry-based dataset access with local caching and programmatic retrieval
type: project
---

# Real Data Module Design

## Background

The legacy notebook `Analysis/Real_Data/Real data.ipynb` runs a comparative experiment across 19 regression datasets loaded from local CSV files not in the repository. This module abstracts dataset access so that experiments can be reproduced without manual data preparation.

## Goal

Implement `experiments/data.py` with a registry of all 19 single-target regression datasets from the legacy notebook. Each entry records expected dimensions and an ordered list of programmatic sources. A `get_dataset(name)` function checks a local cache first, then tries sources in sequence, persists the result, and warns if retrieved dimensions don't match expectations.

## Architecture

Single module `experiments/data.py`. Local cache at `datasets/` (project root, gitignored). No notebook in this project — the module is exercised only via doctests.

---

## API

```python
get_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]
```

Returns `(X, y)` where:
- `X` is a `pd.DataFrame` of feature columns (categorical columns retained as-is, downstream encoding is the caller's responsibility)
- `y` is a `pd.Series` of the target column
- Rows with any missing values in X or y are dropped

**Algorithm:**
1. Check `datasets/<name>.csv` — if present, load and return (X, y)
2. Try each source callable in `REGISTRY[name]['sources']` in order; stop on first success
3. Persist the result to `datasets/<name>.csv`
4. Check `X.shape[0] == n` and `X.shape[1] == p`; emit `warnings.warn` for each mismatch
5. Return (X, y)

Raises `KeyError` for unknown names. Raises `RuntimeError` if all sources fail.

---

## Source Factories

Each factory returns a zero-argument callable that returns `(X: pd.DataFrame, y: pd.Series)` or raises on failure.

```python
def from_sklearn(loader_fn, target: str) -> callable
```
Calls `loader_fn(as_frame=True)`, extracts target column `target` from `frame.frame`.

```python
def from_ucimlrepo(id: int, target: str, drop: list[str] = None) -> callable
```
Calls `ucimlrepo.fetch_ucirepo(id=id)`, combines `features` and `targets` DataFrames, extracts `target` column. Drops `drop` columns from X if provided.

```python
def from_url(url: str, target: str, sep: str = ',', header: int = 0,
             names: list[str] = None, drop: list[str] = None) -> callable
```
Downloads via `pd.read_csv(url, sep=sep, header=header, names=names)`, extracts `target`, drops `drop` columns from X.

---

## Dataset Registry

`REGISTRY` is a module-level dict. Each entry:
```python
{
    'n': int,        # expected rows after NaN drop
    'p': int,        # expected feature columns (raw, before one-hot encoding)
    'sources': list  # ordered list of source callables
}
```

The `n` and `p` values reflect what `get_dataset` returns (post-NaN-drop, pre-encoding). Values marked **[verify]** are estimates from the paper appendix and notebook output that need confirmation during implementation.

| Key | Full name | n | p | Source |
|---|---|---|---|---|
| `abalone` | Abalone | 4177 | 8 | `from_ucimlrepo(1, target='Rings')` |
| `autompg` | Auto MPG | 392 | 7 | `from_ucimlrepo(9, target='mpg')` |
| `automobile` | Automobile | 159 [verify] | 25 [verify] | `from_ucimlrepo(10, target='price')` |
| `airfoil` | Airfoil Self-Noise | 1503 | 5 | `from_ucimlrepo(291, target='Scaled sound pressure level')` |
| `bh` | Boston Housing | 506 | 13 | `from_url(BH_URL, target='MEDV', sep=' ', names=BH_COLS)` — see note |
| `crime` | Communities and Crime | 1994 [verify] | 127 [verify] | `from_ucimlrepo(183, target='ViolentCrimesPerPop')` |
| `concrete` | Concrete Compressive Strength | 1030 | 8 | `from_ucimlrepo(165, target='Concrete compressive strength(MPa)')` |
| `conditionCom` | Naval Propulsion — compressor | 11934 | 14 [verify] | `from_ucimlrepo(316, target='gt_c_decay', drop=['gt_t_decay'])` [verify drop] |
| `conditionTur` | Naval Propulsion — turbine | 11934 | 14 [verify] | `from_ucimlrepo(316, target='gt_t_decay', drop=['gt_c_decay'])` [verify drop] |
| `diabetes` | Diabetes | 442 | 10 | `from_sklearn(load_diabetes, target='target')` |
| `eye` | Scheetz gene expression | 120 | 200 | `from_url(EYE_URL, target='y')` — see note |
| `facebook` | Facebook Metrics | 500 | 18 [verify] | `from_ucimlrepo(368, target='Total Interactions')` |
| `forest` | Forest Fires | 517 | 12 | `from_ucimlrepo(162, target='area')` |
| `parkinson_motor` | Parkinson's Telemonitoring | 5875 | 19 [verify] | `from_ucimlrepo(189, target='motor_UPDRS', drop=['subject#', 'total_UPDRS'])` |
| `parkinson_total` | Parkinson's Telemonitoring | 5875 | 19 [verify] | `from_ucimlrepo(189, target='total_UPDRS', drop=['subject#', 'motor_UPDRS'])` |
| `realEstate` | Real Estate Valuation | 414 | 6 | `from_ucimlrepo(477, target='Y house price of unit area')` |
| `student` | Student Performance | 395 [verify] | 30 [verify] | `from_ucimlrepo(320, target='G3', drop=['G1', 'G2'])` |
| `yacht` | Yacht Hydrodynamics | 308 | 6 | `from_ucimlrepo(243, target='Residuary resistance per unit weight of displacement')` |
| `ribo` | Riboflavin | 71 | 4088 | `from_url(RIBO_URL, target='y')` — see note |

### Notes on uncertain sources

**Boston Housing (`bh`):** No longer in sklearn. UCI archive URL (`https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data`) is whitespace-separated with no header; column names must be hardcoded as `BH_COLS`. Alternative: `ucimlrepo` may have it (verify during implementation). The `n=159` for automobile and NaN-drop counts for crime/bh need empirical confirmation.

**Eye (`eye`):** Scheetz et al. (2006) gene expression data, distributed with the R package `hdi`. A direct CSV URL needs to be identified during implementation (e.g. from a CRAN mirror or published supplementary). Dimensions n=120, p=200 are from the paper appendix.

**Riboflavin (`ribo`):** Bühlmann et al. riboflavin production dataset, also from R package `hdi`. Direct CSV URL needs identification during implementation. Dimensions n=71, p=4088 are from the paper appendix.

**Student Performance:** UCI dataset 320 contains two files (math and Portuguese). The original experiment used one; which needs verification. `G1` and `G2` (earlier grade reports) are dropped per paper appendix note.

**conditionCom / conditionTur:** Both come from UCI dataset 316. The `from_ucimlrepo` callable is identical except for the target column — the source loads the full dataset and extracts the respective target.

---

## Doctests

One doctest per source kind, included in module docstring or function docstrings. All doctests must pass in the standard pytest run — retrieval will use the local cache after first run.

```python
# from_sklearn (fast, no network on cache hit)
>>> X, y = get_dataset('diabetes')
>>> X.shape
(442, 10)
>>> y.name
'target'

# from_ucimlrepo (network on first run, cached thereafter)
>>> X, y = get_dataset('yacht')
>>> X.shape
(308, 6)

# from_url (network on first run, cached thereafter)
>>> X, y = get_dataset('ribo')  # or 'eye' if ribo URL unresolvable
>>> X.shape[1]
4088
```

Doctests that require network access will be slow on first run. The cache makes subsequent runs fast. No `# doctest: +SKIP` — these must actually execute.

---

## Cache Format

`datasets/<name>.csv` — comma-separated, with header row. Written by `pd.DataFrame.to_csv(index=False)` on all columns (X columns + target column combined). Read back by `pd.read_csv`, then target column extracted.

---

## pytest Integration

Add to `pytest.ini` addopts:
```
--doctest-modules experiments/data.py
```

`ucimlrepo` must be added to `requirements.txt` (project dependencies).

---

## Out of Scope

- Running any experiment or producing any figure
- Polynomial feature expansion
- One-hot encoding or other preprocessing beyond NaN-row dropping
- The three UCR time-series classification datasets (CROP, ELEC D, STAR L)
- The four large commented-out regression datasets (BLOG, TWITTER, TOM'S HW, CT SLICES)
