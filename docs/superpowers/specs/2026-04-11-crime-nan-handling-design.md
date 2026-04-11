---
name: Crime NaN Handling
description: Design for handling ? missing value markers in crime dataset and adding nan_policy='drop_cols' to EmpiricalDataProblem
type: project
---

# Crime NaN Handling

## Background

The crime dataset (UCI Communities and Crime, id=183) currently leads to SVD convergence failures on some splits. Two independent root causes: (1) the UCI raw data encodes missing values as `?`, which causes pandas to infer those columns as string dtype, and (2) some columns are metadata identifiers that should be excluded as features.

### Forensic findings

**Source:** crime is retrieved via `from_ucimlrepo(183)` and cached to `datasets/crime.csv`. The UCI raw data uses `?` as a missing value marker. Because the columns mix numeric values and `?` entries, pandas infers them as string dtype rather than numeric. The cache reader in `get_dataset()` uses plain `pd.read_csv` with no `na_values` argument, so `?` is not converted to NaN.

**Effect:** `run_real_data_experiments` detects string-dtype columns as categorical and one-hot encodes them. The crime dataset has 26 such columns. After one-hot encoding, p jumps from 99 to 4089 — making the problem severely underdetermined (n_train≈1395 < p=4089) and causing SVD failures on some splits.

**Legacy behaviour:** Legacy p=99 exactly matches the count of float64 columns. Legacy operated on local CSV files (no longer in the repo) that were pre-processed to contain only the 99 float64 columns. The 26 string columns were never present.

**`na_values='?'` effect (verified):**

After reading with `na_values=['?']`:
- 25 of the 26 str columns become float64 with NaN (numeric features with missing data; 84% NaN rate)
- 1 genuine string column remains: `communityname` (city names, no `?`)
- Total NaN: 0 (in the 99 float64 columns that survive)

**`drop=['state', 'fold', 'communityname']` + `nan_policy='drop_cols'` (verified):**

- `state` (US state code 1–56) and `fold` (CV fold assignment 1–10): metadata identifiers, not predictive features
- `communityname`: high-cardinality city name string; one-hot encoding would create thousands of columns
- After dropping these 3 and removing the 25 NaN columns: p=99, no NaN, no string cols
- n_train = 1994 − ceil(1994 × 0.3) = 1395 ✓ matches legacy exactly

**Side-effect check:** Only crime contains `?` among all currently cached datasets (confirmed by grep). No other dataset is affected by the cache reader change.

---

## Design

### Change 1: Single return point in `get_dataset()` with `na_values` (`experiments/data.py`)

Restructure `get_dataset()` to always read from the cache file as its single return point. If the cache does not exist, fetch from sources and save first:

```python
def get_dataset(name):
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

    return pd.read_csv(cache_path, na_values=_EXTRA_NA_VALUES)
```

With a module-level constant:

```python
# Additional missing value markers beyond pandas defaults:
# '', 'NA', 'N/A', '#N/A', '#N/A N/A', '#NA', 'NaN', '-NaN', '-nan', 'nan',
# 'None', '<NA>', 'NULL', 'null', 'n/a', '1.#IND', '-1.#IND', '1.#QNAN', '-1.#QNAN'
_EXTRA_NA_VALUES = ['?']
```

`na_values` extends pandas' existing default list — it does not replace it. Only crime is currently affected among all cached datasets (confirmed by grep). Adding further symbols is a one-list change.

### Change 2: `nan_policy='drop_cols'` in `EmpiricalDataProblem` (`experiments/problems.py`)

Extend `get_X_y()` to support a new policy value:

```python
if self.nan_policy == 'drop_rows':
    df = df.dropna()
elif self.nan_policy == 'drop_cols':
    df = df.dropna(axis=1)
```

Applied after target-NaN row dropping and before `reset_index`. Drops any column with any NaN.

### Crime task spec (`experiments/real_data.ipynb`)

Uncomment crime in full experiment cells:

```python
EmpiricalDataProblem('crime', 'ViolentCrimesPerPop',
                     drop=['state', 'fold', 'communityname'],
                     nan_policy='drop_cols'),
```

Note: the preview cell must use only git-tracked or sklearn datasets. Since crime.csv is not git-tracked, it should not appear in the preview cell, and enabled only in the full experiment cell (tagged skip-execution).

---

## Testing

### Doctest for `nan_policy='drop_cols'`

`crime.csv` is not git-tracked so cannot be used in CI doctests. Use `automobile` with a synthetic scenario instead — or add a dedicated doctest once crime.csv is committed.

For now, the integration test is the full experiment cell running successfully with crime included.

### Regression coverage for `na_values=['?']`

automobile.csv contains no `?` (confirmed by grep) and the existing automobile doctests verify row counts of 201 and 159 after NaN dropping. Adding `na_values=['?']` to the cache reader has zero effect on automobile (verified: NaN count unchanged at 59). The existing doctests therefore cover any unintended regression from the cache reader change.

---

## Out of Scope

- Handling sentinel numeric missing values (e.g. `999`, `9999`) — require domain knowledge
- Dropping columns above a NaN threshold (rather than any NaN) — not needed for crime
- Git-tracking crime.csv — dataset is large and not needed for CI (full experiment is skip-execution)
