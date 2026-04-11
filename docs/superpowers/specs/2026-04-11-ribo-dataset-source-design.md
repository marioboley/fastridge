---
name: Ribo Dataset Source
description: Design for retrieving the riboflavin dataset from the CRAN hdi package and adding it to the real-data experiments
type: project
---

# Ribo Dataset Source Design

## Background

The `ribo` entry in `DATASETS` currently has an empty sources list. The riboflavin (vitamin B2) production dataset (n=71, p=4088) from Bühlmann, Kalisch, and Meier (2014) is the canonical high-dimensional ridge regression benchmark. It is distributed as `riboflavin.RData` inside the `hdi` R package on CRAN.

The dataset's internal R structure is atypical: it is an R `data.frame` with a matrix-valued column `x` (71×4088 gene expression values) and a scalar column `y` (log riboflavin production rate). This structure cannot be automatically converted to a pandas DataFrame by the `rdata` Python package's standard converter, requiring custom unpacking.

## Design

### New functions in `experiments/data.py`

**`_fetch_cran_rdata(package, version, rdata_path)`** — private shared helper.

Downloads `https://cran.r-project.org/src/contrib/{package}_{version}.tar.gz`, extracts `rdata_path` from the tarball into a temporary file, calls `rdata.parser.parse_file()`, and returns the parsed R object. Uses an inline `import rdata` following the `ucimlrepo` pattern. `urllib.request`, `tarfile`, `io`, and `tempfile` are stdlib.

**`from_cran(package, version, rdata_path)`** — public source factory.

Returns a source callable that calls `_fetch_cran_rdata`, then applies `rdata.conversion.convert()` and extracts the single top-level object as a DataFrame. Intended for future datasets stored as standard flat R `data.frame` objects, where the automatic conversion works without extra code.

**`fetch_riboflavin()`** — public source callable.

Calls `_fetch_cran_rdata('hdi', '0.1-10', 'hdi/data/riboflavin.RData')`, then manually unpacks the parsed R object:

- Navigates the pairlist attribute structure of the x matrix to extract `dim` ([71, 4088]) and `dimnames` (sample IDs as row index, gene names as column names)
- Reshapes the flat REAL value array column-major (`order='F'`) into the 71×4088 matrix
- Extracts the y vector
- Returns a DataFrame with `y` as the first column followed by the 4088 gene columns, row-indexed by sample IDs

The hdi version is pinned at `0.1-10` for reproducibility.

### `DATASETS` registry update

```python
'ribo': {'sources': [fetch_riboflavin]},
```

### `requirements.txt`

Add `rdata` as a project dependency (inline import in `data.py`, but must be present in the environment for the source to function).

### Module-level comment

Add `rdata` to the optional dependencies comment block at the top of `data.py` alongside `ucimlrepo`.

### Notebook update (`experiments/real_data.ipynb`)

Add to the full experiment cell (cell tagged `skip-execution`) only — not the preview cell, since `ribo.csv` is not git-tracked:

```python
EmpiricalDataProblem('ribo', 'y'),
```

n=71 (n_train≈49) and p=4088 make this severely underdetermined — the primary motivation for including it.

## Testing

`ribo.csv` is not git-tracked so cannot appear in CI doctests. The integration test is the full experiment cell running successfully with ribo included. The preview cell and CI are unaffected.

The existing `get_dataset` doctests (yacht, diabetes) continue to serve as regression coverage for the cache reader; they are unaffected by the new functions.

## Out of Scope

- Generalising `fetch_riboflavin` into a parameterised factory — deferred until a second R-package dataset with known structure arrives
- Git-tracking `ribo.csv` — dataset is large and not needed for CI
