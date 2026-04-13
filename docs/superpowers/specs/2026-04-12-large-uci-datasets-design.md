# Large UCI Datasets Design

## Goal

Add four large datasets — Twitter, TomsHW, Blog, and CT Slices — to `experiments/data.py`. These datasets appear in paper Table 2 but were absent from the legacy notebook (commented out as "data file size is too big"). All sources use the beta UCI archive CDN (`cdn.uci-ics-mlr-prod.aws.uci.edu`). For consistency with all existing UCI datasets, `from_ucimlrepo` is primary and the CDN source is the fallback where applicable. Dataset #248 (Buzz in Social Media) contains two distinct files (Twitter and TomsHW) that the API cannot distinguish (analogous to the Student dataset) and, given the API is currently down, it is unclear which of them it would produce; therefore Twitter and TomsHW use the CDN source only for the moment. A future task could add CDN fallbacks to all existing UCI datasets and API for one of Twitter and TomsHW.

## Datasets

### Twitter

Source: UCI dataset #248 "Buzz in Social Media" CDN ZIP (`buzz+in+social+media.zip`, ~33 MB). The ZIP contains a single `regression.tar.gz` holding `regression/Twitter/Twitter.data` — 583,250 rows × 78 cols, no header. Column names: `V1..V78`. Target: `V78` (last column).

**Paper reproduction:** Full file has 583,250 rows; 70% split = **408,275** — exactly matches paper Table 2 (n=408,275, p=77). Source will reproduce paper values.

### TomsHW

Source: same CDN ZIP as Twitter. The tar contains `regression/TomsHardware/TomsHardware.data` — 28,179 rows × 97 cols, no header. Column names: `V1..V97`. Target: `V97` (last column).

**Paper reproduction:** 28,179 × 0.7 = **19,725** — exactly matches paper Table 2 (n=19,725, p=96). Source will reproduce paper values.

### Blog

Source: UCI dataset #304 "BlogFeedback" CDN ZIP (`blogfeedback.zip`, ~2.5 MB). The ZIP contains `blogData_train.csv` (52,397 rows × 281 cols, no header) plus 60 daily test CSVs. Column names: `V1..V281`. Target: `V281` (last column).

**Paper reproduction:** Paper Table 2 shows n=39,355, p=275.

- **n:** A legacy CSV from the paper's authors has 56,222 rows; 56,222 × 0.7 = **39,355** — exact match. The UCI `blogData_train.csv` has 52,397 rows; all 60 test files combined add 7,624 rows (sizes vary: 83–181, avg 127); train + all test = 60,021 ≠ 56,222. The legacy CSV cannot be reconstructed from the current UCI files by any simple combination — it likely derives from a different version of the dataset or a different source.
- **p discrepancy:** The train file has 280 features (281 cols minus target). Four feature columns are zero-variance, giving p=276 after dropping — still 1 more than the paper's 275. Whether the legacy composite has a fifth zero-variance column (restoring p=275) is unverified.

**Decision:** Use `blogData_train.csv` only. The n/p discrepancy relative to the paper is accepted for now; possible modifications (e.g. sourcing the legacy CSV from a stable URL) are deferred until performance figures have been assessed.

### CT Slices

Source: UCI dataset #206 "Relative location of CT slices on axial axis". CDN URL: `cdn.uci-ics-mlr-prod.aws.uci.edu/206/relative%2Blocation%2Bct%2Bslices%2Bon%2Baxial%2Baxis.zip`. The dataset has a named header; target column is `reference`. Full dataset: 53,500 rows.

**Paper reproduction:** 53,500 × 0.7 = **37,450** — exactly matches paper Table 2 n. p discrepancy: UCI reports 386 features (385 after excluding target); paper shows p=379. The source of the 6-column difference is unresolved (possibly zero-variance or duplicate columns dropped by the paper's preprocessing).

**Source availability:** The CDN URL currently returns HTTP 403 for programmatic access (urllib), even though the beta archive website is accessible in a browser. The internal ZIP filename is likely `slice_localization_data.csv` (inferred from old archive naming — unverified). No working fallback exists without `from_ucimlrepo`. The entry is added with the CDN URL; it will fail until the CDN restriction is lifted or an alternative URL is found.

## Implementation

### Shared helpers and new factory function

Two private `_read_X_entry` helpers with matching signatures encapsulate the context managers for zip and tar respectively. `from_zip_tar` takes an explicit `zip_entry` (the tar filename inside the zip) and `tar_entry` (the CSV path inside the tar), composing the two helpers — no auto-detection needed.

```python
def _read_zip_entry(url, entry):
    with urllib.request.urlopen(url) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    return zf.read(entry)


def _read_tar_entry(data, entry):
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        return tf.extractfile(entry).read()


def from_zip(url, entry, **read_csv_kwargs):
    """Return a source callable that downloads a ZIP and reads one entry as a DataFrame.

    Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        return pd.read_csv(
            io.StringIO(_read_zip_entry(url, entry).decode('latin-1')), **read_csv_kwargs)
    return source


def from_zip_tar(url, zip_entry, tar_entry, **read_csv_kwargs):
    """Return a source callable that downloads a ZIP containing a tar and reads one entry.

    zip_entry is the tar filename inside the ZIP; tar_entry is the CSV path inside the tar.
    Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        return pd.read_csv(
            io.StringIO(_read_tar_entry(_read_zip_entry(url, zip_entry), tar_entry).decode('latin-1')),
            **read_csv_kwargs)
    return source
```

### DATASETS entries

```python
'twitter':   {'sources': [from_zip_tar(
    'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
    'regression.tar.gz',
    './regression/Twitter/Twitter.data',
    header=None, names=[f'V{i}' for i in range(1, 79)]
)]},
'tomshw':    {'sources': [from_zip_tar(
    'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
    'regression.tar.gz',
    './regression/TomsHardware/TomsHardware.data',
    header=None, names=[f'V{i}' for i in range(1, 98)]
)]},
'blog':      {'sources': [from_zip(
    'https://cdn.uci-ics-mlr-prod.aws.uci.edu/304/blogfeedback.zip',
    'blogData_train.csv',
    header=None, names=[f'V{i}' for i in range(1, 282)]
)]},
'ct_slices': {'sources': [
    from_zip(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/206/'
        'relative%2Blocation%2Bct%2Bslices%2Bon%2Baxial%2Baxis.zip',
        'slice_localization_data.csv'
    ),  # CDN currently returns 403; internal filename unverified
]},
```

### Notebook cells (appended at end of `experiments/real_data.ipynb`)

All new cells are skip-execution. The large datasets are never part of the preview (CI) section.

Cell structure:

```
[new]  problems_large definition     skip-execution
       EmpiricalDataProblem per dataset, d=1 targets:
         twitter   → target='V78'
         tomshw    → target='V97'
         blog      → target='V281'
         ct_slices → target='reference'

[new]  large d=1 experiment          skip-execution
       results_large = run_real_data_experiments(
           problems_large, estimators_full, n_iterations=30, seed=123, verbose=True)

[new]  large d=1 table               skip-execution

[new]  problems_large_d2 definition  skip-execution
       same four datasets, polynomial=2 targets same as d=1

[new]  large d=2 experiment          skip-execution
       results_large_d2 = run_real_data_experiments(
           problems_large_d2, estimators_full, n_iterations=30, polynomial=2, seed=123, verbose=True)

[new]  large d=2 table               skip-execution

[new]  combined figure call          skip-execution
       make_figure3(results_full + results_large,
                   results_full_d2 + results_large_d2,
                   results_full_d3,
                   output_path='../output/realdata_r2_by_degree.pdf')
       Same 2×3 structure as the existing figure; large dataset results are
       pooled into the d=1 and d=2 panels; d=3 panel unchanged.
```

**Exclusions:** CT Slices source is currently unavailable (CDN 403); it should be commented out in `problems_large` with a note, and reinstated once the source is resolved.

## Testing

Twitter and TomsHW require downloading ~33 MB and extracting a large tar; Blog and CT Slices are also large. No doctests will be added for these four datasets. Verification is via manual `get_dataset` calls after implementation to confirm shape and cached CSV.
