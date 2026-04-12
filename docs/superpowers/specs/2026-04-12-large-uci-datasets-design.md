# Large UCI Datasets Design

## Goal

Add four large datasets — Twitter, TomsHW, Blog, and CT Slices — to `experiments/data.py`. These datasets appear in paper Table 2 but were absent from the legacy notebook (commented out as "data file size is too big"). All sources use the beta UCI archive CDN (`cdn.uci-ics-mlr-prod.aws.uci.edu`).

## Datasets

### Twitter and TomsHW

Both reside in UCI dataset #248 "Buzz in Social Media" (CDN ZIP: `buzz+in+social+media.zip`, ~33 MB). The ZIP contains a single `regression.tar.gz`, which in turn contains:

- `regression/Twitter/Twitter.data` — 583 K rows × 78 cols, no header
- `regression/TomsHardware/TomsHardware.data` — ~28 K rows × 97 cols, no header

Column names: `V1..V78` (Twitter) and `V1..V97` (TomsHW). Target is the last column in each case (`V78` and `V97`). Paper values: Twitter n=408,275 p=77; TomsHW n=19,725 p=96 (70% of full rows).

### Blog

UCI dataset #304 "BlogFeedback" (CDN ZIP: `blogfeedback.zip`, ~2.5 MB). Contains `blogData_train.csv` (52,397 rows × 281 cols, no header) plus 60 daily test CSVs. Column names: `V1..V281`. Target is last column `V281`.

**Discrepancy:** Paper Table 2 shows n=39,355, p=275. Using `blogData_train.csv` with a 70/30 split gives n≈36,678 (not 39,355), and the file has 280 features (not 275). Possible explanations: the paper used a combined train+test split on a different total; or used a different preprocessing that drops 5 columns.

**Decision:** Use `blogData_train.csv` as the source for now — it is the canonical training split and the only single-file option that avoids merging 61 files.

**Alternatives to consider later:**
- Concatenate `blogData_train.csv` + all 60 test CSVs then use a 70/30 split at experiment time (larger n, may approach paper's 39,355).
- Investigate source discrepancy: check if dropping constant or near-zero-variance columns from the 280 features recovers p=275.

### CT Slices

UCI dataset #206 "Relative location of CT slices on axial axis" (53,500 rows). Paper Table 2: n=37,450 (= 53,500 × 0.7), p=379. Dataset has a named header; target column is `reference`.

CDN URL (`cdn.uci-ics-mlr-prod.aws.uci.edu/206/relative%2Blocation%2Bct%2Bslices%2Bon%2Baxial%2Baxis.zip`) currently returns HTTP 403 for programmatic access (urllib). Internal ZIP filename is likely `slice_localization_data.csv` (inferred from old archive naming convention — unverified).

**Decision:** Add `from_zip` with the CDN URL as primary source, `from_ucimlrepo(206)` as fallback. If CDN becomes accessible, confirm ZIP entry filename and remove the fallback if desired.

**p discrepancy:** Paper p=379 vs UCI-reported 386 features (385 after excluding target). Source of the difference (dropped zero-variance or constant columns) is unresolved — to be investigated when experiment results are available.

## Implementation

### New factory function: `from_zip_tar`

Add to `experiments/data.py` after the existing `from_zip`:

```python
def from_zip_tar(url, tar_entry, **read_csv_kwargs):
    """Return a source callable that downloads a ZIP containing a tar.gz and reads one entry.

    The ZIP must contain exactly one .tar.gz member. tar_entry is the path within
    that tar archive. Extra keyword arguments are forwarded to pd.read_csv.
    """
    def source():
        with urllib.request.urlopen(url) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        tar_name = next(n for n in zf.namelist() if n.endswith('.tar.gz') or n.endswith('.tar'))
        with tarfile.open(fileobj=io.BytesIO(zf.read(tar_name))) as tf:
            data = tf.extractfile(tar_entry).read().decode('latin-1')
        return pd.read_csv(io.StringIO(data), **read_csv_kwargs)
    return source
```

### DATASETS entries

```python
'twitter':   {'sources': [from_zip_tar(
    'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
    './regression/Twitter/Twitter.data',
    header=None, names=[f'V{i}' for i in range(1, 79)]
)]},
'tomshw':    {'sources': [from_zip_tar(
    'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
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
    ),   # CDN — currently returns 403 programmatically; filename unverified
    from_ucimlrepo(206),
]},
```

### No notebook changes

Dataset registration only. Notebook problem list additions (target column, exclusion decisions for polynomial experiments) are a separate task.

## Testing

The existing `get_dataset` doctest pattern is used for datasets that are fast to fetch. Twitter and TomsHW require downloading ~33 MB and extracting a large tar, so they are **not** suitable for doctest. CT Slices and Blog are also large. No doctests will be added for these four datasets.

Smoke-test via manual `get_dataset('twitter')` call after implementation to verify the cache file is produced correctly and the shape matches expectations (full rows × 78 for Twitter, etc.).
