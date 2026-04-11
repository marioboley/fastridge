# Ribo Dataset Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `fetch_riboflavin` as a source for the `ribo` dataset entry so it can be included in the full real-data experiment.

**Architecture:** Three new functions in `experiments/data.py`: `_fetch_cran_rdata` (private fetch+parse helper), `from_cran` (public source factory for flat R data.frames), and `fetch_riboflavin` (public source callable with custom matrix unpacking). `rdata` is added to `requirements.txt` and imported inline inside the functions that use it. The `ribo` DATASETS entry is wired to `fetch_riboflavin`. The full experiment notebook cell is updated to include `EmpiricalDataProblem('ribo', 'y')`.

**Tech Stack:** Python, `rdata` (pure Python R data file reader), `tarfile` + `urllib.request` (stdlib), pandas, numpy, pytest doctests, nbmake.

---

### Task 1: Add `rdata` to requirements and install

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add `rdata` to requirements.txt**

Open `requirements.txt`. It currently ends with:
```
ucimlrepo>=0.0.3
tabulate>=0.9
```

Add one line so it becomes:
```
ucimlrepo>=0.0.3
tabulate>=0.9
rdata>=0.10
```

(The version already installed is compatible with `>=0.10`.)

- [ ] **Step 2: Verify it installs cleanly**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pip install -r requirements.txt 2>&1 | tail -5
```

Expected: `rdata` already satisfied (already installed in venv), no errors.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add rdata to requirements for CRAN R data file retrieval"
```

---

### Task 2: Add `_fetch_cran_rdata`, `from_cran`, and `fetch_riboflavin` to `data.py` (TDD)

**Files:**
- Modify: `experiments/data.py`

- [ ] **Step 1: Add a failing doctest for `fetch_riboflavin`**

In `experiments/data.py`, extend the module docstring. After the existing doctests:

```python
"""
Dataset registry and retrieval for real data experiments.

>>> df = get_dataset('yacht')
>>> df.shape
(308, 7)
>>> df = get_dataset('diabetes')
>>> df.shape
(442, 11)
>>> from data import fetch_riboflavin
>>> df_ribo = fetch_riboflavin()
>>> df_ribo.shape
(71, 4089)
>>> list(df_ribo.columns[:3])
['y', 'AADK_at', 'AAPA_at']
>>> df_ribo['y'].dtype.kind
'f'
"""
```

- [ ] **Step 2: Run doctests to confirm the new test fails**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name 'fetch_riboflavin'`.

- [ ] **Step 3: Add numpy to top-level imports and insert the three new functions**

In `experiments/data.py`, the current top-level imports (lines 11–17) are:
```python
import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd
```

Add `import numpy as np` after `import pandas as pd`:
```python
import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
```

Then, after the `from_zip` function (line 72) and before the `from sklearn.datasets` import (line 75), insert:

```python
# Optional dependency — rdata (pure Python R data file reader):
# used by _fetch_cran_rdata / from_cran / fetch_riboflavin


def _fetch_cran_rdata(package, version, rdata_path):
    """Download a CRAN package tarball and return the parsed R object at rdata_path.

    Fetches https://cran.r-project.org/src/contrib/{package}_{version}.tar.gz,
    extracts the file at rdata_path, parses it with rdata.parser.parse_file,
    and returns the parsed R object (not yet converted to a DataFrame).
    """
    import tarfile
    import tempfile
    import rdata
    url = f'https://cran.r-project.org/src/contrib/{package}_{version}.tar.gz'
    raw = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        f = t.extractfile(rdata_path)
        tmp = tempfile.NamedTemporaryFile(suffix='.RData', delete=False)
        tmp.write(f.read())
        tmp.close()
    return rdata.parser.parse_file(tmp.name)


def from_cran(package, version, rdata_path):
    """Return a source callable that fetches an R dataset from a CRAN package.

    The RData file at rdata_path must contain a single flat R data.frame that
    rdata.conversion.convert() can map directly to a pandas DataFrame.
    For datasets with non-standard R structures (e.g. matrix-valued columns),
    write a dedicated source callable that calls _fetch_cran_rdata directly.
    """
    def source():
        import rdata
        parsed = _fetch_cran_rdata(package, version, rdata_path)
        converted = rdata.conversion.convert(parsed)
        return converted[list(converted.keys())[0]]
    return source


def fetch_riboflavin():
    """Fetch the riboflavin dataset from the hdi CRAN package (version 0.1-10).

    Returns a DataFrame with 71 rows and 4089 columns: 'y' (log riboflavin
    production rate) followed by 4088 gene expression columns (e.g. 'AADK_at').
    Row index contains microarray chip IDs.

    The hdi RData structure is a data.frame with a matrix-valued column x
    (71x4088) and a scalar column y — not convertible by rdata's standard
    converter, hence the custom unpacking below.
    """
    from rdata.parser._parser import RObjectType
    parsed = _fetch_cran_rdata('hdi', '0.1-10', 'hdi/data/riboflavin.RData')
    robj = parsed.object.value[0]
    y_robj = robj.value[0]   # REAL vector, length 71
    x_robj = robj.value[1]   # REAL vector, length 71*4088, with dim/dimnames attrs

    def _pairlist_to_dict(obj):
        result = {}
        cur = obj
        while cur is not None and cur.info.type not in (RObjectType.NILVALUE,):
            if cur.info.type == RObjectType.LIST:
                tag = cur.tag
                if tag is not None:
                    name = tag.value.value
                    if isinstance(name, bytes):
                        name = name.decode()
                    result[name] = cur.value[0]
                cur = cur.value[1] if len(cur.value) > 1 else None
            else:
                break
        return result

    def _extract_strvec(obj):
        if obj.info.type in (RObjectType.VEC, RObjectType.STR):
            return [v.value.decode() if isinstance(v.value, bytes) else str(v.value)
                    for v in obj.value]
        return []

    x_attrs = _pairlist_to_dict(x_robj.attributes)
    dim = [int(v) for v in x_attrs['dim'].value]
    dimnames = x_attrs['dimnames'].value
    rownames = _extract_strvec(dimnames[0])
    colnames = _extract_strvec(dimnames[1])

    x = pd.DataFrame(
        np.array(x_robj.value).reshape(dim, order='F'),  # R stores column-major
        index=rownames,
        columns=colnames,
    )
    y = pd.Series(np.array(y_robj.value), index=rownames, name='y')
    return pd.concat([y, x], axis=1).reset_index(drop=True)

- [ ] **Step 4: Wire `fetch_riboflavin` into the DATASETS registry**

In `DATASETS`, replace:
```python
    'ribo':             {'sources': []},  # URL TBD
```
with:
```python
    'ribo':             {'sources': [fetch_riboflavin]},
```

- [ ] **Step 5: Run doctests to confirm they pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -15
```

Expected: all doctests PASS including the new ribo test (shape `(71, 4089)`, first 3 columns `['y', 'AADK_at', 'AAPA_at']`).

Note: this makes a network request to CRAN. If CRAN is unreachable the test will fail with a `URLError` — that is expected and not a code bug.

- [ ] **Step 6: Commit**

```bash
git add experiments/data.py
git commit -m "feat: add _fetch_cran_rdata, from_cran, fetch_riboflavin; wire ribo dataset"
```

---

### Task 3: Add `ribo` to the full experiment notebook cell

**Files:**
- Modify: `experiments/real_data.ipynb` (full experiment cell, tagged `skip-execution`)

**Important:** Close the notebook in VSCode before editing — the extension rewrites the file on save and causes write conflicts with `NotebookEdit`.

- [ ] **Step 1: Read the notebook to confirm the full experiment cell ID**

Use the Read tool on `experiments/real_data.ipynb`. Locate the full experiment cell (currently `cell-5`). Confirm it contains `problems_full = [` and the list of `EmpiricalDataProblem` entries.

- [ ] **Step 2: Add ribo to the full experiment cell**

In cell-5, insert after the crime entry:

```python
    EmpiricalDataProblem('ribo',             'y'),
```

The full updated `problems_full` list should be:

```python
problems_full = [
    EmpiricalDataProblem('abalone',          'Rings'),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('automobile',       'price',               nan_policy='drop_rows'),
    EmpiricalDataProblem('autompg',          'mpg',                 nan_policy='drop_rows'),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols'),
    EmpiricalDataProblem('ribo',             'y'),
    EmpiricalDataProblem('boston',           'medv'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',         'target'),
    EmpiricalDataProblem('facebook',         'Total Interactions',  nan_policy='drop_rows'),  # 'like' and 'share' are also candidate targets
    EmpiricalDataProblem('forest',           'area'),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay', drop=['GT_turbine_decay']),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',    drop=['GT_compressor_decay']),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',         drop=['total_UPDRS']),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',         drop=['motor_UPDRS']),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area'),
    EmpiricalDataProblem('student',          'G3'),
    EmpiricalDataProblem('yacht',            'Residuary_resistance'),
]
```

- [ ] **Step 3: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add ribo to full experiment cell"
```

---

### Task 4: Run full test suite

- [ ] **Step 1: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass. The `--nbmake` run skips the full experiment cell (tagged `skip-execution`); the preview cell runs and must complete without error (`ribo` is not in the preview cell so no network request to CRAN is made during CI).

- [ ] **Step 2: If tests pass, proceed to finishing-a-development-branch**

Use `superpowers:finishing-a-development-branch` to push to dev and merge to main.
