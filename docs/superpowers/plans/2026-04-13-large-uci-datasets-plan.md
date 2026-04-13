# Large UCI Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register Twitter, TomsHW, Blog, and CT Slices in `experiments/data.py` and append d=1 and d=2 experiment/table/figure cells (all skip-execution) to `experiments/real_data.ipynb`.

**Architecture:** Refactor `from_zip` to use a private `_read_zip_entry` helper; add a symmetric `_read_tar_entry` helper and `from_zip_tar` factory (for the ZIP→tar.gz→CSV structure of Buzz in Social Media); add four DATASETS entries. Seven notebook cells are appended at the end: problems_large + d=1 experiment + table, problems_large_d2 + d=2 experiment + table, combined figure reusing the existing `make_figure3`.

**Tech Stack:** `experiments/data.py`, `experiments/real_data.ipynb`, stdlib `zipfile`/`tarfile`/`urllib`, NotebookEdit tool.

---

## Files

- Modify: `experiments/data.py` — add helpers, refactor `from_zip`, add `from_zip_tar`, add 4 DATASETS entries
- Modify: `experiments/real_data.ipynb` — append 7 skip-execution cells

---

### Task 1: Add helpers, refactor `from_zip`, add `from_zip_tar`

**Files:** `experiments/data.py:56-76`

- [ ] **Step 1: Replace the existing `from_zip` block with the three new functions**

Read `experiments/data.py` first to confirm the current `from_zip` block. Replace it with:

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
            io.StringIO(
                _read_tar_entry(
                    _read_zip_entry(url, zip_entry), tar_entry).decode('latin-1')),
            **read_csv_kwargs)
    return source
```

- [ ] **Step 2: Run doctests to verify nothing broke**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -20
```

Expected: yacht (308, 7) and diabetes (442, 11) doctests pass.

- [ ] **Step 3: Commit**

```bash
git add experiments/data.py
git commit -m "refactor: extract _read_zip_entry/_read_tar_entry helpers, add from_zip_tar"
```

---

### Task 2: Add four new DATASETS entries

**Files:** `experiments/data.py` — DATASETS dict, after the `'ribo'` entry, before `'crop'`.

- [ ] **Step 1: Insert the four entries**

Add after `'ribo': {'sources': [fetch_riboflavin]},`:

```python
    'blog':      {'sources': [from_zip(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/304/blogfeedback.zip',
        'blogData_train.csv',
        header=None, names=[f'V{i}' for i in range(1, 282)]
    )]},
    # ct_slices CDN currently returns 403 programmatically; ucimlrepo added as fallback.
    # Internal ZIP filename 'slice_localization_data.csv' is inferred — unverified.
    'ct_slices': {'sources': [
        from_zip(
            'https://cdn.uci-ics-mlr-prod.aws.uci.edu/206/'
            'relative%2Blocation%2Bct%2Bslices%2Bon%2Baxial%2Baxis.zip',
            'slice_localization_data.csv'
        ),
        from_ucimlrepo(206),
    ]},
    'tomshw':    {'sources': [from_zip_tar(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
        'regression.tar.gz',
        './regression/TomsHardware/TomsHardware.data',
        header=None, names=[f'V{i}' for i in range(1, 98)]
    )]},
    'twitter':   {'sources': [from_zip_tar(
        'https://cdn.uci-ics-mlr-prod.aws.uci.edu/248/buzz%2Bin%2Bsocial%2Bmedia.zip',
        'regression.tar.gz',
        './regression/Twitter/Twitter.data',
        header=None, names=[f'V{i}' for i in range(1, 79)]
    )]},
```

- [ ] **Step 2: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add experiments/data.py
git commit -m "feat: register twitter, tomshw, blog, ct_slices datasets in data.py"
```

---

### Task 3: Append problems_large and d=1 experiment/table cells

**Files:** `experiments/real_data.ipynb` — 3 new cells appended (all skip-execution). The notebook must not be open in VSCode during edits.

The skip-execution tagging script (reuse for every new cell in Tasks 3–5):

```bash
python3 -c "
import json
with open('experiments/real_data.ipynb') as f: nb = json.load(f)
cell = nb['cells'][-1]
cell.setdefault('metadata', {}).setdefault('tags', [])
if 'skip-execution' not in cell['metadata']['tags']:
    cell['metadata']['tags'].append('skip-execution')
with open('experiments/real_data.ipynb', 'w') as f: json.dump(nb, f, indent=1)
print('tagged cell', len(nb['cells']) - 1)
"
```

- [ ] **Step 1: Read notebook to find last cell ID**

Read `experiments/real_data.ipynb`. Note the `id` field of the last cell (currently idx 22). Use it as the `cell_id` for the next NotebookEdit insert.

- [ ] **Step 2: Insert problems_large cell after last cell**

```python
problems_large = [
    EmpiricalDataProblem('twitter', 'V78'),
    EmpiricalDataProblem('tomshw', 'V97'),
    EmpiricalDataProblem('blog',   'V281'),
    # EmpiricalDataProblem('ct_slices', 'reference'),  # CDN source unavailable (403)
]
```

Use NotebookEdit with `edit_mode="insert"`, `cell_type="code"`, `cell_id=<last cell id>`.

- [ ] **Step 3: Tag problems_large cell skip-execution** (run tagging script above)

- [ ] **Step 4: Read notebook to find new last cell ID, insert large d=1 experiment cell**

```python
results_large = run_real_data_experiments(problems_large, estimators_full,
                                          n_iterations=30, seed=123, verbose=True)
print()
```

- [ ] **Step 5: Tag d=1 experiment cell skip-execution** (run tagging script)

- [ ] **Step 6: Read notebook to find new last cell ID, insert large d=1 table cell**

```python
rows_large = []
for problem, data_result in zip(problems_large, results_large):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = data_result['EM']['p']
    row['n_train']        = data_result['EM']['n_train']
    row['n:p']            = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_large.append(row)
pd.DataFrame(rows_large).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 7: Tag d=1 table cell skip-execution** (run tagging script)

- [ ] **Step 8: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add large dataset d=1 experiment and table cells"
```

---

### Task 4: Append problems_large_d2 and d=2 experiment/table cells

**Files:** `experiments/real_data.ipynb` — 3 more cells appended (all skip-execution).

- [ ] **Step 1: Read notebook to find last cell ID, insert problems_large_d2 cell**

```python
problems_large_d2 = [
    EmpiricalDataProblem('twitter', 'V78'),
    EmpiricalDataProblem('tomshw', 'V97'),
    EmpiricalDataProblem('blog',   'V281'),
    # EmpiricalDataProblem('ct_slices', 'reference'),  # CDN source unavailable (403)
]
```

- [ ] **Step 2: Tag problems_large_d2 cell skip-execution** (run tagging script from Task 3)

- [ ] **Step 3: Read notebook to find new last cell ID, insert large d=2 experiment cell**

```python
results_large_d2 = run_real_data_experiments(problems_large_d2, estimators_full,
                                              n_iterations=30, polynomial=2,
                                              seed=123, verbose=True)
print()
```

- [ ] **Step 4: Tag d=2 experiment cell skip-execution** (run tagging script)

- [ ] **Step 5: Read notebook to find new last cell ID, insert large d=2 table cell**

```python
rows_large_d2 = []
for problem, data_result in zip(problems_large_d2, results_large_d2):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = data_result['EM']['p']
    row['n_train']        = data_result['EM']['n_train']
    row['n:p']            = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_large_d2.append(row)
pd.DataFrame(rows_large_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 6: Tag d=2 table cell skip-execution** (run tagging script)

- [ ] **Step 7: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add large dataset d=2 experiment and table cells"
```

---

### Task 5: Append combined figure cell

**Files:** `experiments/real_data.ipynb` — 1 cell appended (skip-execution).

- [ ] **Step 1: Read notebook to find last cell ID, insert combined figure cell**

```python
make_figure3(results_full + results_large,
             results_full_d2 + results_large_d2,
             results_full_d3,
             output_path='../output/realdata_r2_by_degree.pdf')
```

- [ ] **Step 2: Tag combined figure cell skip-execution** (run tagging script from Task 3)

- [ ] **Step 3: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add combined figure cell for all datasets by degree"
```

---

### Task 6: Run tests and push

- [ ] **Step 1: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass. The 7 new notebook cells are tagged skip-execution and do not execute in CI.

- [ ] **Step 2: Push to dev and wait for CI**

```bash
git push origin dev
```

- [ ] **Step 3: When CI passes, use `superpowers:finishing-a-development-branch` to merge to main**
