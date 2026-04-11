# Crime NaN Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the crime dataset SVD failures by (1) converting `?` to NaN in the cache reader, and (2) adding `nan_policy='drop_cols'` to `EmpiricalDataProblem` so crime can be included in the full experiment.

**Architecture:** Two-part change: `get_dataset()` gains a module-level `_EXTRA_NA_VALUES = ['?']` constant and is restructured to a single return point that always reads from the cache file with `na_values=_EXTRA_NA_VALUES`. `EmpiricalDataProblem.get_X_y()` gains an `elif` branch for `nan_policy='drop_cols'` that calls `df.dropna(axis=1)`. Crime is then uncommented in the full experiment notebook cell only (not the preview cell, since crime.csv is not git-tracked).

**Tech Stack:** Python, pandas (`dropna`, `read_csv`), pytest doctests, nbmake.

---

### Task 1: Restructure `get_dataset()` with `_EXTRA_NA_VALUES`

**Files:**
- Modify: `experiments/data.py` (lines 130–168)

- [ ] **Step 1: Run existing doctests to establish baseline**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -15
```

Expected: all doctests PASS (yacht shape (308, 7), diabetes shape (442, 11)).

- [ ] **Step 2: Replace `get_dataset()` with restructured version**

In `experiments/data.py`, replace everything from the current `get_dataset` function (line 130) to the end of the file with:

```python
# Additional missing value markers beyond pandas defaults:
# '', 'NA', 'N/A', '#N/A', '#N/A N/A', '#NA', 'NaN', '-NaN', '-nan', 'nan',
# 'None', '<NA>', 'NULL', 'null', 'n/a', '1.#IND', '-1.#IND', '1.#QNAN', '-1.#QNAN'
_EXTRA_NA_VALUES = ['?']


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

    return pd.read_csv(cache_path, na_values=_EXTRA_NA_VALUES)
```

- [ ] **Step 3: Run doctests to confirm they still pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/data.py -v 2>&1 | tail -15
```

Expected: all doctests PASS. The yacht and diabetes results are unchanged because neither dataset contains `?`.

- [ ] **Step 4: Verify automobile NaN count is unchanged**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "
from experiments.data import get_dataset
df = get_dataset('automobile')
print('NaN count:', df.isna().sum().sum())  # expected: 59
print('Shape:', df.shape)                   # expected: (205, 26)
"
```

Expected output:
```
NaN count: 59
Shape: (205, 26)
```

This confirms `na_values=['?']` has no unintended effect on automobile (no `?` in that file).

- [ ] **Step 5: Commit**

```bash
git add experiments/data.py
git commit -m "feat: restructure get_dataset() to single return point with na_values=['?']"
```

---

### Task 2: Add `nan_policy='drop_cols'` to `EmpiricalDataProblem`

**Files:**
- Modify: `experiments/problems.py` (module docstring + `get_X_y()` method)

- [ ] **Step 1: Add failing doctests for `drop_cols`**

In `experiments/problems.py`, extend the module docstring. After the last existing doctest line (`True`), add:

```python
>>> p_auto_cols = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_cols')
>>> X_auto_cols, y_auto_cols = p_auto_cols.get_X_y()
>>> X_auto_cols.shape
(201, 19)
>>> list(X_auto_cols.index) == list(y_auto_cols.index) == list(range(201))
True
```

Explanation of expected values: automobile has 205 rows, 4 rows have NaN in the `price` target column (dropped), leaving 201 rows. Of the remaining 25 feature columns, 6 have any NaN and are dropped by `drop_cols`, leaving 19 columns.

- [ ] **Step 2: Run doctests to confirm the new test fails**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -15
```

Expected: FAIL — `AssertionError` because `drop_cols` is not yet implemented (current code has no branch for it, so the shape will be (201, 25) not (201, 19)).

- [ ] **Step 3: Implement `drop_cols` in `get_X_y()`**

In `experiments/problems.py`, in the `get_X_y()` method, add the `drop_cols` branch after the `drop_rows` branch:

```python
    def get_X_y(self):
        df = get_dataset(self.dataset)
        missing = [c for c in self.drop if c not in df.columns]
        if missing:
            # warn rather than error — drop list may include columns absent in some sources
            warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
        df = df.drop(columns=[c for c in self.drop if c in df.columns])
        df = df.dropna(subset=[self.target])
        if self.nan_policy == 'drop_rows':
            df = df.dropna()
        elif self.nan_policy == 'drop_cols':
            df = df.dropna(axis=1)
        df = df.reset_index(drop=True)
        return df.drop(columns=[self.target]), df[self.target]
```

- [ ] **Step 4: Run doctests to confirm all pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --doctest-modules experiments/problems.py -v 2>&1 | tail -15
```

Expected: all doctests PASS, including the new `drop_cols` test with shape `(201, 19)`.

- [ ] **Step 5: Commit**

```bash
git add experiments/problems.py
git commit -m "feat: add nan_policy='drop_cols' to EmpiricalDataProblem"
```

---

### Task 3: Uncomment crime in the full experiment notebook cell

**Files:**
- Modify: `experiments/real_data.ipynb` (full experiment cell only)

**Important:** Close the notebook in VSCode before editing — the extension rewrites the file on save and causes write conflicts with `NotebookEdit`.

- [ ] **Step 1: Read the notebook to find the correct cell ID**

Use the Read tool on `experiments/real_data.ipynb` and locate the full experiment cell (cell-5 in the current structure). It contains `problems_full = [` and has `# EmpiricalDataProblem('crime', ...)` commented out.

- [ ] **Step 2: Replace the commented crime line in cell-5**

In cell-5, replace:
```python
    # EmpiricalDataProblem('crime',          'ViolentCrimesPerPop'),             # SVM non-converging for some run
```

with:
```python
    EmpiricalDataProblem('crime',          'ViolentCrimesPerPop',
                         drop=['state', 'fold', 'communityname'],
                         nan_policy='drop_cols'),
```

- [ ] **Step 3: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add crime to full experiment with drop_cols nan_policy"
```

---

### Task 4: Run full test suite

- [ ] **Step 1: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass. The `--nbmake` run of `real_data.ipynb` will skip cell-5 (tagged `skip-execution`); the preview cell (cell-2) runs and must complete without error. The preview cell does not include crime.

- [ ] **Step 2: If tests pass, proceed to finishing-a-development-branch**

Use `superpowers:finishing-a-development-branch` to merge to main.
