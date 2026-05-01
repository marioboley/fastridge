# Timing Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track normalization and SVD time separately in both estimators, add metrics for those components, extract all problem/estimator definitions from `real_data.ipynb` into `journal2026.py`, refactor the notebook to import from the module, and create `timing_analysis.ipynb` with tables and stacked bar charts.

**Architecture:** Tasks 1 and 2 are independent. Task 3 (new module) is a prerequisite for Tasks 4 and 5. The module becomes the single source of truth so that `real_data.ipynb` and `timing_analysis.ipynb` share identical problem definitions without duplication.

**Tech Stack:** Python 3, numpy, scipy, scikit-learn, matplotlib, pandas, pytest, nbmake, nbformat 4.5

---

### Task 1: Add normalization_time_ to RidgeEM.fit and RidgeLOOCV.fit

**Files:**
- Modify: `fastridge.py:281-291` (RidgeEM), `fastridge.py:383-399` (RidgeLOOCV)
- Test: `tests/test_fastridge.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_fastridge.py`:

```python
def test_ridge_em_normalization_time():
    X, y, _ = _data()
    est = RidgeEM().fit(X, y)
    assert hasattr(est, 'normalization_time_')
    assert isinstance(est.normalization_time_, float)
    assert est.normalization_time_ >= 0


def test_ridge_loocv_normalization_time():
    X, y, _ = _data()
    est = RidgeLOOCV(alphas=5).fit(X, y)
    assert hasattr(est, 'normalization_time_')
    assert isinstance(est.normalization_time_, float)
    assert est.normalization_time_ >= 0
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_fastridge.py::test_ridge_em_normalization_time tests/test_fastridge.py::test_ridge_loocv_normalization_time -v
```
Expected: FAIL with `AttributeError: 'RidgeEM' object has no attribute 'normalization_time_'`

- [ ] **Step 3: Implement in RidgeEM.fit**

In `fastridge.py`, replace the block at lines 281–289 (from `a_x = x.mean(...)` through the blank line before `svd_start_time`):

```python
        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y = (y - a_y) / b_y

        svd_start_time = time.time()
```

with:

```python
        norm_start_time = time.time()
        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y = (y - a_y) / b_y
        self.normalization_time_ = time.time() - norm_start_time

        svd_start_time = time.time()
```

- [ ] **Step 4: Implement in RidgeLOOCV.fit**

In `fastridge.py`, replace the block at lines 383–397 (from `a_x = x.mean(...)` through the blank line before `if np.isscalar(self.alphas):`):

```python
        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y = (y - a_y) / b_y

        if np.isscalar(self.alphas):
```

with:

```python
        norm_start_time = time.time()
        a_x = x.mean(axis=0) if self.fit_intercept else np.zeros(p)
        b_x = x.std(axis=0) if self.normalize else np.ones(p)
        x = (x - a_x) / b_x

        a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
        b_y = y.std(axis=0) if self.normalize else np.ones(q)
        y = (y - a_y) / b_y
        self.normalization_time_ = time.time() - norm_start_time

        if np.isscalar(self.alphas):
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_fastridge.py::test_ridge_em_normalization_time tests/test_fastridge.py::test_ridge_loocv_normalization_time -v
```
Expected: PASS

- [ ] **Step 6: Run full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add fastridge.py tests/test_fastridge.py
git commit -m "feat: add normalization_time_ attribute to RidgeEM and RidgeLOOCV fit"
```

---

### Task 2: Add SvdTime and NormalizationTime metric classes

**Files:**
- Modify: `experiments/experiments.py` (after `fitting_time = FittingTime()` on line 333)

- [ ] **Step 1: Verify the import fails**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "from experiments import svd_time"
```
Expected: `ImportError: cannot import name 'svd_time'`

- [ ] **Step 2: Insert classes and instances into experiments/experiments.py**

After the line `fitting_time = FittingTime()` and before `prediction_r2 = PredictionR2()`, insert:

```python
class SvdTime(FittingTime):
    """Fitting time spent in SVD decomposition.

    Examples
    --------
    >>> class _E:
    ...     svd_time_ = 0.42
    >>> svd_time(_E(), None, None, None)
    0.42
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return est.svd_time_

    def __str__(self):
        return 'svd_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{svd}$ [s]'


class NormalizationTime(FittingTime):
    """Fitting time spent in input normalization.

    Examples
    --------
    >>> class _E:
    ...     normalization_time_ = 0.07
    >>> normalization_time(_E(), None, None, None)
    0.07
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return est.normalization_time_

    def __str__(self):
        return 'normalization_time'

    @staticmethod
    def symbol():
        return r'$T_\mathrm{norm}$ [s]'


svd_time = SvdTime()
normalization_time = NormalizationTime()
```

- [ ] **Step 3: Run full test suite (includes doctests via --doctest-modules)**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: add SvdTime and NormalizationTime metrics to experiments"
```

---

### Task 3: Create experiments/journal2026.py

**Files:**
- Create: `experiments/journal2026.py`
- Create: `tests/test_journal2026.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_journal2026.py`:

```python
import numpy as np
import pytest
from journal2026 import (
    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE, JOURNAL2026_D1,
    JOURNAL2026_D1_PREVIEW,
    JOURNAL2026_D2_REGULAR, JOURNAL2026_D2_LARGE, JOURNAL2026_D2,
    JOURNAL2026_D2_PREVIEW,
    JOURNAL2026_D3_REGULAR, JOURNAL2026_D3, JOURNAL2026_D3_PREVIEW,
    JOURNAL2026_ESTIMATORS, JOURNAL2026_EST_NAMES,
    TIMING_ESTIMATORS, TIMING_EST_NAMES,
    JOURNAL2026_TRAIN_SIZES,
)


def test_d1_counts():
    assert len(JOURNAL2026_D1_REGULAR) == 17
    assert len(JOURNAL2026_D1_LARGE) == 4
    assert len(JOURNAL2026_D1) == 21
    assert len(JOURNAL2026_D1_PREVIEW) == 9


def test_d2_counts():
    assert len(JOURNAL2026_D2_REGULAR) == 16
    assert len(JOURNAL2026_D2_LARGE) == 4
    assert len(JOURNAL2026_D2) == 20
    assert len(JOURNAL2026_D2_PREVIEW) == 9


def test_d3_counts():
    assert len(JOURNAL2026_D3_REGULAR) == 15
    assert len(JOURNAL2026_D3) == 15
    assert len(JOURNAL2026_D3_PREVIEW) == 8


def test_estimator_names():
    assert JOURNAL2026_EST_NAMES == ['EM', 'CV_fix', 'CV_glm']
    assert len(JOURNAL2026_ESTIMATORS) == 3
    assert TIMING_EST_NAMES == ['EM', 'CV_glm_101', 'CV_glm_11']
    assert len(TIMING_ESTIMATORS) == 3


def test_train_sizes_covers_all_d1_datasets():
    all_datasets = {p.dataset for p in JOURNAL2026_D1}
    assert all_datasets <= set(JOURNAL2026_TRAIN_SIZES)


def test_d3_preview_excludes_eye():
    datasets = {p.dataset for p in JOURNAL2026_D3_PREVIEW}
    assert 'eye' not in datasets


def test_d2_regular_excludes_ribo():
    datasets = {p.dataset for p in JOURNAL2026_D2_REGULAR}
    assert 'ribo' not in datasets
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_journal2026.py::test_d1_counts -v
```
Expected: `ModuleNotFoundError: No module named 'journal2026'`

- [ ] **Step 3: Create experiments/journal2026.py**

```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from problems import EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric
from neurips2023 import NEURIPS2023_TRAIN_SIZES


JOURNAL2026_TRAIN_SIZES = NEURIPS2023_TRAIN_SIZES

JOURNAL2026_D1_REGULAR = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows', x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ribo',             'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         zero_variance_filter=True),
]

JOURNAL2026_D1_LARGE = [
    EmpiricalDataProblem('twitter',   'V78',       zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',    'V97',       zero_variance_filter=True),
    EmpiricalDataProblem('blog',      'V281',      zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices', 'reference', zero_variance_filter=True),
]

JOURNAL2026_D1 = JOURNAL2026_D1_REGULAR + JOURNAL2026_D1_LARGE

JOURNAL2026_D1_PREVIEW = [
    EmpiricalDataProblem('abalone',    'Rings',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',   'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',        'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',     'area',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile', 'price',
                         nan_policy='drop_rows', x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
]

JOURNAL2026_D2_REGULAR = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
]

JOURNAL2026_D2_LARGE = [
    EmpiricalDataProblem('twitter',   'V78',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',    'V97',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('blog',      'V281',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices', 'reference',
                         x_transforms=(PolynomialExpansion(2),), zero_variance_filter=True),
]

JOURNAL2026_D2 = JOURNAL2026_D2_REGULAR + JOURNAL2026_D2_LARGE

JOURNAL2026_D2_PREVIEW = [
    EmpiricalDataProblem('abalone',    'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',   'target',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',        'y',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',     'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(2),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile', 'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(2),),
                         zero_variance_filter=True),
]

JOURNAL2026_D3_REGULAR = [
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'), nan_policy='drop_cols',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         ('comment', 'like', 'share'),
                         drop=('Total Interactions',), nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', ('GT_compressor_decay', 'GT_turbine_decay'),
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       ('motor_UPDRS', 'total_UPDRS'),
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
]

JOURNAL2026_D3 = JOURNAL2026_D3_REGULAR

JOURNAL2026_D3_PREVIEW = [
    EmpiricalDataProblem('abalone',    'Rings',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',   'target',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',     'area',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',    ('G1', 'G2', 'G3'),
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',      'Residuary_resistance',
                         x_transforms=(PolynomialExpansion(3),),
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile', 'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric + (PolynomialExpansion(3),),
                         zero_variance_filter=True),
]

JOURNAL2026_ESTIMATORS = [
    RidgeEM(),
    RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
    RidgeLOOCV(alphas=100),
]
JOURNAL2026_EST_NAMES = ['EM', 'CV_fix', 'CV_glm']

TIMING_ESTIMATORS = [RidgeEM(), RidgeLOOCV(alphas=101), RidgeLOOCV(alphas=11)]
TIMING_EST_NAMES  = ['EM', 'CV_glm_101', 'CV_glm_11']
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest tests/test_journal2026.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add experiments/journal2026.py tests/test_journal2026.py
git commit -m "feat: add journal2026.py with all problem collections, estimators, and timing estimators"
```

---

### Task 4: Refactor real_data.ipynb to import from journal2026

**Files:**
- Modify: `experiments/real_data.ipynb`

Note: Close this notebook in VSCode before editing.

The refactor removes 6 cells that only define problem/estimator variables (cells 229399e5, 994819b1, b12d0253, d245ce96, 502c1187, e4ee47eb) and updates 8 cells that run experiments to use the imported constants.

- [ ] **Step 1: Read the notebook to confirm cell IDs**

```
Read experiments/real_data.ipynb
```
Verify cells 76d89b51, 229399e5, f6bc8769, 994819b1, 094956a8, 2bfbd407, b12d0253, 28f3974b, d245ce96, a96f98ee, 502c1187, 68c60ed3, e4ee47eb, 300ca33b all exist.

- [ ] **Step 2: Modify cell 76d89b51 (imports + preview d1 experiment)**

Replace source with:
```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import Experiment
from journal2026 import (
    JOURNAL2026_TRAIN_SIZES,
    JOURNAL2026_D1_PREVIEW, JOURNAL2026_D2_PREVIEW, JOURNAL2026_D3_PREVIEW,
    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE,
    JOURNAL2026_D2_REGULAR, JOURNAL2026_D2_LARGE,
    JOURNAL2026_D3_REGULAR,
    JOURNAL2026_ESTIMATORS, JOURNAL2026_EST_NAMES,
)

exp = Experiment(
    JOURNAL2026_D1_PREVIEW, JOURNAL2026_ESTIMATORS,
    reps=10, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_PREVIEW],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(ignore_cache=True)
print()
```

- [ ] **Step 3: Delete problem-only cells 229399e5 and 994819b1**

Use NotebookEdit with `action="delete"` for each cell.

- [ ] **Step 4: Modify cell f6bc8769 (preview d2 experiment)**

Replace source with:
```python
exp_d2 = Experiment(
    JOURNAL2026_D2_PREVIEW, JOURNAL2026_ESTIMATORS,
    reps=10, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D2_PREVIEW],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(ignore_cache=True)
print()
```

- [ ] **Step 5: Modify cell 094956a8 (preview d3 experiment)**

Replace source with:
```python
exp_d3 = Experiment(
    JOURNAL2026_D3_PREVIEW, JOURNAL2026_ESTIMATORS,
    reps=10, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D3_PREVIEW],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(ignore_cache=True)
print()
```

- [ ] **Step 6: Modify cell 2bfbd407 (full d1 experiment)**

Replace source with:
```python
exp_full = Experiment(
    JOURNAL2026_D1_REGULAR, JOURNAL2026_ESTIMATORS,
    reps=100, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_REGULAR],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(overwrite_cache=True)
print()
```

- [ ] **Step 7: Delete cell b12d0253**

- [ ] **Step 8: Modify cell 28f3974b (full d2 experiment)**

Replace source with:
```python
exp_full_d2 = Experiment(
    JOURNAL2026_D2_REGULAR, JOURNAL2026_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D2_REGULAR],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(overwrite_cache=True)
print()
```

- [ ] **Step 9: Delete cell d245ce96**

- [ ] **Step 10: Modify cell a96f98ee (full d3 experiment)**

Replace source with:
```python
exp_full_d3 = Experiment(
    JOURNAL2026_D3_REGULAR, JOURNAL2026_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D3_REGULAR],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(overwrite_cache=True)
print()
```

- [ ] **Step 11: Delete cell 502c1187**

- [ ] **Step 12: Modify cell 68c60ed3 (large d1 experiment)**

Replace source with:
```python
exp_large = Experiment(
    JOURNAL2026_D1_LARGE, JOURNAL2026_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_LARGE],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(overwrite_cache=True)
print()
```

- [ ] **Step 13: Delete cell e4ee47eb**

- [ ] **Step 14: Modify cell 300ca33b (large d2 experiment)**

Replace source with:
```python
exp_large_d2 = Experiment(
    JOURNAL2026_D2_LARGE, JOURNAL2026_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D2_LARGE],
    seed=123,
    est_names=JOURNAL2026_EST_NAMES).run(overwrite_cache=True)
print()
```

- [ ] **Step 15: Run full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: All pass. The nbmake run of real_data.ipynb exercises the preview cells (ignore_cache=True); full/large cells are tagged skip-execution and are not executed.

- [ ] **Step 16: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "refactor: real_data.ipynb imports problem and estimator definitions from journal2026"
```

---

### Task 5: Create experiments/timing_analysis.ipynb

**Files:**
- Create: `experiments/timing_analysis.ipynb`

This notebook is NOT added to `testpaths` in setup.cfg — it is not run in CI.
Experiment cells are tagged `skip-execution`.

- [ ] **Step 1: Create the notebook file**

Write `experiments/timing_analysis.ipynb` as a valid nbformat 4.5 notebook. The JSON structure must include `"nbformat": 4`, `"nbformat_minor": 5`, and cells with `"id"` fields. Use the following cell sources:

**Cell 1** (id: `ta-imports`, code, no tags):
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiments import Experiment, fitting_time, svd_time, normalization_time, number_of_features
from journal2026 import (
    JOURNAL2026_TRAIN_SIZES,
    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE,
    TIMING_ESTIMATORS, TIMING_EST_NAMES,
)
```

**Cell 2** (id: `ta-exp-regular`, code, tags: `["skip-execution"]`):
```python
exp_regular = Experiment(
    JOURNAL2026_D1_REGULAR, TIMING_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_REGULAR],
    seed=123,
    stats=[fitting_time, svd_time, normalization_time, number_of_features],
    est_names=TIMING_EST_NAMES).run()
```

**Cell 3** (id: `ta-exp-large`, code, tags: `["skip-execution"]`):
```python
exp_large = Experiment(
    JOURNAL2026_D1_LARGE, TIMING_ESTIMATORS,
    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_LARGE],
    seed=123,
    stats=[fitting_time, svd_time, normalization_time, number_of_features],
    est_names=TIMING_EST_NAMES).run()
```

**Cell 4** (id: `ta-sanity`, code, no tags):
```python
for exp, label in [(exp_regular, 'regular'), (exp_large, 'large')]:
    datasets = [p.dataset for p in exp.problems]
    for i, ds in enumerate(datasets):
        norm_vals = np.stack([exp.normalization_time_[:, i, 0, j]
                              for j in range(len(TIMING_EST_NAMES))])
        svd_vals  = np.stack([exp.svd_time_[:, i, 0, j]
                              for j in range(len(TIMING_EST_NAMES))])
        for name, vals in [('norm', norm_vals), ('svd', svd_vals)]:
            means = vals.mean(axis=1)
            overall_mean = means.mean()
            if overall_mean > 0 and means.std() / overall_mean > 0.10:
                print(f'WARNING [{label}] {ds}: {name} std/mean={means.std()/overall_mean:.2f}')
print('Sanity check complete')
```

**Cell 5** (id: `ta-table-fn`, code, no tags):
```python
def timing_table(exp):
    rows = []
    for i, prob in enumerate(exp.problems):
        t_em    = exp.fitting_time_[:, i, 0, 0].mean()
        t_cv101 = exp.fitting_time_[:, i, 0, 1].mean()
        t_cv11  = exp.fitting_time_[:, i, 0, 2].mean()
        t_svd   = np.stack([exp.svd_time_[:, i, 0, j]
                            for j in range(3)]).mean()
        t_norm  = np.stack([exp.normalization_time_[:, i, 0, j]
                            for j in range(3)]).mean()
        t_prep  = t_svd + t_norm
        rows.append({
            'dataset': prob.dataset,
            'T_EM':    round(t_em,    4),
            'T_CV101': round(t_cv101, 4),
            'T_CV11':  round(t_cv11,  4),
            'T_svd':   round(t_svd,   4),
            'T_norm':  round(t_norm,  4),
            'T_prep':  round(t_prep,  4),
            'SU':      round(t_cv101 / t_em, 2),
            'SU_post': round((t_cv101 - t_prep) / (t_em - t_prep), 2)
                       if (t_em - t_prep) > 0 else float('nan'),
        })
    return pd.DataFrame(rows).set_index('dataset')
```

**Cell 6** (id: `ta-table-regular`, code, no tags):
```python
timing_table(exp_regular)
```

**Cell 7** (id: `ta-table-large`, code, no tags):
```python
timing_table(exp_large)
```

**Cell 8** (id: `ta-chart`, code, no tags):
```python
colors_norm = ['#d9534f', '#e87e77', '#f5b8b5']
colors_svd  = ['#5bc0de', '#7ccfe8', '#b3e4f5']
colors_fit  = ['#5cb85c', '#7fca7f', '#b3e4b3']
width = 0.25
offsets = [-width, 0, width]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, exp, title in [(axes[0], exp_regular, 'Regular datasets'),
                        (axes[1], exp_large,   'Large datasets')]:
    datasets = [p.dataset for p in exp.problems]
    n = len(datasets)
    x = np.arange(n)
    for j, name in enumerate(TIMING_EST_NAMES):
        t_norm_mean = exp.normalization_time_[:, :, 0, j].mean(axis=0)
        t_svd_mean  = exp.svd_time_[:, :, 0, j].mean(axis=0)
        t_fit_mean  = (exp.fitting_time_[:, :, 0, j].mean(axis=0)
                       - t_svd_mean - t_norm_mean)
        t_total     = exp.fitting_time_[:, :, 0, j]
        ci_lo = np.percentile(t_total, 2.5, axis=0)
        ci_hi = np.percentile(t_total, 97.5, axis=0)
        t_mean = t_total.mean(axis=0)
        xpos = x + offsets[j]
        ax.bar(xpos, t_norm_mean, width=width, color=colors_norm[j], label=f'{name}: T_norm')
        ax.bar(xpos, t_svd_mean,  width=width, color=colors_svd[j],
               bottom=t_norm_mean, label=f'{name}: T_svd')
        ax.bar(xpos, t_fit_mean,  width=width, color=colors_fit[j],
               bottom=t_norm_mean + t_svd_mean, label=f'{name}: T_fit')
        ax.errorbar(xpos, t_mean,
                    yerr=[t_mean - ci_lo, ci_hi - t_mean],
                    fmt='none', color='black', capsize=3, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_ylabel('Time [s]')
    ax.set_title(title)

axes[0].legend(loc='upper right', fontsize=7)
plt.tight_layout()
```

To create this notebook, write the complete JSON to `experiments/timing_analysis.ipynb`:

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {
   "cell_type": "code",
   "id": "ta-imports",
   "metadata": {},
   "source": ["import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom experiments import Experiment, fitting_time, svd_time, normalization_time, number_of_features\nfrom journal2026 import (\n    JOURNAL2026_TRAIN_SIZES,\n    JOURNAL2026_D1_REGULAR, JOURNAL2026_D1_LARGE,\n    TIMING_ESTIMATORS, TIMING_EST_NAMES,\n)"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-exp-regular",
   "metadata": {"tags": ["skip-execution"]},
   "source": ["exp_regular = Experiment(\n    JOURNAL2026_D1_REGULAR, TIMING_ESTIMATORS,\n    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_REGULAR],\n    seed=123,\n    stats=[fitting_time, svd_time, normalization_time, number_of_features],\n    est_names=TIMING_EST_NAMES).run()"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-exp-large",
   "metadata": {"tags": ["skip-execution"]},
   "source": ["exp_large = Experiment(\n    JOURNAL2026_D1_LARGE, TIMING_ESTIMATORS,\n    reps=30, ns=[[JOURNAL2026_TRAIN_SIZES[p.dataset]] for p in JOURNAL2026_D1_LARGE],\n    seed=123,\n    stats=[fitting_time, svd_time, normalization_time, number_of_features],\n    est_names=TIMING_EST_NAMES).run()"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-sanity",
   "metadata": {},
   "source": ["for exp, label in [(exp_regular, 'regular'), (exp_large, 'large')]:\n    datasets = [p.dataset for p in exp.problems]\n    for i, ds in enumerate(datasets):\n        norm_vals = np.stack([exp.normalization_time_[:, i, 0, j]\n                              for j in range(len(TIMING_EST_NAMES))])\n        svd_vals  = np.stack([exp.svd_time_[:, i, 0, j]\n                              for j in range(len(TIMING_EST_NAMES))])\n        for name, vals in [('norm', norm_vals), ('svd', svd_vals)]:\n            means = vals.mean(axis=1)\n            overall_mean = means.mean()\n            if overall_mean > 0 and means.std() / overall_mean > 0.10:\n                print(f'WARNING [{label}] {ds}: {name} std/mean={means.std()/overall_mean:.2f}')\nprint('Sanity check complete')"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-table-fn",
   "metadata": {},
   "source": ["def timing_table(exp):\n    rows = []\n    for i, prob in enumerate(exp.problems):\n        t_em    = exp.fitting_time_[:, i, 0, 0].mean()\n        t_cv101 = exp.fitting_time_[:, i, 0, 1].mean()\n        t_cv11  = exp.fitting_time_[:, i, 0, 2].mean()\n        t_svd   = np.stack([exp.svd_time_[:, i, 0, j]\n                            for j in range(3)]).mean()\n        t_norm  = np.stack([exp.normalization_time_[:, i, 0, j]\n                            for j in range(3)]).mean()\n        t_prep  = t_svd + t_norm\n        rows.append({\n            'dataset': prob.dataset,\n            'T_EM':    round(t_em,    4),\n            'T_CV101': round(t_cv101, 4),\n            'T_CV11':  round(t_cv11,  4),\n            'T_svd':   round(t_svd,   4),\n            'T_norm':  round(t_norm,  4),\n            'T_prep':  round(t_prep,  4),\n            'SU':      round(t_cv101 / t_em, 2),\n            'SU_post': round((t_cv101 - t_prep) / (t_em - t_prep), 2)\n                       if (t_em - t_prep) > 0 else float('nan'),\n        })\n    return pd.DataFrame(rows).set_index('dataset')"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-table-regular",
   "metadata": {},
   "source": ["timing_table(exp_regular)"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-table-large",
   "metadata": {},
   "source": ["timing_table(exp_large)"],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "id": "ta-chart",
   "metadata": {},
   "source": ["colors_norm = ['#d9534f', '#e87e77', '#f5b8b5']\ncolors_svd  = ['#5bc0de', '#7ccfe8', '#b3e4f5']\ncolors_fit  = ['#5cb85c', '#7fca7f', '#b3e4b3']\nwidth = 0.25\noffsets = [-width, 0, width]\n\nfig, axes = plt.subplots(1, 2, figsize=(14, 5))\nfor ax, exp, title in [(axes[0], exp_regular, 'Regular datasets'),\n                        (axes[1], exp_large,   'Large datasets')]:\n    datasets = [p.dataset for p in exp.problems]\n    n = len(datasets)\n    x = np.arange(n)\n    for j, name in enumerate(TIMING_EST_NAMES):\n        t_norm_mean = exp.normalization_time_[:, :, 0, j].mean(axis=0)\n        t_svd_mean  = exp.svd_time_[:, :, 0, j].mean(axis=0)\n        t_fit_mean  = (exp.fitting_time_[:, :, 0, j].mean(axis=0)\n                       - t_svd_mean - t_norm_mean)\n        t_total     = exp.fitting_time_[:, :, 0, j]\n        ci_lo = np.percentile(t_total, 2.5, axis=0)\n        ci_hi = np.percentile(t_total, 97.5, axis=0)\n        t_mean = t_total.mean(axis=0)\n        xpos = x + offsets[j]\n        ax.bar(xpos, t_norm_mean, width=width, color=colors_norm[j], label=f'{name}: T_norm')\n        ax.bar(xpos, t_svd_mean,  width=width, color=colors_svd[j],\n               bottom=t_norm_mean, label=f'{name}: T_svd')\n        ax.bar(xpos, t_fit_mean,  width=width, color=colors_fit[j],\n               bottom=t_norm_mean + t_svd_mean, label=f'{name}: T_fit')\n        ax.errorbar(xpos, t_mean,\n                    yerr=[t_mean - ci_lo, ci_hi - t_mean],\n                    fmt='none', color='black', capsize=3, linewidth=0.8)\n    ax.set_xticks(x)\n    ax.set_xticklabels(datasets, rotation=45, ha='right')\n    ax.set_ylabel('Time [s]')\n    ax.set_title(title)\n\naxes[0].legend(loc='upper right', fontsize=7)\nplt.tight_layout()"],
   "outputs": [],
   "execution_count": null
  }
 ]
}
```

- [ ] **Step 2: Run full test suite (timing_analysis.ipynb is NOT in testpaths)**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: All pass. Verify timing_analysis.ipynb does NOT appear in test output.

- [ ] **Step 3: Commit**

```bash
git add experiments/timing_analysis.ipynb
git commit -m "feat: add timing_analysis.ipynb with sanity check, timing tables, and stacked bar charts"
```
