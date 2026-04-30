# NeurIPS Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `neurips2023.py` the self-contained home for all NeurIPS 2023 experiment specifics; add `overwrite_cache` to the experiment runners; clean up `experiments.py` by deleting the legacy `EmpiricalDataExperiment`.

**Architecture:** Constants and `ExperimentWithPerSeriesSeeding` migrate from `experiments.py` / `problems.py` into `neurips2023.py`; private helpers in `experiments.py` are promoted to public; cache control gains an `overwrite_cache` flag that deletes per-combination entries before recomputing; a CLI runner is added to `neurips2023.py`.

**Tech Stack:** Python standard library (`argparse`, `shutil`), `numpy`, `sklearn`, `fastprogress`, `joblib`, `pytest` with `tmp_path` + `monkeypatch` fixtures, `NotebookEdit` for `.ipynb` changes.

---

## File Map

| File | Change |
|---|---|
| `experiments/problems.py` | Rename `_OHE` → `onehot_non_numeric` (line 380); delete lines 382–497 |
| `experiments/neurips2023.py` | Add imports; add NEURIPS2023 constants; add estimator constants; move `ExperimentWithPerSeriesSeeding` here; add `overwrite_cache` to its `run()`; add `__main__` runner |
| `experiments/experiments.py` | Rename `_RUN_FILE_STATE`, `_cache_key`, `_make_run_id` → public; add `import shutil`; add `overwrite_cache` to `Experiment.run()`; delete `ExperimentWithPerSeriesSeeding`; delete `EmpiricalDataExperiment` |
| `tests/test_experiments.py` | Update imports; update `test_series_exp_*` monkeypatches; delete/rewrite `_simple_exp` tests; add `overwrite_cache` tests |
| `experiments/real_data_neurips2023.ipynb` | Insert estimator cell; update all imports |
| `experiments/real_data.ipynb` | Update `NEURIPS2023_TRAIN_SIZES` import; replace local `_OHE` with `onehot_non_numeric` |

---

### Task 1: Rename `_OHE` → `onehot_non_numeric` in `problems.py` and `real_data.ipynb`

**Files:**
- Modify: `experiments/problems.py:380`
- Modify: `experiments/real_data.ipynb` (setup cell)

- [ ] **Step 1: Rename in `problems.py`**

In `experiments/problems.py`, change line 380 from:
```python
_OHE = (OneHotEncodeCategories(),)
```
to:
```python
onehot_non_numeric = (OneHotEncodeCategories(),)
```

- [ ] **Step 2: Verify `problems.py` still imports cleanly**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "from problems import onehot_non_numeric; print(onehot_non_numeric)"
```
Expected: `(<experiments.problems.OneHotEncodeCategories object at 0x...>,)`

- [ ] **Step 3: Read `real_data.ipynb` setup cell**

Use the `Read` tool on `experiments/real_data.ipynb` to find the cell that contains `_OHE = (OneHotEncodeCategories(),)` and note its `cell_id`. (Do not have the notebook open in VSCode.)

- [ ] **Step 4: Update `real_data.ipynb` setup cell**

The cell currently contains (among other things):
```python
from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion, NEURIPS2023_TRAIN_SIZES
_OHE = (OneHotEncodeCategories(),)
```

Replace with:
```python
from problems import EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric
from neurips2023 import NEURIPS2023_TRAIN_SIZES
```

Then replace every occurrence of `_OHE` in the notebook with `onehot_non_numeric` (there will be multiple occurrences in problem-list cells).

- [ ] **Step 5: Run tests**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/problems.py experiments/real_data.ipynb
git commit -m "refactor: rename _OHE to onehot_non_numeric; update real_data.ipynb"
```

---

### Task 2: Add NeurIPS 2023 constants and estimator constants to `neurips2023.py`; remove from `problems.py`

**Files:**
- Modify: `experiments/neurips2023.py`
- Modify: `experiments/problems.py` (delete lines 382–497)

- [ ] **Step 1: Add imports to `neurips2023.py`**

At the top of `experiments/neurips2023.py`, after the existing imports, add:
```python
import dataclasses
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from problems import (EmpiricalDataProblem, PolynomialExpansion, onehot_non_numeric)
from data import DATASETS
```

(Keep the existing `import time`, `from sklearn.base import clone`, `from fastprogress.fastprogress import progress_bar`, `from experiments import default_stats`.)

- [ ] **Step 2: Add NEURIPS2023 constants to `neurips2023.py`**

After the imports, add the following block (verbatim copy from `problems.py` lines 382–497, with `_OHE` already replaced by `onehot_non_numeric` from Task 1):

```python
NEURIPS2023 = frozenset({
    EmpiricalDataProblem('abalone',          'Rings',
                         x_transforms=onehot_non_numeric, zero_variance_filter=True),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                         zero_variance_filter=True),
    EmpiricalDataProblem('automobile',       'price',
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric,
                         y_transforms=(np.log,),
                         zero_variance_filter=True),
    EmpiricalDataProblem('autompg',          'mpg',
                         drop=('car_name',), nan_policy='drop_rows',
                         zero_variance_filter=True),
    EmpiricalDataProblem('blog',             'V281',
                         zero_variance_filter=True),
    EmpiricalDataProblem('boston',           'medv',
                         zero_variance_filter=True),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                         zero_variance_filter=True),
    EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                         drop=('state', 'fold', 'communityname'),
                         nan_policy='drop_cols',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ct_slices',        'reference',
                         zero_variance_filter=True),
    EmpiricalDataProblem('diabetes',         'target',
                         zero_variance_filter=True),
    EmpiricalDataProblem('eye',              'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('facebook',         'Total Interactions',
                         drop=('comment', 'like', 'share'),
                         nan_policy='drop_rows',
                         x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('forest',           'area',
                         x_transforms=onehot_non_numeric,
                         y_transforms=(np.log1p,),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                         drop=('GT_turbine_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                         drop=('GT_compressor_decay',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                         drop=('total_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                         drop=('motor_UPDRS',),
                         zero_variance_filter=True),
    EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                         zero_variance_filter=True),
    EmpiricalDataProblem('ribo',             'y',
                         zero_variance_filter=True),
    EmpiricalDataProblem('student',          'G3',
                         drop=('G1', 'G2'),
                         x_transforms=onehot_non_numeric,
                         zero_variance_filter=True),
    EmpiricalDataProblem('tomshw',           'V97',
                         zero_variance_filter=True),
    EmpiricalDataProblem('twitter',          'V78',
                         zero_variance_filter=True),
    EmpiricalDataProblem('yacht',            'Residuary_resistance',
                         y_transforms=(np.log,),
                         zero_variance_filter=True),
})

NEURIPS2023_D2 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(2),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 1000
)

NEURIPS2023_D3 = frozenset(
    dataclasses.replace(p, x_transforms=p.x_transforms + (PolynomialExpansion(3),))
    for p in NEURIPS2023
    if 'p' in DATASETS[p.dataset]
    and 'n' in DATASETS[p.dataset]
    and DATASETS[p.dataset]['p'] < 150
    and DATASETS[p.dataset]['n'] < 20000
)

# Training set sizes (n_train = floor(0.7 * n_actual)) derived from actual
# post-preprocessing row counts. Keyed by dataset name and shared across
# NEURIPS2023, NEURIPS2023_D2, and NEURIPS2023_D3 (polynomial expansion does
# not change row count; no dataset in these sets has differing n across targets).
#
# Deviations from floor(0.7 * DATASETS[dataset]['n']):
#   automobile: actual n=159 after drop_rows (registry n=205) -> n_train=111 vs 143
#   autompg:    actual n=392 after drop_rows (registry n=398) -> n_train=274 vs 278
#   facebook:   actual n=499 after drop_rows (registry n=500) -> n_train=349 vs 350
NEURIPS2023_TRAIN_SIZES = {
    'abalone':          2923,
    'airfoil':          1052,
    'automobile':        111,
    'autompg':           274,
    'blog':            36677,
    'boston':            354,
    'concrete':          721,
    'crime':            1395,
    'ct_slices':       37450,
    'diabetes':          309,
    'eye':                84,
    'facebook':          349,
    'forest':            361,
    'naval_propulsion': 8353,
    'parkinsons':       4112,
    'real_estate':       289,
    'ribo':               49,
    'student':           454,
    'tomshw':          19725,
    'twitter':        408275,
    'yacht':             215,
}
```

- [ ] **Step 3: Add estimator constants to `neurips2023.py`**

Immediately after `NEURIPS2023_TRAIN_SIZES`, add:

```python
NEURIPS2023_ESTIMATORS = [
    RidgeEM(t2=False),
    RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
    RidgeLOOCV(alphas=100),
]

NEURIPS2023_EST_NAMES = ['EM', 'CV_fix', 'CV_glm']
```

- [ ] **Step 4: Delete the moved constants from `problems.py`**

In `experiments/problems.py`, delete lines 382–497 (everything from `NEURIPS2023 = frozenset({` through the end of `NEURIPS2023_TRAIN_SIZES`). Also remove any imports that are no longer needed by `problems.py` itself (check: `dataclasses` is used only for `NEURIPS2023_D2`/`D3` — if nothing else uses it, delete that import too).

- [ ] **Step 5: Verify module loads correctly**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "
from neurips2023 import (NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3,
                          NEURIPS2023_TRAIN_SIZES, NEURIPS2023_ESTIMATORS,
                          NEURIPS2023_EST_NAMES)
print(len(NEURIPS2023), len(NEURIPS2023_D2), len(NEURIPS2023_D3))
print(NEURIPS2023_EST_NAMES)
"
```
Expected: `23 <some number> <some number>` and `['EM', 'CV_fix', 'CV_glm']`.

- [ ] **Step 6: Run tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add experiments/neurips2023.py experiments/problems.py
git commit -m "refactor: move NEURIPS2023 constants and estimators to neurips2023.py"
```

---

### Task 3: Promote private helpers in `experiments.py` to public

**Files:**
- Modify: `experiments/experiments.py`
- Modify: `tests/test_experiments.py:88`

The three names `_RUN_FILE_STATE`, `_cache_key`, `_make_run_id` are needed outside `experiments.py` after `ExperimentWithPerSeriesSeeding` moves. Promote them now so later tasks import clean names.

- [ ] **Step 1: Rename in `experiments.py`**

Apply three renames throughout `experiments/experiments.py`:
- `_RUN_FILE_STATE` → `RUN_FILE_STATE` (definition at line 22; usages in `Experiment.run()` and `ExperimentWithPerSeriesSeeding.run()`)
- `_cache_key` → `cache_key` (definition at line 28; all call sites; update the doctest inside the docstring too: `>>> _cache_key(` → `>>> cache_key(`)
- `_make_run_id` → `make_run_id` (definition at line 47; all call sites)

- [ ] **Step 2: Update `test_experiments.py` import**

Change line 88 in `tests/test_experiments.py`:
```python
from experiments import _make_run_id
```
to:
```python
from experiments import make_run_id
```

And update the single usage on the next line: `run_id = _make_run_id('Experiment')` → `run_id = make_run_id('Experiment')`.

- [ ] **Step 3: Run tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "refactor: promote _cache_key, _make_run_id, _RUN_FILE_STATE to public"
```

---

### Task 4: Move `ExperimentWithPerSeriesSeeding` to `neurips2023.py`; create `test_neurips2023.py`

**Files:**
- Modify: `experiments/neurips2023.py`
- Modify: `experiments/experiments.py` (delete the class)
- Create: `tests/test_neurips2023.py`
- Modify: `tests/test_experiments.py` (remove series tests and imports)

`ExperimentWithPerSeriesSeeding` uses `CACHE_DIR`, `cache_key`, `make_run_id`, `RUN_FILE_STATE`, `empirical_default_stats` from `experiments.py` and `save_json`, `load_json`, `to_json`, `environment` from `util`. Its tests move to `test_neurips2023.py` alongside the class — tests follow the module they test. The monkeypatch target changes to `neurips2023.CACHE_DIR` because `CACHE_DIR` is looked up in the module where it is defined at call time.

- [ ] **Step 1: Add required imports to `neurips2023.py`**

Add to the imports at the top of `experiments/neurips2023.py`:
```python
import os
import time
import warnings
import datetime
import shutil
import numpy as np
from sklearn.base import clone
from experiments import (CACHE_DIR, cache_key, make_run_id, RUN_FILE_STATE,
                          empirical_default_stats)
from util import save_json, load_json, to_json, environment
```

(Some of `import time`, `import numpy as np`, `from sklearn.base import clone` are already present — add only what is missing. The `import shutil` added here is used in Task 6; adding it now avoids a second import-block edit.)

- [ ] **Step 2: Cut `ExperimentWithPerSeriesSeeding` from `experiments.py`**

Delete the entire class `ExperimentWithPerSeriesSeeding` (lines 484–624) from `experiments/experiments.py`.

- [ ] **Step 3: Paste `ExperimentWithPerSeriesSeeding` into `neurips2023.py`**

Add the class after the `NEURIPS2023_EST_NAMES` constant. The class body is identical except:
- Replace every `_cache_key(` with `cache_key(`
- Replace every `_make_run_id(` with `make_run_id(`
- Replace every `_RUN_FILE_STATE` with `RUN_FILE_STATE`

The full class (after substitution):

```python
class ExperimentWithPerSeriesSeeding:
    """Run repeated train/test experiments with per-series MT19937 seeding.

    Numerically identical to EmpiricalDataExperiment(generator='MT19937',
    seed_scope='series', seed_progression='fixed') at the same seed. Results
    are cached per (problem, n_train, estimator, reps, seed).

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    reps : int
    ns : array-like of shape (n_problems, n_sizes)
        A 1-D input is broadcast to all problems.
    seed : int or None
    stats : list of Metric or None
    est_names : list of str or None
    verbose : bool, default True
    """

    def __init__(self, problems, estimators, reps, ns,
                 seed=None, stats=None, est_names=None, verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.reps = reps
        self.ns = np.atleast_2d(ns)
        if len(self.ns) != len(self.problems):
            self.ns = self.ns.repeat(len(self.problems), axis=0)
        self.seed = seed
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def _series_cache_dir(self, prob_idx, n_idx, est_idx):
        return os.path.join(
            CACHE_DIR, 'series',
            self.problem_keys_[prob_idx],
            str(int(self.ns[prob_idx][n_idx])),
            self.estimator_keys_[est_idx],
            str(self.reps),
            str(self.seed),
        )

    def _all_stats_in_series_cache(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        return all(
            load_json(os.path.join(d, str(stat) + '.json'),
                      default={'computations': [], 'retrievals': []})['computations']
            for stat in self.stats
        )

    def _retrieve_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            values = [np.asarray(c['value']) for c in data['computations']]
            mean_val = np.mean(values, axis=0)
            self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx] = mean_val
            data['retrievals'].append({'value': to_json(mean_val), 'run_id': self.run_id_})
            save_json(path, data, indent=None)
            msg = stat.warn_retrieval(data['computations'])
            if msg:
                warnings.warn(msg)
        if self.verbose:
            for _ in range(self.reps):
                print('.', end='', flush=True)

    def _run_series(self, prob_idx, n_idx, est_idx):
        problem = self.problems[prob_idx]
        n_train = int(self.ns[prob_idx][n_idx])
        rng = np.random.RandomState(self.seed)
        for rep_idx in range(self.reps):
            X_train, X_test, y_train, y_test = problem.get_X_y(n_train, rng=rng)
            _est = clone(self.estimators[est_idx], safe=False)
            try:
                t0 = time.time()
                _est.fit(X_train, y_train)
                _est.fitting_time_ = time.time() - t0
            except Exception as e:
                warnings.warn(f"rep {rep_idx} failed for '{self.est_names[est_idx]}'"
                              f" on '{problem.dataset}': {e}")
            else:
                for stat in self.stats:
                    val = stat(_est, problem, X_test, y_test)
                    self.__dict__[str(stat) + '_'][rep_idx, prob_idx, n_idx, est_idx] = val
            if self.verbose:
                print('.', end='', flush=True)

    def _write_series(self, prob_idx, n_idx, est_idx):
        d = self._series_cache_dir(prob_idx, n_idx, est_idx)
        for stat in self.stats:
            path = os.path.join(d, str(stat) + '.json')
            data = load_json(path, default={'computations': [], 'retrievals': []})
            new_values = self.__dict__[str(stat) + '_'][:, prob_idx, n_idx, est_idx]
            if data['computations']:
                msg = stat.warn_recompute(
                    [c['value'] for c in data['computations']], new_values)
                if msg:
                    warnings.warn(msg)
            data['computations'].append({'value': to_json(new_values), 'run_id': self.run_id_})
            save_json(path, data, indent=None)

    def run(self, force_recompute=False, ignore_cache=False):
        n_problems = len(self.problems)
        n_sizes = len(self.ns[0])
        n_estimators = len(self.estimators)
        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.reps, n_problems, n_sizes, n_estimators), np.nan)
        self.run_id_ = make_run_id(type(self).__name__)
        self.environment_ = environment()
        self.timestamp_start_ = datetime.datetime.now().isoformat()
        self.trials_computed_ = self.trials_retrieved_ = 0
        self.estimator_keys_ = [cache_key(est) for est in self.estimators]
        self.problem_keys_ = [cache_key(prob) for prob in self.problems]
        run_path = os.path.join(CACHE_DIR, 'runs', f'{self.run_id_}.json')
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        for prob_idx in range(n_problems):
            if self.verbose:
                print(self.problems[prob_idx].dataset, end=' ')
            for n_idx in range(n_sizes):
                for est_idx in range(n_estimators):
                    if (not force_recompute and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        self.trials_retrieved_ += self.reps
                    else:
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        self.trials_computed_ += self.reps
            if self.verbose:
                print()
        self.timestamp_end_ = datetime.datetime.now().isoformat()
        if not ignore_cache:
            save_json(run_path, to_json(self, include_computed=RUN_FILE_STATE))
        return self
```

- [ ] **Step 4: Update `test_experiments.py` imports and monkeypatches**

Change the import block at the top of `tests/test_experiments.py`:
```python
# old
from experiments import (EmpiricalDataExperiment, Experiment, ExperimentWithPerSeriesSeeding, Metric, ...)
# new — remove EmpiricalDataExperiment and ExperimentWithPerSeriesSeeding from this line
from experiments import (EmpiricalDataExperiment, Experiment, Metric,
                         parameter_mean_squared_error, prediction_mean_squared_error,
                         regularization_parameter, number_of_iterations, variance_abs_error,
                         fitting_time, prediction_r2, number_of_features)
import neurips2023
from neurips2023 import ExperimentWithPerSeriesSeeding
```

Change the two `_simple_series_exp` test functions that use `monkeypatch`:
```python
# test_series_exp_cache_hit — change:
monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
# to:
monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))

# test_series_exp_ignore_cache — same substitution
```

- [ ] **Step 5: Run tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/neurips2023.py experiments/experiments.py tests/test_experiments.py
git commit -m "refactor: move ExperimentWithPerSeriesSeeding to neurips2023.py"
```

---

### Task 5: Delete `EmpiricalDataExperiment`; rewrite affected tests

**Files:**
- Modify: `experiments/experiments.py` (delete class)
- Modify: `tests/test_experiments.py`

- [ ] **Step 1: Write the replacement tests before deleting anything**

In `tests/test_experiments.py`, **replace** the following functions:

**Delete** (tests of EmpiricalDataExperiment-specific internals with no replacement needed — the behaviour they verify is not part of the public API of any remaining class):
- `_simple_exp` fixture
- `test_result_shape` (covered by `test_new_experiment_result_shape`)
- `test_make_rng_fixed_progression_same_seed`
- `test_make_rng_sequential_progression_different_seeds`
- `test_pcg64_and_mt19937_differ`

**Rewrite** `test_ns_shape` to use `Experiment`:
```python
def test_ns_shape():
    prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
    exp = Experiment([prob], [RidgeEM()], reps=2, ns=n_train_from_proportion([prob]))
    assert exp.ns.shape == (1, 1)
    assert int(exp.ns[0, 0]) == 309
```

**Rewrite** `test_series_scope_reproducible` to use `ExperimentWithPerSeriesSeeding`:
```python
def test_series_exp_reproducible():
    exp1 = _simple_series_exp().run(ignore_cache=True)
    exp2 = _simple_series_exp().run(ignore_cache=True)
    np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)
```

**Replace** `test_series_exp_numerical_equivalence` with the above (delete the old function — it referenced `EmpiricalDataExperiment` as a reference implementation; reproducibility is the property we care about going forward).

- [ ] **Step 2: Remove `EmpiricalDataExperiment` from the test import**

```python
# old
from experiments import (EmpiricalDataExperiment, Experiment, Metric, ...)
# new
from experiments import (Experiment, Metric, ...)
```

- [ ] **Step 3: Run tests to verify they pass with EmpiricalDataExperiment still present**

```bash
pytest tests/test_experiments.py
```
Expected: all pass (EmpiricalDataExperiment still exists; we just aren't testing it anymore).

- [ ] **Step 4: Delete `EmpiricalDataExperiment` from `experiments.py`**

Remove the entire `EmpiricalDataExperiment` class definition (from `class EmpiricalDataExperiment:` through the end of the class body — roughly lines 671 to end of file, or wherever it ends). Check that `experiments.py` still ends with valid Python after the deletion.

- [ ] **Step 5: Run tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add experiments/experiments.py tests/test_experiments.py
git commit -m "refactor: delete EmpiricalDataExperiment; rewrite its tests"
```

---

### Task 6: Add `overwrite_cache` to `Experiment` and `ExperimentWithPerSeriesSeeding`

**Files:**
- Modify: `experiments/experiments.py`
- Modify: `experiments/neurips2023.py`
- Modify: `tests/test_experiments.py`

`overwrite_cache=True` deletes the cache entry for each (problem, n_train, estimator) combination immediately before computing it — not upfront. `ignore_cache` overrides all other flags.

- [ ] **Step 1: Add `import shutil` to `experiments.py`**

Add `import shutil` to the imports block at the top of `experiments/experiments.py`. (`neurips2023.py` already has `import shutil` from Task 4.)

- [ ] **Step 2: Write the failing `overwrite_cache` test for `Experiment`**

Add to `tests/test_experiments.py`:
```python
def test_new_experiment_overwrite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, 'CACHE_DIR', str(tmp_path))
    _simple_new_exp().run()
    _simple_new_exp().run(force_recompute=True)  # accumulates 2 computations
    _simple_new_exp().run(overwrite_cache=True)  # deletes per combo, rewrites: back to 1
    trial_dir = os.path.join(str(tmp_path), 'trial')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(trial_dir)
                  for f in fs if f.endswith('.json')]
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 1
```

- [ ] **Step 3: Run it to verify it fails**

```bash
pytest tests/test_experiments.py::test_new_experiment_overwrite_cache -v
```
Expected: FAIL (`unexpected keyword argument 'overwrite_cache'` or similar).

- [ ] **Step 4: Implement `overwrite_cache` in `Experiment.run()`**

In `experiments/experiments.py`, change the signature of `Experiment.run()`:
```python
def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
```

In the inner loop, change:
```python
                        if (not force_recompute and not ignore_cache
                                and self._all_stats_in_trial_cache(
                                    prob_idx, n_idx, est_idx, rep_idx)):
                            self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_retrieved_ += 1
                        else:
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_computed_ += 1
```
to:
```python
                        if (not force_recompute and not overwrite_cache and not ignore_cache
                                and self._all_stats_in_trial_cache(
                                    prob_idx, n_idx, est_idx, rep_idx)):
                            self._retrieve_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_retrieved_ += 1
                        else:
                            if overwrite_cache and not ignore_cache:
                                shutil.rmtree(
                                    self._trial_cache_dir(prob_idx, n_idx, est_idx, rep_idx),
                                    ignore_errors=True)
                            self._run_trial(prob_idx, n_idx, est_idx, rep_idx)
                            if not ignore_cache:
                                self._write_trial(prob_idx, n_idx, est_idx, rep_idx)
                            self.trials_computed_ += 1
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_experiments.py::test_new_experiment_overwrite_cache -v
```
Expected: PASS.

- [ ] **Step 6: Write the failing `overwrite_cache` test for `ExperimentWithPerSeriesSeeding`**

Add to `tests/test_experiments.py`:
```python
def test_series_exp_overwrite_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(neurips2023, 'CACHE_DIR', str(tmp_path))
    _simple_series_exp().run()
    _simple_series_exp().run(force_recompute=True)  # accumulates 2 computations
    _simple_series_exp().run(overwrite_cache=True)  # deletes per combo, rewrites: back to 1
    series_dir = os.path.join(str(tmp_path), 'series')
    json_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(series_dir)
                  for f in fs if f.endswith('.json')]
    with open(json_files[0]) as f:
        data = json.load(f)
    assert len(data['computations']) == 1
```

- [ ] **Step 7: Run it to verify it fails**

```bash
pytest tests/test_experiments.py::test_series_exp_overwrite_cache -v
```
Expected: FAIL.

- [ ] **Step 8: Implement `overwrite_cache` in `ExperimentWithPerSeriesSeeding.run()`**

In `ExperimentWithPerSeriesSeeding.run()`, change signature:
```python
def run(self, force_recompute=False, ignore_cache=False, overwrite_cache=False):
```

In the inner loop, change:
```python
                    if (not force_recompute and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        self.trials_retrieved_ += self.reps
                    else:
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        self.trials_computed_ += self.reps
```
to:
```python
                    if (not force_recompute and not overwrite_cache and not ignore_cache
                            and self._all_stats_in_series_cache(prob_idx, n_idx, est_idx)):
                        self._retrieve_series(prob_idx, n_idx, est_idx)
                        self.trials_retrieved_ += self.reps
                    else:
                        if overwrite_cache and not ignore_cache:
                            shutil.rmtree(
                                self._series_cache_dir(prob_idx, n_idx, est_idx),
                                ignore_errors=True)
                        self._run_series(prob_idx, n_idx, est_idx)
                        if not ignore_cache:
                            self._write_series(prob_idx, n_idx, est_idx)
                        self.trials_computed_ += self.reps
```

- [ ] **Step 9: Run all tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add experiments/experiments.py experiments/neurips2023.py tests/test_experiments.py
git commit -m "feat: add overwrite_cache option to Experiment and ExperimentWithPerSeriesSeeding"
```

---

### Task 7: Add `__main__` runner with CLI to `neurips2023.py`

**Files:**
- Modify: `experiments/neurips2023.py`

- [ ] **Step 1: Add `import argparse` and `import sys` to `neurips2023.py`**

Add to the imports block.

- [ ] **Step 2: Add the runner**

At the end of `experiments/neurips2023.py`, add:

```python
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Re-run all NeurIPS 2023 empirical experiments.')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--ignore_cache',    action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    args = parser.parse_args()

    for problems in [NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3]:
        problems_sorted = sorted(problems, key=lambda p: DATASETS[p.dataset]['n'])
        ExperimentWithPerSeriesSeeding(
            problems=problems_sorted,
            estimators=NEURIPS2023_ESTIMATORS,
            reps=100,
            ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_sorted],
            seed=123,
            est_names=NEURIPS2023_EST_NAMES,
        ).run(
            force_recompute=args.force_recompute,
            ignore_cache=args.ignore_cache,
            overwrite_cache=args.overwrite_cache,
        )
```

- [ ] **Step 3: Verify the CLI shows help without crashing**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge/experiments && source ../.venv/bin/activate && python neurips2023.py --help
```
Expected: prints usage with `--force_recompute`, `--ignore_cache`, `--overwrite_cache` options and exits 0.

- [ ] **Step 4: Run tests**

```bash
pytest
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/neurips2023.py
git commit -m "feat: add __main__ CLI runner to neurips2023.py"
```

---

### Task 8: Update `real_data_neurips2023.ipynb`

**Files:**
- Modify: `experiments/real_data_neurips2023.ipynb`

Do not open this notebook in VSCode during editing.

- [ ] **Step 1: Read the notebook**

Use the `Read` tool on `experiments/real_data_neurips2023.ipynb`. Note cell IDs for:
- `a001` — intro markdown
- `a002` — "## No Interaction Variables" markdown
- `a003` — d1 experiment cell (imports, estimators inline, `exp_d1`)
- `a006` — d2 experiment cell
- `a009` — d3 experiment cell

- [ ] **Step 2: Insert estimator cell between `a001` and `a002`**

Announce: "Inserting new code cell after `a001` with estimator import and output display."

Use `NotebookEdit` to insert a new cell after `a001`:
```python
from neurips2023 import NEURIPS2023_ESTIMATORS, NEURIPS2023_EST_NAMES
NEURIPS2023_ESTIMATORS
```

- [ ] **Step 3: Update cell `a003`**

Announce: "Updating cell `a003` to remove inline estimator definitions and update imports."

The new content for `a003`:
```python
import numpy as np
from neurips2023 import (NEURIPS2023, NEURIPS2023_TRAIN_SIZES, ExperimentWithPerSeriesSeeding)
from data import DATASETS

problems_d1 = sorted(NEURIPS2023, key=lambda p: DATASETS[p.dataset]['n'])
exp_d1 = ExperimentWithPerSeriesSeeding(
    problems_d1, NEURIPS2023_ESTIMATORS,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d1],
    seed=123,
    est_names=NEURIPS2023_EST_NAMES, verbose=True).run()
print()
```

- [ ] **Step 4: Update cell `a006`**

Announce: "Updating cell `a006` to change import source and use module constants."

The new content for `a006`:
```python
from neurips2023 import NEURIPS2023_D2

problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
exp_d2 = ExperimentWithPerSeriesSeeding(
    problems_d2, NEURIPS2023_ESTIMATORS,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d2],
    seed=123,
    est_names=NEURIPS2023_EST_NAMES, verbose=True).run()
print()
```

- [ ] **Step 5: Update cell `a009`**

Announce: "Updating cell `a009` to change import source and use module constants."

The new content for `a009`:
```python
from neurips2023 import NEURIPS2023_D3

problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
exp_d3 = ExperimentWithPerSeriesSeeding(
    problems_d3, NEURIPS2023_ESTIMATORS,
    reps=100, ns=[[NEURIPS2023_TRAIN_SIZES[p.dataset]] for p in problems_d3],
    seed=123,
    est_names=NEURIPS2023_EST_NAMES, verbose=True).run()
print()
```

- [ ] **Step 6: Run tests**

```bash
pytest
```
Expected: all pass (notebook is tagged `skip-execution` for the experiment cells so it runs only structural checks in CI).

- [ ] **Step 7: Commit**

```bash
git add experiments/real_data_neurips2023.ipynb
git commit -m "refactor: update real_data_neurips2023.ipynb imports; add estimator display cell"
```

---

## Manual Testing Instructions (for the user)

These steps verify the `FittingTime` warning and `overwrite_cache` end-to-end using the live NeurIPS cache. Run from `experiments/`:

```bash
source ../.venv/bin/activate
```

**1. Accumulate a second computation record:**
```bash
python neurips2023.py --force_recompute
```
Cancel (Ctrl-C) after a few datasets complete. A `FittingTime` warning about "only one computation stored" will now appear on the next retrieval for those datasets.

**2. Verify the warning appears on retrieval:**
```bash
python neurips2023.py
```
You should see `UserWarning: FittingTime: only one computation stored` for the datasets completed above. Cancel midway again.

**3. Clean the cache with `--overwrite_cache`:**
```bash
python neurips2023.py --overwrite_cache
```
Cancel after the same datasets. Re-run once more without flags — the `FittingTime` warning should be gone for the overwritten entries (single clean computation).
