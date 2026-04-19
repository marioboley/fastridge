# Seeding Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `EmpiricalDataProblem.get_X_y` and `EmpiricalDataExperiment` to thread explicit generator objects through the data pipeline, enabling reproducible seeding at configurable scopes and a stable `n_train` value before the trial loop.

**Architecture:** `get_X_y(n_train, rng)` absorbs the train/test split and optional zero-variance filter; the experiment runner creates generator objects via `_make_rng` and passes them down. `get_X_y` is the single normalization boundary — it accepts None/int/Generator/RandomState and normalizes to Generator or RandomState before forwarding. x_transforms receive a required `rng` argument (always typed Generator or RandomState) so stochastic transforms (e.g. `PolynomialExpansion` subsampling) consume the same generator as the split. `EmpiricalDataExperiment` replaces `test_prop`/`n_iterations` with `ns`/`reps` and adds `generator`, `seed_scope`, `seed_progression` to control seeding granularity.

**Tech Stack:** numpy (`np.random.Generator`, `np.random.PCG64`, `np.random.RandomState`), sklearn `train_test_split`, pytest doctests + unit tests.

**Spec:** `docs/superpowers/specs/2026-04-18-empirical-experiment-seeding-refactor-design.md`

---

## Testing Philosophy

Existing doctests serve as usage documentation and are preserved. New behavioral tests
(reproducibility, filter correctness, seeding scope interactions) go in dedicated unit test
files rather than doctests — they don't improve API documentation and add noise to the
docstrings.

---

## File Map

| File | Changes |
|------|---------|
| `experiments/problems.py` | Add required `rng` parameter to `PolynomialExpansion.__call__` and `OneHotEncodeCategories.__call__`; add `zero_variance_filter` to `EmpiricalDataProblem`; refactor `get_X_y` with normalization at entry; add `n_train_from_proportion`; update NEURIPS2023 sets |
| `experiments/experiments.py` | Replace `EmpiricalDataExperiment` constructor and `run()`; add `_RNG_FACTORIES` and `_make_rng` |
| `experiments/real_data.ipynb` | Update imports and all `EmpiricalDataExperiment` call sites |
| `experiments/real_data_neurips2023.ipynb` | Update all `EmpiricalDataExperiment` call sites |
| `tests/test_problems.py` | New: behavioral tests for `get_X_y`, `zero_variance_filter`, rng threading |
| `tests/test_experiments.py` | New: behavioral tests for `_make_rng`, seeding scopes, result shapes |

---

## Task 1: Add `rng` parameter to x_transforms

**Files:**
- Modify: `experiments/problems.py`
- Create: `tests/test_problems.py`

- [ ] **Step 1: Create `tests/test_problems.py` with a failing test**

  ```python
  import sys
  import os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

  import numpy as np
  import pandas as pd
  import pytest
  from problems import PolynomialExpansion, OneHotEncodeCategories


  def test_polynomial_expansion_rng_deterministic():
      X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
      small = PolynomialExpansion(2, max_entries=9)
      rng1 = np.random.default_rng(0)
      rng2 = np.random.default_rng(0)
      assert list(small(X, rng=rng1).columns) == list(small(X, rng=rng2).columns)


  def test_one_hot_encode_accepts_rng():
      X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
      enc = OneHotEncodeCategories()
      result = enc(X, rng=np.random.default_rng(0))
      assert result.equals(X)
  ```

- [ ] **Step 2: Run to verify tests fail**

  ```bash
  cd /Users/marioboley/Documents/GitHub/fastridge && python -m pytest tests/test_problems.py::test_polynomial_expansion_rng_deterministic tests/test_problems.py::test_one_hot_encode_accepts_rng -v
  ```

  Expected: FAIL — `__call__() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Update `PolynomialExpansion.__call__` in `experiments/problems.py`**

  Replace the current `__call__` method:

  ```python
  def __call__(self, X, rng):
      poly = PolynomialFeatures(degree=self.degree, include_bias=False)
      X_poly = pd.DataFrame(
          poly.fit_transform(X),
          columns=poly.get_feature_names_out(X.columns),
          index=X.index
      )
      n, p = X_poly.shape
      if n * p > self.max_entries:
          linear_cols = list(X.columns)
          interaction_cols = [c for c in X_poly.columns if c not in linear_cols]
          p_budget = int(np.ceil(self.max_entries / n))
          pnew = max(0, min(len(interaction_cols), p_budget - len(linear_cols)))
          sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
          return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
      return X_poly
  ```

- [ ] **Step 4: Update `OneHotEncodeCategories.__call__` signature**

  Change only the signature line (body unchanged):

  ```python
  def __call__(self, X, rng):
  ```

- [ ] **Step 5: Run tests to verify both pass**

  ```bash
  python -m pytest tests/test_problems.py::test_polynomial_expansion_rng_deterministic tests/test_problems.py::test_one_hot_encode_accepts_rng -v
  ```

  Expected: PASS.

- [ ] **Step 6: Run existing doctests to confirm no regressions**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.PolynomialExpansion experiments/problems.py::problems.OneHotEncodeCategories -v
  ```

  Expected: PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add experiments/problems.py tests/test_problems.py
  git commit -m "feat: add rng parameter to PolynomialExpansion and OneHotEncodeCategories"
  ```

---

## Task 2: Add `zero_variance_filter` to `EmpiricalDataProblem`

**Files:**
- Modify: `experiments/problems.py`
- Modify: `tests/test_problems.py`

- [ ] **Step 1: Add a failing test to `tests/test_problems.py`**

  ```python
  from problems import EmpiricalDataProblem


  def test_zero_variance_filter_default_false():
      assert EmpiricalDataProblem('diabetes', 'target').zero_variance_filter is False


  def test_zero_variance_filter_in_repr_only_when_true():
      assert 'zero_variance_filter' not in repr(EmpiricalDataProblem('diabetes', 'target'))
      assert repr(EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)) == \
          "EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)"


  def test_zero_variance_filter_affects_equality():
      p_false = EmpiricalDataProblem('diabetes', 'target')
      p_true = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
      assert p_false == EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=False)
      assert p_false != p_true
  ```

- [ ] **Step 2: Run to verify they fail**

  ```bash
  python -m pytest tests/test_problems.py::test_zero_variance_filter_default_false tests/test_problems.py::test_zero_variance_filter_in_repr_only_when_true tests/test_problems.py::test_zero_variance_filter_affects_equality -v
  ```

  Expected: FAIL — `__init__() got an unexpected keyword argument 'zero_variance_filter'`.

- [ ] **Step 3: Add `zero_variance_filter` to `EmpiricalDataProblem.__init__`**

  Replace the `__init__` signature and `_repr` construction:

  ```python
  def __init__(self, dataset, target, drop=None, nan_policy=None,
               x_transforms=None, y_transforms=None, zero_variance_filter=False):
      self.dataset = dataset
      self.target = target
      self.drop = tuple(drop or [])
      self.nan_policy = nan_policy
      self.x_transforms = tuple(x_transforms or [])
      self.y_transforms = tuple(y_transforms or [])
      self.zero_variance_filter = zero_variance_filter
      self._repr = (
          f'EmpiricalDataProblem({self.dataset!r}, {self.target!r}'
          + (f', drop={list(self.drop)!r}' if self.drop else '')
          + (f', nan_policy={self.nan_policy!r}' if self.nan_policy else '')
          + (f', x_transforms={list(self.x_transforms)!r}' if self.x_transforms else '')
          + (f', y_transforms={list(self.y_transforms)!r}' if self.y_transforms else '')
          + (', zero_variance_filter=True' if self.zero_variance_filter else '')
          + ')'
      )
  ```

- [ ] **Step 4: Run to verify tests pass**

  ```bash
  python -m pytest tests/test_problems.py -k "zero_variance_filter" -v
  ```

  Expected: PASS.

- [ ] **Step 5: Run existing `EmpiricalDataProblem` doctests to confirm no regressions**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/problems.py tests/test_problems.py
  git commit -m "feat: add zero_variance_filter parameter to EmpiricalDataProblem"
  ```

---

## Task 3: Refactor `get_X_y`

**Files:**
- Modify: `experiments/problems.py`
- Modify: `tests/test_problems.py`

- [ ] **Step 1: Add `train_test_split` import to `experiments/problems.py`**

  Add alongside existing sklearn imports:

  ```python
  from sklearn.model_selection import train_test_split
  ```

- [ ] **Step 2: Add failing unit tests**

  ```python
  def test_get_X_y_returns_four_tuple_with_correct_sizes():
      prob = EmpiricalDataProblem('diabetes', 'target')
      X_train, X_test, y_train, y_test = prob.get_X_y(300)
      assert X_train.shape == (300, 10)
      assert X_test.shape == (142, 10)
      assert len(y_train) == 300
      assert len(y_test) == 142


  def test_get_X_y_seeded_reproducible():
      prob = EmpiricalDataProblem('diabetes', 'target')
      Xtr1, _, _, _ = prob.get_X_y(300, rng=np.random.default_rng(0))
      Xtr2, _, _, _ = prob.get_X_y(300, rng=np.random.default_rng(0))
      assert list(Xtr1.index) == list(Xtr2.index)


  def test_get_X_y_zero_variance_filter_applied():
      # crime dataset has zero-variance columns after splitting — use it to test filtering
      prob_no_filter = EmpiricalDataProblem(
          'crime', 'ViolentCrimesPerPop',
          drop=['state', 'fold', 'communityname'], nan_policy='drop_cols')
      prob_filter = EmpiricalDataProblem(
          'crime', 'ViolentCrimesPerPop',
          drop=['state', 'fold', 'communityname'], nan_policy='drop_cols',
          zero_variance_filter=True)
      rng = np.random.default_rng(1)
      Xtr_no, Xte_no, _, _ = prob_no_filter.get_X_y(1000, rng=np.random.default_rng(1))
      Xtr_f, Xte_f, _, _ = prob_filter.get_X_y(1000, rng=np.random.default_rng(1))
      # filter should not increase column count
      assert Xtr_f.shape[1] <= Xtr_no.shape[1]
      # train and test must have matching columns
      assert list(Xtr_f.columns) == list(Xte_f.columns)
  ```

  Note: if `crime` turns out not to have zero-variance columns at this split size, use a
  split where one of the numeric columns becomes constant, or verify by inspection during
  implementation and substitute a dataset that does exhibit this behavior.

- [ ] **Step 3: Run to verify they fail**

  ```bash
  python -m pytest tests/test_problems.py::test_get_X_y_returns_four_tuple_with_correct_sizes tests/test_problems.py::test_get_X_y_seeded_reproducible tests/test_problems.py::test_get_X_y_zero_variance_filter_applied -v
  ```

  Expected: FAIL — `get_X_y() missing required argument 'n_train'` or unpacking error.

- [ ] **Step 4: Replace the existing `get_X_y` doctests**

  In the `EmpiricalDataProblem` docstring replace all existing `get_X_y` call examples
  (the `>>> X, y = ...` lines and their output lines) with these minimal usage examples:

  ```python
      Basic usage — returns (X_train, X_test, y_train, y_test):

      >>> import numpy as np
      >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
      >>> X_train, X_test, y_train, y_test = diabetes.get_X_y(300)
      >>> X_train.shape
      (300, 10)

      Dropping columns:

      >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
      ...                                        drop=['Froude_number'])
      >>> X_train, _, _, _ = yacht_no_froude.get_X_y(200)
      >>> X_train.shape[1]
      5

      NaN handling — drop rows:

      >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
      >>> X_train, _, _, _ = auto.get_X_y(100)
      >>> X_train.shape[0]
      100

      y_transforms:

      >>> diabetes_log = EmpiricalDataProblem('diabetes', 'target',
      ...                                     y_transforms=[np.log])
      >>> _, _, y_train_log, _ = diabetes_log.get_X_y(300, rng=np.random.default_rng(0))
      >>> _, _, y_train_base, _ = diabetes.get_X_y(300, rng=np.random.default_rng(0))
      >>> np.allclose(y_train_log.values, np.log(y_train_base.values))
      True

      x_transforms:

      >>> ohe = EmpiricalDataProblem('automobile', 'price',
      ...                            nan_policy='drop_rows',
      ...                            x_transforms=[OneHotEncodeCategories()])
      >>> X_train_ohe, _, _, _ = ohe.get_X_y(100)
      >>> 'fuel-type_gas' in X_train_ohe.columns
      True

      zero_variance_filter drops constant columns from both train and test:

      >>> p = EmpiricalDataProblem('crime', 'ViolentCrimesPerPop',
      ...     drop=['state', 'fold', 'communityname'], nan_policy='drop_cols')
      >>> p_zvf = EmpiricalDataProblem('crime', 'ViolentCrimesPerPop',
      ...     drop=['state', 'fold', 'communityname'], nan_policy='drop_cols',
      ...     zero_variance_filter=True)
      >>> Xtr, _, _, _ = p.get_X_y(1000, rng=np.random.default_rng(0))
      >>> Xtr_f, _, _, _ = p_zvf.get_X_y(1000, rng=np.random.default_rng(0))
      >>> Xtr_f.shape[1] < Xtr.shape[1]
      True
  ```

  **Note for implementer:** before finalising this doctest, verify that `crime` with
  `nan_policy='drop_cols'`, `n_train=1000`, and `rng=np.random.default_rng(0)` actually
  produces at least one zero-variance training column. If not, find a dataset/seed/n_train
  combination that does and substitute it. The intent is to show the filter removing a
  real constant column, not just a no-op call.

- [ ] **Step 5: Implement the new `get_X_y`**

  Replace the existing method body:

  ```python
  def get_X_y(self, n_train, rng=None):
      if not isinstance(rng, np.random.RandomState):
          rng = np.random.default_rng(rng)
      df = get_dataset(self.dataset)
      missing = [c for c in self.drop if c not in df.columns]
      if missing:
          warnings.warn(f"Columns not found in '{self.dataset}', skipping drop: {missing}")
      df = df.drop(columns=[c for c in self.drop if c in df.columns])
      df = df.dropna(subset=[self.target])
      if self.nan_policy == 'drop_rows':
          df = df.dropna()
      elif self.nan_policy == 'drop_cols':
          df = df.dropna(axis=1)
      df = df.reset_index(drop=True)
      X = df.drop(columns=[self.target])
      y = df[self.target]
      for fn in self.y_transforms:
          y = fn(y)
      for fn in self.x_transforms:
          X = fn(X, rng)
      X_train, X_test, y_train, y_test = train_test_split(
          X, y, train_size=n_train, random_state=rng)
      if self.zero_variance_filter:
          std = X_train.std()
          non_zero = std[std != 0].index
          X_train = X_train[non_zero]
          X_test = X_test[non_zero]
      return X_train, X_test, y_train, y_test
  ```

- [ ] **Step 6: Run all tests**

  ```bash
  python -m pytest tests/test_problems.py -v
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: all PASS.

- [ ] **Step 7: Commit**

  ```bash
  git add experiments/problems.py tests/test_problems.py
  git commit -m "feat: refactor get_X_y to accept n_train and rng, return train/test split"
  ```

---

## Task 4: Add `n_train_from_proportion` helper

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add the function with a doctest**

  Place after `EmpiricalDataProblem` and before `PolynomialExpansion`:

  ```python
  def n_train_from_proportion(problems, prop=0.7):
      """Return per-problem n_train ints derived from a proportion of dataset size.

      Uses the 'n' entry in DATASETS for each problem. Raises KeyError if 'n' is
      absent. The registry count may differ from the actual loaded row count when
      nan_policy drops rows.

      Examples
      --------
      >>> probs = [EmpiricalDataProblem('diabetes', 'target')]
      >>> int(n_train_from_proportion(probs)[0])
      309
      >>> int(n_train_from_proportion(probs, prop=0.8)[0])
      353
      """
      return np.array([int(DATASETS[p.dataset]['n'] * prop) for p in problems])
  ```

  (diabetes has n=442 in the registry; 442*0.7=309.4 → 309; 442*0.8=353.6 → 353)

- [ ] **Step 2: Run doctest**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py -k n_train_from_proportion -v
  ```

  Expected: PASS.

- [ ] **Step 3: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: add n_train_from_proportion helper"
  ```

---

## Task 5: Update NEURIPS2023 problem sets

**Files:**
- Modify: `experiments/problems.py`
- Modify: `tests/test_problems.py`

- [ ] **Step 1: Add a failing test**

  ```python
  from problems import NEURIPS2023, NEURIPS2023_D2, NEURIPS2023_D3


  def test_neurips2023_zero_variance_filter():
      assert all(p.zero_variance_filter for p in NEURIPS2023)
      assert all(p.zero_variance_filter for p in NEURIPS2023_D2)
      assert all(p.zero_variance_filter for p in NEURIPS2023_D3)
  ```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  python -m pytest tests/test_problems.py::test_neurips2023_zero_variance_filter -v
  ```

  Expected: FAIL.

- [ ] **Step 3: Add `zero_variance_filter=True` to all 23 NEURIPS2023 problems**

  Replace the `NEURIPS2023` frozenset. Every `EmpiricalDataProblem(...)` call gains
  `zero_variance_filter=True`:

  ```python
  NEURIPS2023 = frozenset({
      EmpiricalDataProblem('abalone',          'Rings',
                           x_transforms=_OHE, zero_variance_filter=True),
      EmpiricalDataProblem('airfoil',          'scaled-sound-pressure',
                           zero_variance_filter=True),
      EmpiricalDataProblem('automobile',       'price',
                           nan_policy='drop_rows',
                           x_transforms=_OHE,
                           y_transforms=[np.log],
                           zero_variance_filter=True),
      EmpiricalDataProblem('autompg',          'mpg',
                           drop=['car_name'], nan_policy='drop_rows',
                           zero_variance_filter=True),
      EmpiricalDataProblem('blog',             'V281',
                           zero_variance_filter=True),
      EmpiricalDataProblem('boston',           'medv',
                           zero_variance_filter=True),
      EmpiricalDataProblem('concrete',         'Concrete compressive strength',
                           zero_variance_filter=True),
      EmpiricalDataProblem('crime',            'ViolentCrimesPerPop',
                           drop=['state', 'fold', 'communityname'],
                           nan_policy='drop_cols',
                           zero_variance_filter=True),
      EmpiricalDataProblem('ct_slices',        'reference',
                           zero_variance_filter=True),
      EmpiricalDataProblem('diabetes',         'target',
                           zero_variance_filter=True),
      EmpiricalDataProblem('eye',              'y',
                           zero_variance_filter=True),
      EmpiricalDataProblem('facebook',         'Total Interactions',
                           drop=['comment', 'like', 'share'],
                           nan_policy='drop_rows',
                           x_transforms=_OHE,
                           zero_variance_filter=True),
      EmpiricalDataProblem('forest',           'area',
                           x_transforms=_OHE,
                           y_transforms=[np.log1p],
                           zero_variance_filter=True),
      EmpiricalDataProblem('naval_propulsion', 'GT_compressor_decay',
                           drop=['GT_turbine_decay'],
                           zero_variance_filter=True),
      EmpiricalDataProblem('naval_propulsion', 'GT_turbine_decay',
                           drop=['GT_compressor_decay'],
                           zero_variance_filter=True),
      EmpiricalDataProblem('parkinsons',       'motor_UPDRS',
                           drop=['total_UPDRS'],
                           zero_variance_filter=True),
      EmpiricalDataProblem('parkinsons',       'total_UPDRS',
                           drop=['motor_UPDRS'],
                           zero_variance_filter=True),
      EmpiricalDataProblem('real_estate',      'Y house price of unit area',
                           zero_variance_filter=True),
      EmpiricalDataProblem('ribo',             'y',
                           zero_variance_filter=True),
      EmpiricalDataProblem('student',          'G3',
                           drop=['G1', 'G2'],
                           x_transforms=_OHE,
                           zero_variance_filter=True),
      EmpiricalDataProblem('tomshw',           'V97',
                           zero_variance_filter=True),
      EmpiricalDataProblem('twitter',          'V78',
                           zero_variance_filter=True),
      EmpiricalDataProblem('yacht',            'Residuary_resistance',
                           y_transforms=[np.log],
                           zero_variance_filter=True),
  })
  ```

- [ ] **Step 4: Update `_with_polynomial` to copy `zero_variance_filter`**

  ```python
  def _with_polynomial(p, degree):
      return EmpiricalDataProblem(
          p.dataset, p.target, list(p.drop), p.nan_policy,
          x_transforms=list(p.x_transforms) + [PolynomialExpansion(degree)],
          y_transforms=list(p.y_transforms),
          zero_variance_filter=p.zero_variance_filter,
      )
  ```

- [ ] **Step 5: Run to verify tests pass**

  ```bash
  python -m pytest tests/test_problems.py::test_neurips2023_zero_variance_filter -v
  python -m pytest --doctest-modules experiments/problems.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/problems.py tests/test_problems.py
  git commit -m "feat: add zero_variance_filter=True to NEURIPS2023 problem sets"
  ```

---

## Task 6: Refactor `EmpiricalDataExperiment`

**Files:**
- Modify: `experiments/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Create `tests/test_experiments.py` with failing tests**

  ```python
  import sys
  import os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

  import numpy as np
  import pytest
  from fastridge import RidgeEM
  from problems import EmpiricalDataProblem, n_train_from_proportion
  from experiments import EmpiricalDataExperiment


  def _simple_exp(**kwargs):
      prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
      ns = n_train_from_proportion([prob])
      defaults = dict(seed=1, generator='MT19937', verbose=False)
      defaults.update(kwargs)
      return EmpiricalDataExperiment([prob], [RidgeEM()], reps=2, ns=ns, **defaults)


  def test_result_shape():
      assert _simple_exp().run().prediction_r2_.shape == (2, 1, 1, 1)


  def test_ns_shape():
      exp = _simple_exp()
      assert exp.ns.shape == (1, 1)
      assert int(exp.ns[0, 0]) == 309  # 442 * 0.7


  def test_make_rng_fixed_progression_same_seed():
      exp = _simple_exp(seed_progression='fixed')
      r0 = exp._make_rng(unit_idx=0)
      r5 = exp._make_rng(unit_idx=5)
      assert r0.randint(10000) == r5.randint(10000)


  def test_make_rng_sequential_progression_different_seeds():
      exp = _simple_exp(seed_progression='sequential')
      r0 = exp._make_rng(unit_idx=0)
      r1 = exp._make_rng(unit_idx=1)
      assert r0.randint(10000) != r1.randint(10000)


  def test_series_scope_reproducible():
      exp1 = _simple_exp()
      exp2 = _simple_exp()
      exp1.run()
      exp2.run()
      np.testing.assert_array_equal(exp1.prediction_r2_, exp2.prediction_r2_)


  def test_pcg64_and_mt19937_differ():
      exp_mt = _simple_exp(generator='MT19937').run()
      exp_pc = _simple_exp(generator='PCG64').run()
      # same seed, different generators — results should differ
      assert not np.array_equal(exp_mt.prediction_r2_, exp_pc.prediction_r2_)
  ```

  Note: `MT19937` uses `RandomState` which has `.randint`; `PCG64` uses `Generator`
  which has `.integers`. The `_make_rng` tests use `MT19937` (via `_simple_exp` default)
  to use `.randint`.

- [ ] **Step 2: Run to verify tests fail**

  ```bash
  python -m pytest tests/test_experiments.py -v
  ```

  Expected: FAIL — `__init__() got an unexpected keyword argument 'reps'`.

- [ ] **Step 3: Add `_RNG_FACTORIES` to `experiments/experiments.py`**

  Add after the imports and before the stat classes:

  ```python
  _RNG_FACTORIES = {
      'PCG64':   lambda seed: np.random.Generator(np.random.PCG64(seed)),
      'MT19937': lambda seed: np.random.RandomState(seed),
  }
  ```

- [ ] **Step 4: Replace `EmpiricalDataExperiment.__init__` and add `_make_rng`**

  Replace the existing `__init__` and update the class docstring Parameters section:

  ```python
  def __init__(self, problems, estimators, reps, ns,
               seed=None,
               generator='PCG64',
               seed_scope='series',
               seed_progression='fixed',
               stats=None, est_names=None, verbose=True):
      self.problems = problems
      self.estimators = estimators
      self.reps = reps
      self.ns = np.atleast_2d(ns)
      if len(self.ns) != len(self.problems):
          self.ns = self.ns.repeat(len(self.problems), axis=0)
      self.seed = seed
      self.generator = generator
      self.seed_scope = seed_scope
      self.seed_progression = seed_progression
      self._rng_factory = _RNG_FACTORIES[generator]
      self.stats = empirical_default_stats if stats is None else stats
      self.est_names = [str(e) for e in estimators] if est_names is None else est_names
      self.verbose = verbose

  def _make_rng(self, unit_idx):
      seed_val = None if self.seed is None else (
          self.seed if self.seed_progression == 'fixed' else self.seed + unit_idx
      )
      return self._rng_factory(seed_val)
  ```

  Update the docstring Parameters block to:

  ```
  Parameters
  ----------
  problems : list of EmpiricalDataProblem
  estimators : list of estimator objects
  reps : int
  ns : array-like
      Training set sizes. Broadcast to shape (n_problems, n_sizes).
      Use n_train_from_proportion() to derive from the dataset registry.
  seed : int or None
  generator : {'PCG64', 'MT19937'}, default 'PCG64'
      'MT19937' uses np.random.RandomState for legacy numerical equivalence.
  seed_scope : {'series', 'trial', 'experiment'}, default 'series'
      When the seed is reset: per-problem, per-rep, or once for the run.
  seed_progression : {'fixed', 'sequential'}, default 'fixed'
      'fixed' reuses seed; 'sequential' uses seed + unit_idx.
  stats : list of metric callables or None
  est_names : list of str or None
  verbose : bool, default True
  ```

- [ ] **Step 5: Replace `EmpiricalDataExperiment.run` and update its doctest**

  Replace the entire `run` method. Update the class-level doctest Examples to:

  ```python
      Examples
      --------
      >>> from fastridge import RidgeEM
      >>> from problems import EmpiricalDataProblem, n_train_from_proportion
      >>> prob = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
      >>> ns = n_train_from_proportion([prob])
      >>> exp = EmpiricalDataExperiment(
      ...     [prob], [RidgeEM()], reps=2, ns=ns,
      ...     seed=1, generator='MT19937', verbose=False)
      >>> exp.run().prediction_r2_.shape
      (2, 1, 1, 1)
      >>> exp.ns.shape
      (1, 1)
      >>> int(exp.ns[0, 0]) > 0
      True
  ```

  New `run` method:

  ```python
  def run(self):
      n_problems = len(self.problems)
      n_estimators = len(self.estimators)
      n_sizes = len(self.ns[0])

      for stat in self.stats:
          self.__dict__[str(stat) + '_'] = np.full(
              (self.reps, n_problems, n_sizes, n_estimators), np.nan)

      if self.seed_scope == 'experiment':
          self.rng = self._make_rng(unit_idx=0)

      for prob_idx, problem in enumerate(self.problems):
          if self.verbose:
              print(problem.dataset, end=' ')

          for n_idx, n_train in enumerate(self.ns[prob_idx]):
              if self.seed_scope == 'series':
                  self.rng = self._make_rng(unit_idx=prob_idx)

              for iter_idx in range(self.reps):
                  if self.verbose:
                      print('.', end='')

                  if self.seed_scope == 'trial':
                      self.rng = self._make_rng(unit_idx=iter_idx)

                  X_train, X_test, y_train, y_test = problem.get_X_y(
                      n_train, rng=self.rng)

                  for est_idx, est in enumerate(self.estimators):
                      _est = clone(est, safe=False)
                      try:
                          t0 = time.time()
                          _est.fit(X_train, y_train)
                          _est.fitting_time_ = time.time() - t0
                      except Exception as e:
                          warnings.warn(
                              f"Run {iter_idx} failed for '{self.est_names[est_idx]}'"
                              f" on '{problem.dataset}': {e}")
                          continue

                      for stat in self.stats:
                          self.__dict__[str(stat) + '_'][
                              iter_idx, prob_idx, n_idx, est_idx] = stat(
                                  _est, problem, X_test, y_test)

          if self.verbose:
              print()

      return self
  ```

- [ ] **Step 6: Run all tests**

  ```bash
  python -m pytest tests/test_experiments.py -v
  python -m pytest --doctest-modules experiments/experiments.py::experiments.EmpiricalDataExperiment -v
  ```

  Expected: all PASS.

- [ ] **Step 7: Run full test suite**

  ```bash
  python -m pytest --doctest-modules fastridge.py experiments/data.py experiments/problems.py experiments/experiments.py tests/ -v
  ```

  Expected: all PASS.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/experiments.py tests/test_experiments.py
  git commit -m "feat: refactor EmpiricalDataExperiment with explicit generators and ns/reps"
  ```

---

## Task 7: Update notebooks

**Files:**
- Modify: `experiments/real_data.ipynb`
- Modify: `experiments/real_data_neurips2023.ipynb`

These notebooks have `skip-execution` tagged cells for the heavy experiment runs. Only
update the experiment runner call sites and imports; do not alter problem definitions,
estimator lists, or analysis cells.

### `real_data_neurips2023.ipynb`

- [ ] **Step 1: Update cell 2 — import and d1 experiment**

  Read cell 2 with the `Read` tool before editing. Replace the cell source with:

  ```python
  import numpy as np
  from fastridge import RidgeEM, RidgeLOOCV
  from experiments import EmpiricalDataExperiment
  from problems import NEURIPS2023, n_train_from_proportion
  from data import DATASETS

  # estimator indices: 0=EM, 1=CV_fix, 2=CV_glm
  estimators = [
      RidgeEM(t2=False),
      RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
      RidgeLOOCV(alphas=100),
  ]
  est_names = ['EM', 'CV_fix', 'CV_glm']

  problems_d1 = sorted(NEURIPS2023, key=lambda p: DATASETS[p.dataset]['n'])
  exp_d1 = EmpiricalDataExperiment(
      problems_d1, estimators,
      reps=100, ns=n_train_from_proportion(problems_d1),
      seed=123, generator='MT19937',
      est_names=est_names, verbose=True).run()
  print()
  ```

- [ ] **Step 2: Update cell 5 — d2 experiment**

  ```python
  from problems import NEURIPS2023_D2

  problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
  exp_d2 = EmpiricalDataExperiment(
      problems_d2, estimators,
      reps=100, ns=n_train_from_proportion(problems_d2),
      seed=123, generator='MT19937',
      est_names=est_names, verbose=True).run()
  print()
  ```

- [ ] **Step 3: Update cell 8 — d3 experiment**

  ```python
  from problems import NEURIPS2023_D3

  problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
  exp_d3 = EmpiricalDataExperiment(
      problems_d3, estimators,
      reps=100, ns=n_train_from_proportion(problems_d3),
      seed=123, generator='MT19937',
      est_names=est_names, verbose=True).run()
  print()
  ```

### `real_data.ipynb`

- [ ] **Step 4: Update cell 2 — add `n_train_from_proportion` to imports**

  Read cell 2 first. Add `n_train_from_proportion` to the `from problems import` line:

  ```python
  from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion, n_train_from_proportion
  ```

  **Note:** `real_data.ipynb` contains inline `EmpiricalDataProblem` definitions in cells
  13+ (`problems_full`, `problems_large`, etc.) that will default to
  `zero_variance_filter=False` after the refactor. This intentionally changes behavior
  relative to the current code (which applied the filter unconditionally). If unexpected
  results arise in those cells, add `zero_variance_filter=True` to each inline problem
  definition to restore the old behavior.

- [ ] **Step 5: Update cells 5, 8, 16, 19, 23, 26 — all `EmpiricalDataExperiment` calls**

  For each cell, replace `n_iterations=<N>, seed=123` with
  `reps=<N>, ns=n_train_from_proportion(<problem_list>), seed=123, generator='MT19937'`.

  | Cell | Problem list variable | n_iterations value |
  |------|-----------------------|-------------------|
  | 5    | `problems_d2`         | 10 |
  | 8    | `problems_d3`         | 10 |
  | 16   | `problems_full_d2`    | 30 |
  | 19   | `problems_full_d3`    | 30 |
  | 23   | `problems_large`      | 30 |
  | 26   | `problems_large_d2`   | 30 |

  Example for cell 5:

  ```python
  exp_d2 = EmpiricalDataExperiment(
      problems_d2, list(estimators.values()),
      reps=10, ns=n_train_from_proportion(problems_d2),
      seed=123, generator='MT19937',
      est_names=list(estimators.keys())).run()
  print()
  ```

- [ ] **Step 6: Run the non-skipped notebook tests**

  ```bash
  python -m pytest --nbmake experiments/real_data.ipynb experiments/real_data_neurips2023.ipynb -v
  ```

  Light cells (imports, problem definitions, analysis) should execute without error.

- [ ] **Step 7: Run full test suite**

  ```bash
  python -m pytest --doctest-modules fastridge.py experiments/data.py experiments/problems.py experiments/experiments.py tests/ -v
  ```

  Expected: all PASS.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/real_data.ipynb experiments/real_data_neurips2023.ipynb
  git commit -m "feat: update notebooks to use new EmpiricalDataExperiment API"
  ```
