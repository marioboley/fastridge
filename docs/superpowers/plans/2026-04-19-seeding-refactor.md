# Seeding Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `EmpiricalDataProblem.get_X_y` and `EmpiricalDataExperiment` to thread explicit generator objects through the data pipeline, enabling reproducible seeding at configurable scopes and a stable `n_train` value before the trial loop.

**Architecture:** `get_X_y(n_train, rng)` absorbs the train/test split and optional zero-variance filter; the experiment runner creates generator objects via `_make_rng` and passes them down. x_transforms gain an `rng=None` keyword so stochastic transforms (e.g. `PolynomialExpansion` subsampling) consume the same generator as the split. `EmpiricalDataExperiment` replaces `test_prop`/`n_iterations` with `ns`/`reps` and adds `generator`, `seed_scope`, `seed_progression` to control seeding granularity.

**Tech Stack:** numpy (`np.random.Generator`, `np.random.PCG64`, `np.random.RandomState`), sklearn `train_test_split`, pytest doctests.

**Spec:** `docs/superpowers/specs/2026-04-18-empirical-experiment-seeding-refactor-design.md`

---

## File Map

| File | Changes |
|------|---------|
| `experiments/problems.py` | Add `rng=None` to `PolynomialExpansion.__call__` and `OneHotEncodeCategories.__call__`; add `zero_variance_filter` to `EmpiricalDataProblem`; refactor `get_X_y`; add `n_train_from_proportion`; update NEURIPS2023 sets |
| `experiments/experiments.py` | Replace `EmpiricalDataExperiment` constructor and `run()`; add `_RNG_FACTORIES` and `_make_rng` |
| `experiments/real_data.ipynb` | Update imports and all `EmpiricalDataExperiment` call sites |
| `experiments/real_data_neurips2023.ipynb` | Update all `EmpiricalDataExperiment` call sites |

---

## Task 1: Add `rng` parameter to x_transforms

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add a failing doctest to `PolynomialExpansion`**

  In the `PolynomialExpansion` docstring, after the existing `'a' in result.columns and 'b' in result.columns` test, add:

  ```python
      Subsampling is deterministic when rng is provided:

      >>> rng1 = np.random.default_rng(0)
      >>> rng2 = np.random.default_rng(0)
      >>> list(small(X, rng=rng1).columns) == list(small(X, rng=rng2).columns)
      True
  ```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  cd /Users/marioboley/Documents/GitHub/fastridge && python -m pytest --doctest-modules experiments/problems.py::problems.PolynomialExpansion -v
  ```

  Expected: FAIL — `__call__() takes 2 positional arguments but 3 were given` or similar.

- [ ] **Step 3: Update `PolynomialExpansion.__call__`**

  Replace the current `__call__` in `experiments/problems.py`:

  ```python
  def __call__(self, X, rng=None):
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
          _rng = rng or np.random.default_rng()
          sampled = sorted(_rng.choice(len(interaction_cols), size=pnew, replace=False))
          return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
      return X_poly
  ```

- [ ] **Step 4: Add `rng=None` to `OneHotEncodeCategories.__call__`**

  Replace the current signature:

  ```python
  def __call__(self, X, rng=None):
  ```

  Body is unchanged — `rng` is accepted and ignored.

- [ ] **Step 5: Run doctests to verify both pass**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.PolynomialExpansion experiments/problems.py::problems.OneHotEncodeCategories -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: add rng parameter to PolynomialExpansion and OneHotEncodeCategories"
  ```

---

## Task 2: Add `zero_variance_filter` to `EmpiricalDataProblem`

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add failing doctests for the new parameter**

  In the `EmpiricalDataProblem` docstring, add after the existing value-object identity tests:

  ```python
      zero_variance_filter defaults to False and appears in repr only when True:

      >>> EmpiricalDataProblem('diabetes', 'target').zero_variance_filter
      False
      >>> repr(EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True))
      "EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)"
      >>> EmpiricalDataProblem('diabetes', 'target') == EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=False)
      True
      >>> EmpiricalDataProblem('diabetes', 'target') == EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
      False
  ```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: FAIL — `__init__() got an unexpected keyword argument 'zero_variance_filter'`.

- [ ] **Step 3: Add `zero_variance_filter` to `__init__` and `_repr`**

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
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: add zero_variance_filter parameter to EmpiricalDataProblem"
  ```

---

## Task 3: Refactor `get_X_y`

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add `train_test_split` import**

  At the top of `experiments/problems.py`, add to the sklearn import line:

  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
  ```

- [ ] **Step 2: Replace the existing `get_X_y` doctests**

  In the `EmpiricalDataProblem` docstring, replace all existing `get_X_y` usage examples (the lines starting `>>> X, y = ...` through the `x_transforms` OHE example) with:

  ```python
      Basic usage — returns (X_train, X_test, y_train, y_test):

      >>> import numpy as np
      >>> diabetes = EmpiricalDataProblem('diabetes', 'target')
      >>> X_train, X_test, y_train, y_test = diabetes.get_X_y(300)
      >>> X_train.shape
      (300, 10)
      >>> X_test.shape
      (142, 10)

      Seeded calls are reproducible:

      >>> rng1 = np.random.default_rng(0)
      >>> rng2 = np.random.default_rng(0)
      >>> Xtr1, _, _, _ = diabetes.get_X_y(300, rng=rng1)
      >>> Xtr2, _, _, _ = diabetes.get_X_y(300, rng=rng2)
      >>> list(Xtr1.index) == list(Xtr2.index)
      True

      Dropping columns:

      >>> yacht_no_froude = EmpiricalDataProblem('yacht', 'Residuary_resistance',
      ...                                        drop=['Froude_number'])
      >>> X_train, _, _, _ = yacht_no_froude.get_X_y(200)
      >>> X_train.shape[1]
      5

      NaN handling — drop rows (index is reset before splitting):

      >>> auto = EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows')
      >>> X_train, X_test, y_train, y_test = auto.get_X_y(100)
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
      >>> X_train_ohe.shape[0]
      100
      >>> 'fuel-type_gas' in X_train_ohe.columns
      True

      zero_variance_filter drops constant columns from both train and test:

      >>> p_zvf = EmpiricalDataProblem('diabetes', 'target', zero_variance_filter=True)
      >>> X_train, X_test, _, _ = p_zvf.get_X_y(300, rng=np.random.default_rng(0))
      >>> X_train.shape[1]
      10
  ```

  Note: `(442 - 300) = 142` for diabetes test size.

- [ ] **Step 3: Run to verify the new doctests fail**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: FAIL — `get_X_y() missing required argument 'n_train'` or return-value unpacking error.

- [ ] **Step 4: Implement the new `get_X_y`**

  Replace the existing `get_X_y` method body:

  ```python
  def get_X_y(self, n_train, rng=None):
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
          X = fn(X, rng=rng)
      X_train, X_test, y_train, y_test = train_test_split(
          X, y, train_size=n_train, random_state=rng)
      if self.zero_variance_filter:
          std = X_train.std()
          non_zero = std[std != 0].index
          X_train = X_train[non_zero]
          X_test = X_test[non_zero]
      return X_train, X_test, y_train, y_test
  ```

- [ ] **Step 5: Run to verify all `EmpiricalDataProblem` doctests pass**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py::problems.EmpiricalDataProblem -v
  ```

  Expected: PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: refactor get_X_y to accept n_train and rng, return train/test split"
  ```

---

## Task 4: Add `n_train_from_proportion` helper

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add a failing doctest**

  Add the following as a module-level doctest — place the function and its docstring directly after the `EmpiricalDataProblem` class definition and before `PolynomialExpansion`:

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

  (diabetes has n=442; 442*0.7=309.4 → int=309; 442*0.8=353.6 → int=353)

- [ ] **Step 2: Run to verify it fails**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py -k n_train_from_proportion -v
  ```

  Expected: FAIL — `name 'n_train_from_proportion' is not defined`.

- [ ] **Step 3: Run after adding to file**

  The code in Step 1 already includes the implementation. Confirm pytest collects and passes it:

  ```bash
  python -m pytest --doctest-modules experiments/problems.py -k n_train_from_proportion -v
  ```

  Expected: PASS.

- [ ] **Step 4: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: add n_train_from_proportion helper"
  ```

---

## Task 5: Update NEURIPS2023 problem sets

**Files:**
- Modify: `experiments/problems.py`

- [ ] **Step 1: Add a failing doctest**

  At module level in `problems.py`, add a comment and doctest block (or add to the module docstring). The cleanest place is directly after the `NEURIPS2023_D3` definition:

  ```python
  def _neurips_sets_have_zero_variance_filter():
      """All NEURIPS2023 problem sets use zero_variance_filter=True.

      Examples
      --------
      >>> all(p.zero_variance_filter for p in NEURIPS2023)
      True
      >>> all(p.zero_variance_filter for p in NEURIPS2023_D2)
      True
      >>> all(p.zero_variance_filter for p in NEURIPS2023_D3)
      True
      """
  ```

- [ ] **Step 2: Run to verify it fails**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py -k _neurips_sets_have_zero_variance_filter -v
  ```

  Expected: FAIL — all assertions return `False`.

- [ ] **Step 3: Add `zero_variance_filter=True` to all 23 NEURIPS2023 problems**

  Replace the `NEURIPS2023` frozenset definition. Every `EmpiricalDataProblem(...)` call gains `zero_variance_filter=True`:

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

  Replace the existing `_with_polynomial` function:

  ```python
  def _with_polynomial(p, degree):
      return EmpiricalDataProblem(
          p.dataset, p.target, list(p.drop), p.nan_policy,
          x_transforms=list(p.x_transforms) + [PolynomialExpansion(degree)],
          y_transforms=list(p.y_transforms),
          zero_variance_filter=p.zero_variance_filter,
      )
  ```

- [ ] **Step 5: Run to verify all tests pass**

  ```bash
  python -m pytest --doctest-modules experiments/problems.py -v
  ```

  Expected: all PASS.

- [ ] **Step 6: Commit**

  ```bash
  git add experiments/problems.py
  git commit -m "feat: add zero_variance_filter=True to NEURIPS2023 problem sets"
  ```

---

## Task 6: Refactor `EmpiricalDataExperiment`

**Files:**
- Modify: `experiments/experiments.py`

- [ ] **Step 1: Update the docstring example to use the new API**

  Replace the existing `EmpiricalDataExperiment` docstring `Examples` section:

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

- [ ] **Step 2: Run to verify it fails**

  ```bash
  python -m pytest --doctest-modules experiments/experiments.py::experiments.EmpiricalDataExperiment -v
  ```

  Expected: FAIL — `__init__() got an unexpected keyword argument 'reps'`.

- [ ] **Step 3: Add `_RNG_FACTORIES` module-level dict**

  Add after the imports and before the stat classes:

  ```python
  _RNG_FACTORIES = {
      'PCG64':   lambda seed: np.random.Generator(np.random.PCG64(seed)),
      'MT19937': lambda seed: np.random.RandomState(seed),
  }
  ```

- [ ] **Step 4: Replace `EmpiricalDataExperiment.__init__`**

  Replace the existing `__init__` method and add `_make_rng`:

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

  Also update the docstring parameters section to reflect the new parameters:

  ```python
      Parameters
      ----------
      problems : list of EmpiricalDataProblem
      estimators : list of estimator objects
      reps : int
      ns : array-like
          Training set sizes. Broadcast to shape (n_problems, n_sizes).
          Use n_train_from_proportion() to derive from dataset registry.
      seed : int or None
      generator : {'PCG64', 'MT19937'}, default 'PCG64'
          'MT19937' uses np.random.RandomState for legacy equivalence.
      seed_scope : {'series', 'trial', 'experiment'}, default 'series'
          When the seed is reset.
      seed_progression : {'fixed', 'sequential'}, default 'fixed'
          Whether to use seed or seed + unit_idx at each scope boundary.
      stats : list of metric callables or None
      est_names : list of str or None
      verbose : bool, default True
  ```

- [ ] **Step 5: Replace `EmpiricalDataExperiment.run`**

  Replace the entire `run` method:

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

  Note: `train_test_split` import in `experiments.py` can be removed — it is no longer used here.

- [ ] **Step 6: Run to verify doctest passes**

  ```bash
  python -m pytest --doctest-modules experiments/experiments.py::experiments.EmpiricalDataExperiment -v
  ```

  Expected: PASS.

- [ ] **Step 7: Run full test suite**

  ```bash
  python -m pytest --doctest-modules fastridge.py experiments/data.py experiments/problems.py experiments/experiments.py -v
  ```

  Expected: all PASS.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/experiments.py
  git commit -m "feat: refactor EmpiricalDataExperiment with explicit generators and ns/reps"
  ```

---

## Task 7: Update notebooks

**Files:**
- Modify: `experiments/real_data.ipynb`
- Modify: `experiments/real_data_neurips2023.ipynb`

These notebooks are tagged `skip-execution` on heavy cells — check CI expectations before editing. Only update the experiment runner call sites and imports; do not alter problem definitions, estimator lists, or analysis cells.

### `real_data_neurips2023.ipynb`

- [ ] **Step 1: Update cell 2 — import and d1 experiment**

  Read cell 2 with the `Read` tool before editing. Replace the `EmpiricalDataExperiment` call:

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

  Read cell 2 first. Add `n_train_from_proportion` to the `from problems import` line.

  **Note:** `real_data.ipynb` contains inline `EmpiricalDataProblem` definitions in cells 13+
  (`problems_full`, `problems_large`, etc.) that will default to `zero_variance_filter=False`
  after the refactor. This intentionally changes behavior relative to the current code (which
  applied the filter unconditionally). If unexpected results arise in those cells, add
  `zero_variance_filter=True` to each inline problem definition to restore the old behavior.

  ```python
  from problems import EmpiricalDataProblem, OneHotEncodeCategories, PolynomialExpansion, n_train_from_proportion
  ```

- [ ] **Step 5: Update cells 5, 8, 16, 19, 23, 26 — all `EmpiricalDataExperiment` calls**

  For each cell, replace the `n_iterations=<N>, seed=123` pattern with
  `reps=<N>, ns=n_train_from_proportion(<problem_list>), seed=123, generator='MT19937'`.

  The problem list variable name to pass to `n_train_from_proportion` matches the
  first argument of `EmpiricalDataExperiment` in that cell:

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

  These notebooks have `skip-execution` tagged cells for the heavy experiment runs. The light cells should execute without error. If a cell fails due to a missing dataset download, that is pre-existing and unrelated.

- [ ] **Step 7: Run full test suite**

  ```bash
  python -m pytest --doctest-modules fastridge.py experiments/data.py experiments/problems.py experiments/experiments.py -v
  ```

  Expected: all PASS.

- [ ] **Step 8: Commit**

  ```bash
  git add experiments/real_data.ipynb experiments/real_data_neurips2023.ipynb
  git commit -m "feat: update notebooks to use new EmpiricalDataExperiment API"
  ```
