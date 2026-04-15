# EmpiricalDataExperiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `EmpiricalDataExperiment` as a peer class to `Experiment`, storing per-run results as numpy arrays with graceful NaN-on-failure handling, and update both notebooks to use it.

**Architecture:** Add two new metric classes (`PredictionR2`, `NumberOfFeatures`) and `EmpiricalDataExperiment` to `experiments/experiments.py`. Update `real_data.ipynb` (preview + full experiment) and `real_data_neurips2023.ipynb` to call the class instead of `run_real_data_experiments`. The legacy function is retained unchanged.

**Tech Stack:** `experiments/experiments.py`, `experiments/real_data.ipynb`, `experiments/real_data_neurips2023.ipynb`, pytest doctests, nbmake.

---

## Files

- Modify: `experiments/experiments.py` — add metric classes, `empirical_default_stats`, `EmpiricalDataExperiment`
- Modify: `pytest.ini` — add `experiments/experiments.py` to doctest-modules
- Modify: `experiments/real_data.ipynb` — update 14 cells (Task 2 + Task 3)
- Modify: `experiments/real_data_neurips2023.ipynb` — update 7 cells (Task 4)

---

## Task 1: Add metric classes and `EmpiricalDataExperiment` to `experiments.py`

**Files:** `experiments/experiments.py`, `pytest.ini`

- [ ] **Step 1: Add `experiments/experiments.py` to doctest-modules in `pytest.ini`**

Edit `pytest.ini` so it reads:

```ini
[pytest]
addopts =
    --doctest-modules fastridge.py
    --doctest-modules experiments/data.py
    --doctest-modules experiments/problems.py
    --doctest-modules experiments/experiments.py
    --codeblocks README.md
    --nbmake experiments/double_asymptotic_trends.ipynb
    --nbmake experiments/sparse_designs.ipynb
    --nbmake experiments/tutorial.ipynb
    --nbmake experiments/real_data.ipynb
```

- [ ] **Step 2: Add `import warnings` to `experiments/experiments.py`**

Insert `import warnings` after the existing `import time` line (line 1).

- [ ] **Step 3: Add `PredictionR2` metric class and singleton**

Insert after the `FittingTime` class and before the singleton instances (currently line 112):

```python
class PredictionR2:
    """Computes R² between predictions and test targets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> round(float(prediction_r2(est, None, X, y)), 4)
    1.0
    """

    @staticmethod
    def __call__(est, prob, x, y):
        return r2_score(y, est.predict(x))

    @staticmethod
    def __str__():
        return 'prediction_r2'

    @staticmethod
    def symbol():
        return r'$R^2$'
```

- [ ] **Step 4: Add `NumberOfFeatures` metric class and singleton**

Insert immediately after `PredictionR2`:

```python
class NumberOfFeatures:
    """Returns the number of features used by the estimator (len of coef_).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> X = np.arange(10).reshape(-1, 1).astype(float)
    >>> y = np.arange(10).astype(float)
    >>> est = Ridge(alpha=0.0001).fit(X, y)
    >>> number_of_features(est, None, X, y)
    1
    """

    @staticmethod
    def __call__(est, prob, x, y):
        if hasattr(est, 'coef_'):
            return len(est.coef_)
        return float('nan')

    @staticmethod
    def __str__():
        return 'number_of_features'

    @staticmethod
    def symbol():
        return r'$p$'
```

- [ ] **Step 5: Add singletons and `empirical_default_stats`**

Add these lines alongside the existing singletons (after line 117 `fitting_time = FittingTime()`):

```python
prediction_r2 = PredictionR2()
number_of_features = NumberOfFeatures()

empirical_default_stats = [
    prediction_mean_squared_error,
    prediction_r2,
    regularization_parameter,
    number_of_iterations,
    fitting_time,
    number_of_features,
]
```

- [ ] **Step 6: Run doctests to confirm new metric classes pass**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest experiments/experiments.py --doctest-modules -v 2>&1 | tail -15
```

Expected: doctests for `PredictionR2` and `NumberOfFeatures` PASS.

- [ ] **Step 7: Add `EmpiricalDataExperiment` class**

Insert after `run_real_data_experiments` at the end of `experiments/experiments.py`:

```python
class EmpiricalDataExperiment:
    """Run repeated train/test experiments on a list of EmpiricalDataProblem instances.

    Stores per-run results as numpy arrays of shape
    ``(n_iterations, n_problems, 1, n_estimators)`` per metric, matching the
    array layout of ``Experiment``. Failed runs (exception during fit) are
    recorded as NaN and trigger a ``warnings.warn``. The seed is reset before
    each problem's iteration loop, replicating the exact behaviour of the
    legacy ``run_real_data_experiments`` function so that each problem gets the
    same deterministic split sequence regardless of list ordering.

    Parameters
    ----------
    problems : list of EmpiricalDataProblem
    estimators : list of estimator objects
    n_iterations : int
    test_prop : float, default 0.3
    seed : int or None
    polynomial : int or None
        If given, applies ``PolynomialFeatures(degree=polynomial)`` after OHE.
    stats : list of metric callables or None
        Each callable has signature ``(est, prob, x, y)``. Defaults to
        ``empirical_default_stats``.
    est_names : list of str or None
        Defaults to ``[str(e) for e in estimators]``.
    verbose : bool, default True

    Examples
    --------
    >>> from fastridge import RidgeEM
    >>> from problems import EmpiricalDataProblem
    >>> prob = EmpiricalDataProblem('diabetes', 'target')
    >>> exp = EmpiricalDataExperiment(
    ...     [prob], [RidgeEM()], n_iterations=2, seed=1, verbose=False)
    >>> exp.run().prediction_r2_.shape
    (2, 1, 1, 1)
    >>> exp.ns.shape
    (1, 1)
    >>> int(exp.ns[0, 0]) > 0
    True
    """

    def __init__(self, problems, estimators, n_iterations, test_prop=0.3,
                 seed=None, polynomial=None, stats=None, est_names=None,
                 verbose=True):
        self.problems = problems
        self.estimators = estimators
        self.n_iterations = n_iterations
        self.test_prop = test_prop
        self.seed = seed
        self.polynomial = polynomial
        self.stats = empirical_default_stats if stats is None else stats
        self.est_names = [str(e) for e in estimators] if est_names is None else est_names
        self.verbose = verbose

    def run(self):
        n_problems = len(self.problems)
        n_estimators = len(self.estimators)

        for stat in self.stats:
            self.__dict__[str(stat) + '_'] = np.full(
                (self.n_iterations, n_problems, 1, n_estimators), np.nan)
        self.ns = np.zeros((n_problems, 1), dtype=int)

        for prob_idx, problem in enumerate(self.problems):
            X, y = problem.get_X_y()

            if self.verbose:
                print(problem.dataset, end=' ')

            categorical_cols = [col for col in X.columns
                                if not pd.api.types.is_numeric_dtype(X[col])]
            if categorical_cols:
                encoder = OneHotEncoder(drop='first', sparse_output=False)
                encoded = encoder.fit_transform(X[categorical_cols])
                X = pd.concat([
                    X.drop(categorical_cols, axis=1),
                    pd.DataFrame(encoded,
                                 columns=encoder.get_feature_names_out(categorical_cols))
                ], axis=1)

            if self.polynomial is not None:
                poly = PolynomialFeatures(degree=self.polynomial, include_bias=False)
                X_poly = poly.fit_transform(X)
                X_poly = pd.DataFrame(X_poly,
                                      columns=poly.get_feature_names_out(X.columns))
                npoly, ppoly = X_poly.shape
                if npoly * ppoly > 35_000_000:
                    X_poly = X_poly.drop(X.columns, axis=1)
                    pnew = int(np.ceil(35_000_000 / npoly))
                    X_poly = X_poly.iloc[
                        :, np.random.choice(X_poly.shape[1], size=pnew, replace=False)]
                    X = pd.concat([X, X_poly], axis=1)
                else:
                    X = X_poly

            self.ns[prob_idx, 0] = int(X.shape[0] * (1 - self.test_prop))

            if self.verbose:
                print(f'(n={X.shape[0]}, p={X.shape[1]})', end='')

            if self.seed is not None:
                np.random.seed(self.seed)

            for iter_idx in range(self.n_iterations):
                if self.verbose:
                    print('.', end='')

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_prop)

                std = X_train.std()
                non_zero = std[std != 0].index
                X_train = X_train[non_zero]
                X_test = X_test[non_zero]

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
                            iter_idx, prob_idx, 0, est_idx] = stat(
                                _est, problem, X_test, y_test)

            if self.verbose:
                print()

        return self
```

- [ ] **Step 8: Run full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --ignore=experiments/real_data.ipynb 2>&1 | tail -10
```

Expected: all tests pass (excluding `real_data.ipynb` which currently fails due to SVD error — fixed in Task 2).

- [ ] **Step 9: Commit**

```bash
git add experiments/experiments.py pytest.ini
git commit -m "feat: add PredictionR2, NumberOfFeatures, EmpiricalDataExperiment to experiments"
```

---

## Task 2: Update `real_data.ipynb` preview experiment (non-skip-execution cells)

**Files:** `experiments/real_data.ipynb`

These are the cells executed by `pytest --nbmake`. Fixing them makes CI pass again.

**Result access pattern used throughout:** define a helper `stat_mean(exp, stat_name, est_name, problem_idx)` once in the first table cell, then reuse it. `stat_name` is the string returned by `str(stat)` without the trailing `_` (e.g. `'prediction_r2'`, `'fitting_time'`, `'number_of_features'`).

- [ ] **Step 1: Read `real_data.ipynb` to confirm cell IDs**

Use the Read tool on `experiments/real_data.ipynb` before any NotebookEdit. Confirm:
- cell id `76d89b51` — imports + d=1 run
- cell id `6b69df71` — d=1 table
- cell id `f6bc8769` — d=2 run
- cell id `e8c4ddaa` — d=2 table
- cell id `094956a8` — d=3 run
- cell id `59535500` — d=3 table
- cell id `65e15066` — `make_figure3` definition (the old duplicate definition cell `38ec21c7` has been removed)
- cell id `0fccfcde` — `make_figure3` call

- [ ] **Step 2: Update cell `76d89b51` — imports + d=1 run**

Replace source with:

```python
import numpy as np
import pandas as pd

from fastridge import RidgeEM, RidgeLOOCV
from experiments import EmpiricalDataExperiment
from problems import EmpiricalDataProblem

problems = [
    EmpiricalDataProblem('abalone',    'Rings'),
    EmpiricalDataProblem('airfoil',    'scaled-sound-pressure'),
    EmpiricalDataProblem('concrete',   'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',   'target'),
    EmpiricalDataProblem('eye',        'y'),
    EmpiricalDataProblem('forest',     'area'),
    EmpiricalDataProblem('student',    'G3', drop=['G1', 'G2']),
    EmpiricalDataProblem('yacht',      'Residuary_resistance'),
    EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows'),
]

estimators = {
    'EM':     RidgeEM(),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
    'CV_glm': RidgeLOOCV(alphas=100),
}

exp = EmpiricalDataExperiment(
    problems, list(estimators.values()),
    n_iterations=10, seed=123,
    est_names=list(estimators.keys())).run()
print()
```

- [ ] **Step 3: Update cell `6b69df71` — d=1 table**

Replace source with:

```python
def stat_mean(exp, stat_name, est_name, problem_idx):
    j = exp.est_names.index(est_name)
    return np.nanmean(getattr(exp, stat_name + '_')[:, problem_idx, 0, j])

rows = []
for i, problem in enumerate(exp.problems):
    em_time = stat_mean(exp, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp, 'prediction_r2', est, i) for est in exp.est_names})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = stat_mean(exp, 'number_of_features', 'EM', i)
    row['n_train']  = int(exp.ns[i, 0])
    row['n:p']      = int(exp.ns[i, 0]) / stat_mean(exp, 'number_of_features', 'EM', i)
    rows.append(row)
pd.DataFrame(rows).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 4: Update cell `f6bc8769` — d=2 run**

Replace source with:

```python
exp_d2 = EmpiricalDataExperiment(
    problems_d2, list(estimators.values()),
    n_iterations=10, seed=123, polynomial=2,
    est_names=list(estimators.keys())).run()
print()
```

Note: `problems_d2` is defined in cell `229399e5` which is unchanged. `estimators` is defined in cell `76d89b51`.

- [ ] **Step 5: Update cell `e8c4ddaa` — d=2 table**

Replace source with:

```python
rows_d2 = []
for i, problem in enumerate(exp_d2.problems):
    em_time = stat_mean(exp_d2, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_d2, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_d2, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_d2, 'prediction_r2', est, i) for est in exp_d2.est_names})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = stat_mean(exp_d2, 'number_of_features', 'EM', i)
    row['n_train']  = int(exp_d2.ns[i, 0])
    row['n:p']      = int(exp_d2.ns[i, 0]) / stat_mean(exp_d2, 'number_of_features', 'EM', i)
    rows_d2.append(row)
pd.DataFrame(rows_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 6: Update cell `094956a8` — d=3 run**

Replace source with:

```python
exp_d3 = EmpiricalDataExperiment(
    problems_d3, list(estimators.values()),
    n_iterations=10, seed=123, polynomial=3,
    est_names=list(estimators.keys())).run()
print()
```

Note: `problems_d3` is defined in cell `994819b1` which is unchanged.

- [ ] **Step 7: Update cell `59535500` — d=3 table**

Replace source with:

```python
rows_d3 = []
for i, problem in enumerate(exp_d3.problems):
    em_time = stat_mean(exp_d3, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_d3, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_d3, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_d3, 'prediction_r2', est, i) for est in exp_d3.est_names})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = stat_mean(exp_d3, 'number_of_features', 'EM', i)
    row['n_train']  = int(exp_d3.ns[i, 0])
    row['n:p']      = int(exp_d3.ns[i, 0]) / stat_mean(exp_d3, 'number_of_features', 'EM', i)
    rows_d3.append(row)
pd.DataFrame(rows_d3).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 8: Update cell `65e15066` — `make_figure3` (2nd definition)**

This is the active definition (overrides cell `38ec21c7`). Replace source with:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_figure3(exp_d1, exp_d2, exp_d3, output_path=None):
    """2x3 scatter of EM vs CV R² for d=1,2,3. Color = speed-up ratio.

    Top row: CV_glm on y-axis. Bottom row: CV_fix on y-axis.
    Values below CLIP_MIN=-0.1 are clipped to CLIP_MIN and shown with a dashed edge.
    Axis limits extend PAD beyond [CLIP_MIN, 1] so boundary points are fully visible.

    Each argument may be a single EmpiricalDataExperiment or a list of them
    (use a list to concatenate results from multiple experiments for the same degree).
    """
    def _as_list(e):
        return e if isinstance(e, list) else [e]

    def _r2(exps, est_name):
        result = []
        for exp in _as_list(exps):
            j = exp.est_names.index(est_name)
            result.extend(np.nanmean(exp.prediction_r2_[:, :, 0, j], axis=0).tolist())
        return result

    def _time(exps, est_name):
        result = []
        for exp in _as_list(exps):
            j = exp.est_names.index(est_name)
            result.extend(np.nanmean(exp.fitting_time_[:, :, 0, j], axis=0).tolist())
        return result

    experiments = [exp_d1, exp_d2, exp_d3]
    cv_rows = [('CV_glm', 'CV GLM Grid'), ('CV_fix', 'CV Fixed Grid')]
    CLIP_MIN = -0.1
    PAD      =  0.03

    all_su = [
        t_cv / t_em
        for exps in experiments
        for cv, _ in cv_rows
        for t_cv, t_em in zip(_time(exps, cv), _time(exps, 'EM'))
    ]
    norm = mcolors.Normalize(vmin=min(all_su), vmax=max(all_su))
    cmap = plt.cm.Greens

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.11, top=0.93,
                        hspace=0.06, wspace=0.04)

    for col, exps in enumerate(experiments):
        for row, (cv_name, cv_label) in enumerate(cv_rows):
            ax = axes[row, col]

            true_em = _r2(exps, 'EM')
            true_cv = _r2(exps, cv_name)
            su      = [t_cv / t_em
                       for t_cv, t_em in zip(_time(exps, cv_name), _time(exps, 'EM'))]
            disp_em = [max(CLIP_MIN, x) for x in true_em]
            disp_cv = [max(CLIP_MIN, y) for y in true_cv]
            clipped = [e < CLIP_MIN or c < CLIP_MIN
                       for e, c in zip(true_em, true_cv)]
            colors  = cmap(norm(np.array(su)))

            idx_in  = [i for i, cl in enumerate(clipped) if not cl]
            idx_out = [i for i, cl in enumerate(clipped) if cl]

            if idx_in:
                ax.scatter([disp_em[i] for i in idx_in],
                           [disp_cv[i] for i in idx_in],
                           c=[colors[i] for i in idx_in],
                           s=50, zorder=3, edgecolors='k', linewidths=0.6)
            if idx_out:
                sc = ax.scatter([disp_em[i] for i in idx_out],
                                [disp_cv[i] for i in idx_out],
                                c=[colors[i] for i in idx_out],
                                s=50, zorder=4, edgecolors='k', linewidths=0.8)
                sc.set_linestyle('--')

            ax.plot([CLIP_MIN, 1], [CLIP_MIN, 1], 'k--', lw=0.8, zorder=2)
            ax.axhline(0, color='0.8', lw=0.5, zorder=1)
            ax.axvline(0, color='0.8', lw=0.5, zorder=1)
            ax.set_xlim(CLIP_MIN - PAD, 1 + PAD)
            ax.set_ylim(CLIP_MIN - PAD, 1 + PAD)

            if row == 1:
                ax.set_xlabel('BayesEM $R^2$')
                ax.set_xticks([0.0, 0.5, 1.0])
            if col == 0:
                ax.set_ylabel(f'{cv_label} $R^2$')
                ax.set_yticks([0.0, 0.5, 1.0])
            if row == 0:
                ax.set_title(f'$d = {col + 1}$')

    cbar_ax = fig.add_axes([0.86, 0.28, 0.02, 0.46])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label('speed-up ratio', rotation=90, labelpad=-30)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
```

- [ ] **Step 9: Update cell `0fccfcde` — `make_figure3` call**

Replace source with:

```python
make_figure3(exp, exp_d2, exp_d3)
```

- [ ] **Step 10: User review checkpoint — do not proceed until approved**

The user will run the updated preview cells in JupyterLab themselves (not VS Code — extension causes write conflicts). Wait for the user to confirm the output looks correct before continuing to Step 11.

- [ ] **Step 11: Run nbmake on `real_data.ipynb` to verify CI passes**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest --nbmake experiments/real_data.ipynb -v 2>&1 | tail -20
```

Expected: PASSED. The d=2 SVD failure (forest) is now handled gracefully and should emit a warning rather than crashing.

- [ ] **Step 12: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: update real_data.ipynb preview cells to use EmpiricalDataExperiment"
```

---

## Task 3: Update `real_data.ipynb` skip-execution cells

**Files:** `experiments/real_data.ipynb`

These cells are skipped by CI but must be updated for consistency. `stat_mean` is available from cell `6b69df71` (defined in the preview section and persists in the kernel session). Cell `65e15066` defines the updated `make_figure3`.

- [ ] **Step 1: Update cell `2bfbd407` — full experiment d=1 run**

Replace only the last three lines (the `estimators_full` definition and run call), keeping the `problems_full` list unchanged. The updated tail of the cell:

```python
estimators_full = {
    'EM':     RidgeEM(),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10)),
    'CV_glm': RidgeLOOCV(alphas=100),
}

exp_full = EmpiricalDataExperiment(
    problems_full, list(estimators_full.values()),
    n_iterations=100, seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

- [ ] **Step 2: Update cell `bf1e6b1e` — full experiment d=1 table**

Replace source with:

```python
rows_full = []
for i, problem in enumerate(exp_full.problems):
    em_time = stat_mean(exp_full, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_full, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_full, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_full, 'prediction_r2', est, i) for est in exp_full.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_full, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_full.ns[i, 0])
    row['n:p']            = int(exp_full.ns[i, 0]) / stat_mean(exp_full, 'number_of_features', 'EM', i)
    rows_full.append(row)
pd.DataFrame(rows_full).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 3: Update cell `28f3974b` — full experiment d=2 run**

Replace source with:

```python
exp_full_d2 = EmpiricalDataExperiment(
    problems_full_d2, list(estimators_full.values()),
    n_iterations=30, seed=123, polynomial=2,
    est_names=list(estimators_full.keys())).run()
print()
```

Note: `problems_full_d2` is defined in cell `b12d0253` (unchanged, no skip tag).

- [ ] **Step 4: Update cell `f54e9b3d` — full experiment d=2 table**

Replace source with:

```python
rows_full_d2 = []
for i, problem in enumerate(exp_full_d2.problems):
    em_time = stat_mean(exp_full_d2, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_full_d2, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_full_d2, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_full_d2, 'prediction_r2', est, i) for est in exp_full_d2.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_full_d2, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_full_d2.ns[i, 0])
    row['n:p']            = int(exp_full_d2.ns[i, 0]) / stat_mean(exp_full_d2, 'number_of_features', 'EM', i)
    rows_full_d2.append(row)
pd.DataFrame(rows_full_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 5: Update cell `a96f98ee` — full experiment d=3 run**

Replace source with:

```python
exp_full_d3 = EmpiricalDataExperiment(
    problems_full_d3, list(estimators_full.values()),
    n_iterations=30, seed=123, polynomial=3,
    est_names=list(estimators_full.keys())).run()
print()
```

Note: `problems_full_d3` is defined in cell `d245ce96` (unchanged, no skip tag).

- [ ] **Step 6: Update cell `8ae40591` — full experiment d=3 table**

Replace source with:

```python
rows_full_d3 = []
for i, problem in enumerate(exp_full_d3.problems):
    em_time = stat_mean(exp_full_d3, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_full_d3, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_full_d3, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_full_d3, 'prediction_r2', est, i) for est in exp_full_d3.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_full_d3, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_full_d3.ns[i, 0])
    row['n:p']            = int(exp_full_d3.ns[i, 0]) / stat_mean(exp_full_d3, 'number_of_features', 'EM', i)
    rows_full_d3.append(row)
pd.DataFrame(rows_full_d3).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 7: Update cell `07437ade` — full experiment figure call**

Replace source with:

```python
make_figure3(exp_full, exp_full_d2, exp_full_d3,
             output_path='../output/paper2023_figure3.pdf')
```

- [ ] **Step 8: Update cell `68c60ed3` — large datasets d=1 run**

Replace source with:

```python
exp_large = EmpiricalDataExperiment(
    problems_large, list(estimators_full.values()),
    n_iterations=30, seed=123,
    est_names=list(estimators_full.keys())).run()
print()
```

Note: `problems_large` is defined in cell `502c1187` (unchanged, skip-execution).

- [ ] **Step 9: Update cell `1f3885a0` — large datasets d=1 table**

Replace source with:

```python
rows_large = []
for i, problem in enumerate(exp_large.problems):
    em_time = stat_mean(exp_large, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_large, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_large, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_large, 'prediction_r2', est, i) for est in exp_large.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_large, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_large.ns[i, 0])
    row['n:p']            = int(exp_large.ns[i, 0]) / stat_mean(exp_large, 'number_of_features', 'EM', i)
    rows_large.append(row)
pd.DataFrame(rows_large).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 10: Update cell `300ca33b` — large datasets d=2 run**

Replace source with:

```python
exp_large_d2 = EmpiricalDataExperiment(
    problems_large_d2, list(estimators_full.values()),
    n_iterations=30, seed=123, polynomial=2,
    est_names=list(estimators_full.keys())).run()
print()
```

Note: `problems_large_d2` is defined in cell `e4ee47eb` (unchanged, skip-execution).

- [ ] **Step 11: Update cell `522384c3` — large datasets d=2 table**

Replace source with:

```python
rows_large_d2 = []
for i, problem in enumerate(exp_large_d2.problems):
    em_time = stat_mean(exp_large_d2, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_large_d2, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_large_d2, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: stat_mean(exp_large_d2, 'prediction_r2', est, i) for est in exp_large_d2.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_large_d2, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_large_d2.ns[i, 0])
    row['n:p']            = int(exp_large_d2.ns[i, 0]) / stat_mean(exp_large_d2, 'number_of_features', 'EM', i)
    rows_large_d2.append(row)
pd.DataFrame(rows_large_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 12: Update cell `f36b7874` — combined figure call**

This call previously concatenated two lists of dicts (`results_full + results_large`). With `make_figure3` now accepting a list of experiments, pass lists instead:

```python
make_figure3([exp_full, exp_large],
             [exp_full_d2, exp_large_d2],
             exp_full_d3,
             output_path='../output/realdata_r2_by_degree.pdf')
```

- [ ] **Step 13: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: update real_data.ipynb full-experiment cells to use EmpiricalDataExperiment"
```

---

## Task 4: Update `real_data_neurips2023.ipynb`

**Files:** `experiments/real_data_neurips2023.ipynb`

All cells in this notebook are skip-execution; this task is for consistency. The `stat_mean` helper and `make_figure3` must be updated with the same pattern used in `real_data.ipynb`.

- [ ] **Step 1: Read `real_data_neurips2023.ipynb` to confirm cell IDs**

Use the Read tool. Confirm:
- cell id `a001` — markdown header (mentions "Figure 2" — needs fixing to "Figure 3")
- cell id `a003` — imports + d=1 run
- cell id `a004` — d=1 table
- cell id `a006` — d=2 run
- cell id `a007` — d=2 table
- cell id `a009` — d=3 run
- cell id `a010` — d=3 table
- cell id `a013` — `make_figure3` definition + call

- [ ] **Step 2: Update cell `a001` — fix figure number in markdown header**

Replace "Figure 2" with "Figure 3" in the markdown source of cell `a001`. The corrected first sentence should read:

```
Reproduces Table 2 and Figure 3 of the NeurIPS 2023 paper comparing:
```

- [ ] **Step 3: Update cell `a003` — imports + d=1 run**

Replace source with:

```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from experiments import EmpiricalDataExperiment
from problems import NEURIPS2023
from data import DATASETS

estimators = {
    'EM':     RidgeEM(t2=False),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
    'CV_glm': RidgeLOOCV(alphas=100),
}

problems_d1 = sorted(NEURIPS2023, key=lambda p: DATASETS[p.dataset]['n'])
exp_d1 = EmpiricalDataExperiment(
    problems_d1, list(estimators.values()),
    n_iterations=100, seed=123,
    est_names=list(estimators.keys()), verbose=True).run()
print()
```

- [ ] **Step 4: Update cell `a004` — d=1 table**

Replace source with:

```python
import pandas as pd

def stat_mean(exp, stat_name, est_name, problem_idx):
    j = exp.est_names.index(est_name)
    return np.nanmean(getattr(exp, stat_name + '_')[:, problem_idx, 0, j])

rows_d1 = []
for i, problem in enumerate(exp_d1.problems):
    em_time = stat_mean(exp_d1, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_d1, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_d1, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target, 'd': 1}
    row.update({est: stat_mean(exp_d1, 'prediction_r2', est, i) for est in exp_d1.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_d1, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_d1.ns[i, 0])
    row['n:p']            = int(exp_d1.ns[i, 0]) / stat_mean(exp_d1, 'number_of_features', 'EM', i)
    rows_d1.append(row)
df_d1 = pd.DataFrame(rows_d1).sort_values('n_train', ascending=False).round(2)
df_d1
```

- [ ] **Step 5: Update cell `a006` — d=2 run**

Replace source with:

```python
from problems import NEURIPS2023_D2

problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
exp_d2 = EmpiricalDataExperiment(
    problems_d2, list(estimators.values()),
    n_iterations=100, seed=123, polynomial=2,
    est_names=list(estimators.keys()), verbose=True).run()
print()
```

- [ ] **Step 6: Update cell `a007` — d=2 table**

Replace source with:

```python
rows_d2 = []
for i, problem in enumerate(exp_d2.problems):
    em_time = stat_mean(exp_d2, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_d2, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_d2, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target, 'd': 2}
    row.update({est: stat_mean(exp_d2, 'prediction_r2', est, i) for est in exp_d2.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_d2, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_d2.ns[i, 0])
    row['n:p']            = int(exp_d2.ns[i, 0]) / stat_mean(exp_d2, 'number_of_features', 'EM', i)
    rows_d2.append(row)
df_d2 = pd.DataFrame(rows_d2).sort_values('n_train', ascending=False).round(2)
df_d2
```

- [ ] **Step 7: Update cell `a009` — d=3 run**

Replace source with:

```python
from problems import NEURIPS2023_D3

problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
exp_d3 = EmpiricalDataExperiment(
    problems_d3, list(estimators.values()),
    n_iterations=100, seed=123, polynomial=3,
    est_names=list(estimators.keys()), verbose=True).run()
print()
```

- [ ] **Step 8: Update cell `a010` — d=3 table**

Replace source with:

```python
rows_d3 = []
for i, problem in enumerate(exp_d3.problems):
    em_time = stat_mean(exp_d3, 'fitting_time', 'EM', i)
    cv_time = (stat_mean(exp_d3, 'fitting_time', 'CV_glm', i)
               + stat_mean(exp_d3, 'fitting_time', 'CV_fix', i)) / 2
    row = {'dataset': problem.dataset, 'target': problem.target, 'd': 3}
    row.update({est: stat_mean(exp_d3, 'prediction_r2', est, i) for est in exp_d3.est_names})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = stat_mean(exp_d3, 'number_of_features', 'EM', i)
    row['n_train']        = int(exp_d3.ns[i, 0])
    row['n:p']            = int(exp_d3.ns[i, 0]) / stat_mean(exp_d3, 'number_of_features', 'EM', i)
    rows_d3.append(row)
df_d3 = pd.DataFrame(rows_d3).sort_values('n_train', ascending=False).round(2)
df_d3
```

- [ ] **Step 9: Update cell `a013` — `make_figure3` definition + call**

Replace source with:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_figure3(exp_d1, exp_d2, exp_d3, output_path=None):
    """2x3 scatter of EM vs CV R² for d=1,2,3. Color = speed-up ratio.

    Top row: CV_glm on y-axis. Bottom row: CV_fix on y-axis.
    Values below CLIP_MIN=-0.1 are clipped to CLIP_MIN and shown with a dashed edge.
    Axis limits extend PAD beyond [CLIP_MIN, 1] so boundary points are fully visible.
    """
    def _r2(exp, est_name):
        j = exp.est_names.index(est_name)
        return np.nanmean(exp.prediction_r2_[:, :, 0, j], axis=0).tolist()

    def _time(exp, est_name):
        j = exp.est_names.index(est_name)
        return np.nanmean(exp.fitting_time_[:, :, 0, j], axis=0).tolist()

    experiments = [exp_d1, exp_d2, exp_d3]
    cv_rows = [('CV_glm', 'CV GLM Grid'), ('CV_fix', 'CV Fixed Grid')]
    CLIP_MIN = -0.1
    PAD      =  0.03

    all_su = [
        t_cv / t_em
        for exp in experiments
        for cv, _ in cv_rows
        for t_cv, t_em in zip(_time(exp, cv), _time(exp, 'EM'))
    ]
    norm = mcolors.Normalize(vmin=min(all_su), vmax=max(all_su))
    cmap = plt.cm.Greens

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.11, top=0.93,
                        hspace=0.06, wspace=0.04)

    for col, exp in enumerate(experiments):
        for row, (cv_name, cv_label) in enumerate(cv_rows):
            ax = axes[row, col]

            true_em = _r2(exp, 'EM')
            true_cv = _r2(exp, cv_name)
            su      = [t_cv / t_em
                       for t_cv, t_em in zip(_time(exp, cv_name), _time(exp, 'EM'))]
            disp_em = [max(CLIP_MIN, x) for x in true_em]
            disp_cv = [max(CLIP_MIN, y) for y in true_cv]
            clipped = [e < CLIP_MIN or c < CLIP_MIN
                       for e, c in zip(true_em, true_cv)]
            colors  = cmap(norm(np.array(su)))

            idx_in  = [i for i, cl in enumerate(clipped) if not cl]
            idx_out = [i for i, cl in enumerate(clipped) if cl]

            if idx_in:
                ax.scatter([disp_em[i] for i in idx_in],
                           [disp_cv[i] for i in idx_in],
                           c=[colors[i] for i in idx_in],
                           s=50, zorder=3, edgecolors='k', linewidths=0.6)
            if idx_out:
                sc = ax.scatter([disp_em[i] for i in idx_out],
                                [disp_cv[i] for i in idx_out],
                                c=[colors[i] for i in idx_out],
                                s=50, zorder=4, edgecolors='k', linewidths=0.8)
                sc.set_linestyle('--')

            ax.plot([CLIP_MIN, 1], [CLIP_MIN, 1], 'k--', lw=0.8, zorder=2)
            ax.axhline(0, color='0.8', lw=0.5, zorder=1)
            ax.axvline(0, color='0.8', lw=0.5, zorder=1)
            ax.set_xlim(CLIP_MIN - PAD, 1 + PAD)
            ax.set_ylim(CLIP_MIN - PAD, 1 + PAD)

            if row == 1:
                ax.set_xlabel('BayesEM $R^2$')
                ax.set_xticks([0.0, 0.5, 1.0])
            if col == 0:
                ax.set_ylabel(f'{cv_label} $R^2$')
                ax.set_yticks([0.0, 0.5, 1.0])
            if row == 0:
                ax.set_title(f'$d = {col + 1}$')

    cbar_ax = fig.add_axes([0.86, 0.28, 0.02, 0.46])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label('speed-up ratio', rotation=90, labelpad=-30)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')

make_figure3(exp_d1, exp_d2, exp_d3,
             output_path='../output/paper2023_figure3.pdf')
```

- [ ] **Step 10: Commit**

```bash
git add experiments/real_data_neurips2023.ipynb
git commit -m "feat: update real_data_neurips2023.ipynb to use EmpiricalDataExperiment"
```
