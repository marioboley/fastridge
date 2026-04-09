---
name: Tutorial Behaviour Fix
description: Restore tutorial.ipynb figure output to match original notebook; commit conditional LaTeX detection
type: project
---

# Tutorial Behaviour Fix Design

## Background

During the `unimodality_convexity` integration project, two unapproved behaviour changes were introduced and committed to `dev`/`main`:

**Change 1 (more severe): removal of in-place normalization from `tutorial.ipynb` cell 4.**
The original notebook cell 4 pre-normalized `x_train`/`y_train` in-place before passing them to any estimator. The tutorial cell 4 was committed without this normalization. Since `x_train`/`y_train` are referenced as session-level variables in cells 7 and 12 (and optionally 15), those cells received raw data where the original received normalized data. This affects `RidgeLOOCV.fit`, `RidgeEM.fit`, and the EM contour plot in cell 12, in addition to cell 4's own figure.

**Change 2: test data normalization in `RidgePathExperiment`.**
`RidgeTrueRisk.fit` normalized `x_test`/`y_test` by their **own** stats; `RidgePathExperiment.run()` normalizes them by **train** stats. This changes the scale and shape of `true_risk_` and affects figures 1, 2, and 4. The original behaviour (normalize test by test stats) is correct for oracle evaluation: `true_risk_` values are dimensionless relative to the test distribution.

Additionally, two uncommitted changes are sitting in the working tree that are correct and should be committed as part of this fix:
- `tutorial.ipynb` cell 4: pre-normalization of `x_train`/`y_train` in-place (restoring original behaviour)
- `plotting.py` / `plotting2d.py`: conditional LaTeX detection via `shutil.which('latex')` — the only approved behaviour change relative to the original

## Goal

Restore all tutorial figures to match the original notebook's output, with the single approved exception of conditional LaTeX rendering.

## Behaviour Audit

### RidgePathExperiment vs RidgeTrueRisk — test normalization

**Original `RidgeTrueRisk.fit(x_train, y_train, x_test, y_test)`** — two symmetric blocks, each four lines:
```python
# train block
a_x, a_y = (x.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
b_x, b_y = (x.std(axis=0), y.std()) if self.normalize else (np.ones(p), 1.0)
x = (x - a_x) / b_x
y = (y - a_y) / b_y

# test block — own stats
a_x_test, a_y_test = (x_test.mean(axis=0), y_test.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
b_x_test, b_y_test = (x_test.std(axis=0), y_test.std()) if self.normalize else (np.ones(p), 1.0)
x_test = (x_test - a_x_test) / b_x_test
y_test = (y_test - a_y_test) / b_y_test
```

**Current (wrong) `RidgePathExperiment.run()`** — test normalized by train stats:
```python
x_te = (self.x_test - a_x) / b_x   # a_x, b_x are train stats — wrong
y_te = (self.y_test - a_y) / b_y   # a_y, b_y are train stats — wrong
```

This is the root cause of the figure difference. Normalizing test data by its own stats is correct for oracle evaluation: `true_risk_` values are dimensionless (unit variance), reflecting the true predictive risk relative to the test distribution rather than the train distribution.

### Cell 4: original vs tutorial (committed)

**Original notebook cell 4:**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)
a_x, a_y = (x_train.mean(axis=0), y_train.mean())
b_x, b_y = (x_train.std(axis=0), y_train.std())
x_train = (x_train - a_x)/b_x      # in-place normalization
y_train = (y_train - a_y)/b_y      # in-place normalization
# x_test, y_test passed raw to RidgeTrueRisk
ridgeCV_test = RidgeTrueRisk(alphas=alphas, fit_intercept=True, normalize=True)
ridgeCV_test.fit(x_train, y_train, x_test, y_test)
```

**Tutorial cell 4 (committed):**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)
# no pre-normalization — missing
path_exp = RidgePathExperiment(x_train, y_train, x_test, y_test, alphas,
                               fit_intercept=True, normalize=True).run()
```

The uncommitted working tree version adds pre-normalization of `x_train`/`y_train` — this is correct and must be committed. Note that the double-normalization (pre-normalize externally, then `RidgePathExperiment` normalizes train again as a near-no-op internally) matches the original pattern and should be preserved for now. It is flagged as a design smell for future cleanup.

### Cell 7: no structural difference

Cell 7 uses `x_train`/`y_train` as set by cell 4 (pre-normalized in both original and tutorial after the fix), and `x_test`/`y_test` raw. `RidgeLOOCV.fit` and `RidgeEM.fit` receive the same pre-normalized data. `RidgePathExperiment`'s test normalization bug affects `true_risk_` here too — same fix applies.

### Cell 15: original vs tutorial

**Original cell 15 loop iteration:**
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=s, shuffle=True, random_state=180)
a_x, a_y = (x_train.mean(axis=0), y_train.mean())
b_x, b_y = (x_train.std(axis=0), y_train.std())
x_train = (x_train - a_x)/b_x      # in-place normalize train
y_train = (y_train - a_y)/b_y
# x_test raw
ridgeCV_test = RidgeTrueRisk(alphas=alphas, fit_intercept=True, normalize=True)
ridgeCV_test.fit(x_train, y_train, x_test, y_test)
ridgeCV_test.alphas_ = 1/ridgeCV_test.alphas_   # flip to τ² scale
plot_marg_profile(x_train, y_train, 1/alphas, ...)  # receives normalized train data
```

**Tutorial cell 15 loop iteration:**
```python
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(...)
a_x, b_x, a_y, b_y = ...  # train stats
x_tr_norm = (x_train_s - a_x) / b_x   # explicit local normalization (not in-place)
y_tr_norm = (y_train_s - a_y) / b_y
path_s = RidgePathExperiment(x_train_s, y_train_s, x_test_s, y_test_s, alphas_s,
                             fit_intercept=True, normalize=True).run()
t2_grid = 1 / path_s.alphas_          # τ² conversion
plot_marg_profile(x_tr_norm, y_tr_norm, t2_grid, ...)  # receives normalized train data ✓
```

The tutorial cell 15 correctly passes normalized train data to `plot_marg_profile` — matching the original. The τ² axis conversion is also equivalent (`ridgeCV_test.alphas_ = 1/ridgeCV_test.alphas_` in original vs `t2_grid = 1/path_s.alphas_` in tutorial).

However, the committed tutorial cell 15 passed raw `x_train_s`/`y_train_s` to `RidgePathExperiment`, while the original passed pre-normalized `x_train`/`y_train` (after in-place normalization). This is a third unapproved behaviour change: `x_tr_norm`/`y_tr_norm` must be passed to `RidgePathExperiment` instead.

Note: the tutorial uses local variables (`x_train_s`, `x_tr_norm`) rather than overwriting the session-level `x_train` in-place. This is a deliberate improvement — the original's in-place overwrite in the loop would shadow the `x_train` from cell 4, which is a side effect not present in the tutorial. This is an intentional (and approved) improvement, not a bug.

## Changes Required

### 1. Fix `experiments/experiments.py` — `RidgePathExperiment.run()`

Replace train-stats-based test normalization with test-stats-based:

```python
# replace:
x_te = (self.x_test - a_x) / b_x
y_te = (self.y_test - a_y) / b_y

# with — mirroring the two-block structure of RidgeTrueRisk:
a_x_te, a_y_te = (self.x_test.mean(axis=0), self.y_test.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
b_x_te, b_y_te = (self.x_test.std(axis=0), self.y_test.std()) if self.normalize else (np.ones(p), 1.0)
x_te = (self.x_test - a_x_te) / b_x_te
y_te = (self.y_test - a_y_te) / b_y_te
```

### 2. Stage uncommitted change to `tutorial.ipynb` — cell 4

Add pre-normalization of `x_train`/`y_train`:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)

a_x, a_y = x_train.mean(axis=0), y_train.mean()
b_x, b_y = x_train.std(axis=0), y_train.std()
x_train = (x_train - a_x) / b_x
y_train = (y_train - a_y) / b_y

alphas = np.logspace(-5, 5, 400, endpoint=True, base=10)

path_exp = RidgePathExperiment(x_train, y_train, x_test, y_test, alphas,
                               fit_intercept=True, normalize=True).run()

plot_pathway_risk(path_exp, variable_names=[f.upper() for f in diabetes.feature_names],
                  best_lambda=True, dpi=100, save_file='../output/tutorial_figure1.pdf')
```

### 3. Fix `tutorial.ipynb` cell 15 — pass pre-normalized train data to RidgePathExperiment

Change:
```python
    path_s = RidgePathExperiment(x_train_s, y_train_s, x_test_s, y_test_s, alphas_s,
                                 fit_intercept=True, normalize=True).run()
```
to:
```python
    path_s = RidgePathExperiment(x_tr_norm, y_tr_norm, x_test_s, y_test_s, alphas_s,
                                 fit_intercept=True, normalize=True).run()
```

### 4. Stage uncommitted changes to `plotting.py` and `plotting2d.py` — conditional LaTeX

Add at module level (already in working tree):
```python
import shutil
matplotlib.rcParams['text.usetex'] = shutil.which('latex') is not None
```
Remove all `plt.rcParams['text.usetex'] = True` calls from individual functions.

## Out of Scope

- Cleaning up the double-normalization pattern (external pre-normalize + internal near-no-op) — deferred to a future plotting/experiment refactoring project
- Any other changes to `tutorial.ipynb` cells
- Changes to `fastridge.py`, `problems.py`

## Future Note

The double normalization in cells 4 and 7 (pre-normalize externally, `RidgePathExperiment` normalizes train again internally as a near-no-op) is a design smell inherited from the original `RidgeTrueRisk` pattern. A future refactoring should determine what normalization is required, where it should be applied (class vs notebook), and how train and test normalization relate — including documenting that test normalization by test stats is intentional oracle behaviour.

Additionally, cell 15's normalization pattern is suspicious: `x_tr_norm`/`y_tr_norm` are passed to `RidgePathExperiment` as training data (pre-normalized externally), while `x_test_s`/`y_test_s` are passed raw (to be normalized internally by test stats). This asymmetry — normalized train, raw test — is inherited from the original and preserved here for behaviour fidelity. However, it is unclear whether this is intentional: in cells 4 and 7, the same asymmetry arises naturally from the session state (cell 4 pre-normalizes `x_train` in-place, leaving `x_test` raw). In cell 15, it arises from a deliberate local variable choice. Whether the oracle risk should be evaluated on test data normalized by test stats — independent of what was done to train — is a design question left for future analysis.
