---
name: EmpiricalDataProblem
description: Design for EmpiricalDataProblem class and integration into run_real_data_experiments and real_data.ipynb
type: project
---

# EmpiricalDataProblem Design

## Background

The real data module (`experiments/data.py`, `experiments/real_data.ipynb`) established a two-tier architecture: tier 1 handles raw dataset retrieval (`get_dataset`); tier 2 (task assembly — target selection, drop lists, encoding, train/test split) was deferred to a future project.

This project implements tier 2 via a new `EmpiricalDataProblem` class. Two datasets require special handling: `naval_propulsion` has two target variables (`GT_compressor_decay`, `GT_turbine_decay`) and `parkinsons` has two target variables (`motor_UPDRS`, `total_UPDRS`). Each target becomes a separate problem, with the other target moved into the drop list to prevent leakage into features.

## Goal

Introduce `EmpiricalDataProblem` in `experiments/problems.py`, update `run_real_data_experiments` to accept a list of problems, expand the notebook task spec to cover both targets for naval and parkinsons, and add a `target` column to the output table.

---

## Architecture

### `EmpiricalDataProblem` (`experiments/problems.py`)

Plain class alongside `linear_problem`. PascalCase — consistent with Python standard and the rest of the project (net gain away from the `linear_problem` naming exception).

```python
class EmpiricalDataProblem:
    def __init__(self, dataset, target, drop=None):
        self.dataset = dataset
        self.target = target
        self.drop = drop or []

    def get_X_y(self):
        df = get_dataset(self.dataset)
        df = df.drop(columns=self.drop)
        return df.drop(columns=[self.target]), df[self.target]
```

- `dataset`: registry key passed to `get_dataset`
- `target`: name of the response column
- `drop`: list of columns to remove before splitting X and y (used to exclude non-target response columns)
- `get_X_y()`: the empirical analogue of `linear_problem.rvs()` — intended unification point in a future refactor

No sampling logic. Preprocessing (one-hot encoding, polynomial features, NaN handling) remains in `run_real_data_experiments` for now and migrates incrementally.

### `run_real_data_experiments` (`experiments/experiments.py`)

Signature change: `dataframes, targets, names` replaced by `problems: list[EmpiricalDataProblem]`.

- Returns a **list** parallel to `problems` (integer-indexed) — consistent with `Experiment`'s positional result arrays, and required because the same dataset can appear multiple times with different targets
- Internally calls `problem.get_X_y()` instead of `df.drop([target])`
- Uses `problem.dataset` for verbose progress output
- `problem.target` is available for table assembly in the notebook

### Results representation

Results are a list of per-problem dicts (one entry per estimator per problem). Integer index. No string key. The notebook is responsible for assembling display tables; the experiment function is not.

Future: an experiment hash fully characterising `(dataset, target, drop, preprocessing, seed, ...)` would serve as a stable key. Out of scope here.

---

## Notebook task spec (`experiments/real_data.ipynb`)

Task list becomes a list of `EmpiricalDataProblem` instances. Naval and parkinsons are expanded from one entry to two:

```python
problems = [
    EmpiricalDataProblem('abalone',          'Rings'),
    EmpiricalDataProblem('airfoil',          'scaled-sound-pressure'),
    EmpiricalDataProblem('concrete',         'Concrete compressive strength'),
    EmpiricalDataProblem('diabetes',         'target'),
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

Table assembly zips `problems` and `results`. `problem.target` added as a column. Integer index used internally; the presentation table for the paper is a separate formatting concern.

---

## `experiments/data.py`

No changes. `DATASETS` remains sources-only. Target and drop are task-level concerns.

---

## Review Note — DataFrame vs numpy in fastridge classes

`get_X_y()` returns a DataFrame `X` and Series `y`. sklearn estimators strip the pandas layer early via `check_array`, so there is no performance concern there. However, `RidgeEM` and `RidgeLOOCV` in `fastridge.py` do not go through sklearn's validation pipeline and likely receive the DataFrame directly. If numpy operations are performed on a DataFrame rather than a numpy array, there can be significant overhead. This should be reviewed — and if needed, an explicit `.to_numpy()` call added in `get_X_y()` or at the entry point of the fastridge classes — but is deferred to after this integration is running.

---

## Out of Scope

- Sampling design abstraction (bootstrap, CV, repeated subsampling) — future unification with `Experiment`
- Experiment hash / stable result keys
- NaN handling — design decision deferred (see real data module memory)
- Refactoring `linear_problem` to PascalCase
- DataFrame vs numpy performance review for fastridge classes — deferred, noted above
