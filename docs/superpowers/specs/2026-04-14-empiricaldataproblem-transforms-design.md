# EmpiricalDataProblem Column Transforms Design

## Goal

Allow `EmpiricalDataProblem` to apply simple ufunc-style transforms to individual columns before returning from `get_X_y()`. The immediate use cases are log and log1p target transforms needed to reproduce paper Table 2 results (yacht, forest, automobile).

## Interface

Add a `transforms` parameter to `EmpiricalDataProblem.__init__`: an ordered list of `(column_name, callable)` pairs.

```python
EmpiricalDataProblem('yacht', 'Residuary_resistance',
                     transforms=[('Residuary_resistance', np.log)])

EmpiricalDataProblem('forest', 'area',
                     transforms=[('area', np.log1p)])

EmpiricalDataProblem('automobile', 'price', nan_policy='drop_rows',
                     transforms=[('price', np.log)])
```

- `column_name` is a literal DataFrame column name — no special aliases.
- `callable` is any function mapping a `pd.Series` to a `pd.Series` of the same length (numpy ufuncs satisfy this).
- Transforms are applied in list order, enabling future composition (e.g. shift then log).
- If a column in `transforms` is absent from the DataFrame after dropping and NaN handling, raise `ValueError`.

## Pipeline position

Transforms are applied inside `get_X_y()` after NaN handling and column dropping, before splitting into X and y. This ensures transforms operate on clean data and that the target column is still present when its transform is applied.

```
get_dataset() → drop columns → dropna(target) → nan_policy → apply transforms → split X / y
```

## Files

- Modify: `experiments/problems.py` — add `transforms` parameter, apply in `get_X_y()`

## Testing

Add doctests to `problems.py` verifying:
- `transforms=[]` (default) produces identical output to no `transforms` argument
- `transforms=[('target', np.log)]` on the diabetes dataset (target column is literally named `'target'`) produces `log`-transformed y values
- A misspelled column name raises `ValueError`

## Future extensibility

- **Multi-column transforms:** widen the first element of each tuple to a list of column names and pass a sub-DataFrame to the callable.
- **Dataset-level designated targets:** declare default target columns in `data.py`'s `DATASETS` registry. `EmpiricalDataProblem` could then omit the `target` argument and infer it. This is independent of the transform mechanism.
- **`tasks.py` named collections:** once transforms are expressible in `EmpiricalDataProblem`, named task collections (e.g. `PAPER_TASKS`) can encode the full preprocessing including transforms without duplication across notebooks.
