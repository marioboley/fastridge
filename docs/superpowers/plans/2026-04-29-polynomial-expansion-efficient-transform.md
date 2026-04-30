# Polynomial Expansion Efficient Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sklearn's `PolynomialFeatures.fit_transform` in `PolynomialExpansion.__call__` with a direct itertools + numpy implementation that only materialises the P selected feature columns, achieving O(nP) instead of O(n * C(p+d, d)).

**Architecture:** A `@staticmethod _feature_name` on `PolynomialExpansion` derives sklearn-compatible column names from combination tuples. `PolynomialExpansion.__call__` enumerates interaction combinations as integer tuples (cheap), applies the budget/subsampling logic, then computes only the selected columns with numpy element-wise products. The sklearn import is removed.

**Tech Stack:** Python, NumPy, pandas, itertools (stdlib)

**Spec:** `docs/superpowers/specs/2026-04-29-polynomial-expansion-efficient-transform.md`

---

### Task 1: Replace fit_transform with itertools + numpy bypass

**Files:**
- Modify: `experiments/problems.py` (lines 1–13, 199–255)
- Test: `tests/test_problems.py`

---

- [ ] **Step 1: Write tests (all pass before implementation; serve as regression guard after)**

Add to `tests/test_problems.py`:

```python
def test_polynomial_expansion_matches_sklearn_degree2():
    from sklearn.preprocessing import PolynomialFeatures
    X = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    result = PolynomialExpansion(2)(X, np.random.default_rng(0))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    expected = pd.DataFrame(
        poly.fit_transform(X),
        columns=poly.get_feature_names_out(X.columns),
        index=X.index
    )
    pd.testing.assert_frame_equal(result, expected)


def test_polynomial_expansion_matches_sklearn_degree3():
    from sklearn.preprocessing import PolynomialFeatures
    X = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0], 'c': [5.0, 6.0]})
    result = PolynomialExpansion(3)(X, np.random.default_rng(0))
    poly = PolynomialFeatures(degree=3, include_bias=False)
    expected = pd.DataFrame(
        poly.fit_transform(X),
        columns=poly.get_feature_names_out(X.columns),
        index=X.index
    )
    pd.testing.assert_frame_equal(result, expected)


def test_polynomial_expansion_feature_values_correct():
    X = pd.DataFrame({'a': [2.0, 3.0], 'b': [4.0, 5.0]})
    result = PolynomialExpansion(2)(X, np.random.default_rng(0))
    np.testing.assert_array_equal(result['a^2'].values, [4.0, 9.0])
    np.testing.assert_array_equal(result['a b'].values, [8.0, 15.0])
    np.testing.assert_array_equal(result['b^2'].values, [16.0, 25.0])


def test_polynomial_expansion_subsampled_values_match_sklearn():
    from sklearn.preprocessing import PolynomialFeatures
    X = pd.DataFrame({'a': [2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0]})
    # budget = ceil(12/3) = 4 cols; 2 linear + 2 of 3 interactions
    result = PolynomialExpansion(2, max_entries=12)(X, np.random.default_rng(0))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    full = pd.DataFrame(
        poly.fit_transform(X),
        columns=poly.get_feature_names_out(X.columns),
        index=X.index
    )
    for col in result.columns:
        np.testing.assert_array_almost_equal(result[col].values, full[col].values,
                                             err_msg=f'column {col!r} mismatch')
```

- [ ] **Step 2: Run the tests to confirm they currently pass**

```
pytest tests/test_problems.py::test_polynomial_expansion_matches_sklearn_degree2 tests/test_problems.py::test_polynomial_expansion_matches_sklearn_degree3 tests/test_problems.py::test_polynomial_expansion_feature_values_correct tests/test_problems.py::test_polynomial_expansion_subsampled_values_match_sklearn -v
```

Expected: all PASS (current sklearn implementation is correct; these are our regression guards).

- [ ] **Step 3: Update imports in `experiments/problems.py`**

Replace line 10:
```python
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
```
with:
```python
from itertools import combinations_with_replacement

from sklearn.preprocessing import OneHotEncoder
```

- [ ] **Step 4: Add `_feature_name` static method to `PolynomialExpansion`**

Add before `__call__` inside the `PolynomialExpansion` dataclass:

```python
    @staticmethod
    def _feature_name(col_names, combo):
        """Return sklearn-compatible name for a polynomial combination.

        combo is a tuple of column indices with repetition, sorted ascending.
        Example: (0, 0, 1) with col_names ['a', 'b'] gives 'a^2 b'.

        >>> PolynomialExpansion._feature_name(['a', 'b', 'c'], (0, 0, 1))
        'a^2 b'
        >>> PolynomialExpansion._feature_name(['a', 'b'], (1, 1, 1))
        'b^3'
        """
        counts = {}
        for idx in combo:
            counts[idx] = counts.get(idx, 0) + 1
        return ' '.join(
            f'{col_names[idx]}^{exp}' if exp > 1 else col_names[idx]
            for idx, exp in sorted(counts.items())
        )
```

- [ ] **Step 5: Replace `PolynomialExpansion.__call__`**

Replace the current `__call__` method (lines 240–255) with:

```python
    def __call__(self, X, rng):
        X_arr = np.asarray(X, dtype=float)
        n, p = X_arr.shape
        col_names = list(X.columns)
        all_combos = [
            combo
            for d in range(2, self.degree + 1)
            for combo in combinations_with_replacement(range(p), d)
        ]
        if n * (p + len(all_combos)) > self.max_entries:
            p_budget = int(np.ceil(self.max_entries / n))
            pnew = max(0, min(len(all_combos), p_budget - p))
            sampled = sorted(rng.choice(len(all_combos), size=pnew, replace=False))
            selected_combos = [all_combos[i] for i in sampled]
        else:
            selected_combos = all_combos
        higher_degree_cols = []
        higher_degree_names = []
        for combo in selected_combos:
            col = np.ones(n)
            for idx in combo:
                col *= X_arr[:, idx]
            higher_degree_cols.append(col)
            higher_degree_names.append(self._feature_name(col_names, combo))
        all_cols = [X_arr[:, i] for i in range(p)] + higher_degree_cols
        return pd.DataFrame(
            np.column_stack(all_cols),
            columns=col_names + higher_degree_names,
            index=X.index
        )
```

Also update the `degree` parameter docstring line from
`"Polynomial degree passed to PolynomialFeatures."` to `"Polynomial degree."`.

- [ ] **Step 6: Run all tests**

```
pytest
```

Expected: all pass, including the regression test and existing doctests for `PolynomialExpansion`.

- [ ] **Step 7: Commit**

```bash
git add experiments/problems.py tests/test_problems.py
git commit -m "perf: bypass sklearn PolynomialFeatures in PolynomialExpansion for O(nP) transform"
```
