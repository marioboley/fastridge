# Polynomial Subsampling Budget Discrepancy

## Status

Pre-existing bugs preserved in current `PolynomialExpansion`. Documented here
for future correction; no behaviour change has been made.

---

## Background

When polynomial feature expansion produces a matrix exceeding `max_entries`
total entries, `PolynomialExpansion` subsamples interaction columns to reduce
dimensionality. The NeurIPS 2023 paper describes this as:

> "if the number of transformed predictors exceeded this limit, we uniformly
> sub-sampled the interaction variables to ensure that p∗ ≤ 35000000/(0.7n)"

The paper's intent is that the *training* matrix (`0.7n` rows after a 70/30
split) has at most 35 million entries, i.e. total features `p* ≤ 35M/(0.7n)`.

---

## Two bugs in the legacy experiment runner (both preserved in current code)

The original code used full-dataset `n` as the denominator **and** applied the
result as the count of *interaction columns* to keep:

```python
X_poly = X_poly.drop(X.columns, axis=1)        # interaction cols only
pnew = int(np.ceil(35_000_000 / npoly))         # npoly = n (full dataset)
X = pd.concat([X_linear, X_sampled_interaction], axis=1)
# => total p = p_linear + ceil(35M/n)
```

**Bug 1 — denominator**: uses `n` (full dataset) instead of `n_train`
(training set). The paper bounds the size of the *training* matrix.

**Bug 2 — budget semantics**: interprets the budget as the number of
*interaction columns* to keep. The paper's bound is on *total* features p∗,
so the correct interaction count is `max(0, ceil(35M/n_train) - p_linear)`.

Both bugs must be fixed together to reproduce the paper's results. Fixing only
Bug 1 gives `pnew_interaction = ceil(35M/n_train) = 86`, total p = 163 — still
wrong. Fixing both gives total p = 86, matching the paper.

### Effect — Twitter dataset (n=583 250, p_linear=77, d=2, test_prop=0.3)

| | interaction cols | total p | training matrix entries |
|---|---|---|---|
| Paper / correct (both fixes) | 9 | **86** | 35.1M ✓ |
| Fix Bug 1 only (`n_train`, wrong semantics) | 86 | 163 | 66.5M |
| Legacy / current (both bugs) | 61 | 138 | 56.3M |

The paper-reported p∗ = 86 matches only when both fixes are applied.

---

## Note on a third bug (introduced and fixed during this refactoring)

During the preprocessing-pipeline refactoring the formula was briefly written as

```python
pnew = int(np.ceil(self.max_entries / n)) - len(linear_cols)
```

This accidentally fixed Bug 2 (subtracting linear cols from budget) while
retaining Bug 1 (denominator = n). For Twitter this produced a negative `pnew`
and a crash. It was corrected to `min(len(interaction_cols), int(np.ceil(...)))`
to match legacy semantics. The fix is in commit `9c6b50f..`.

---

## Correct implementation

```python
def __call__(self, X):
    ...
    if n * p > self.max_entries:
        linear_cols = list(X.columns)
        interaction_cols = [c for c in X_poly.columns if c not in linear_cols]
        p_total_budget = int(np.ceil(self.max_entries / (self.train_fraction * n)))
        pnew = max(0, min(len(interaction_cols), p_total_budget - len(linear_cols)))
        rng = np.random.default_rng(self.degree)
        sampled = sorted(rng.choice(len(interaction_cols), size=pnew, replace=False))
        return X_poly[linear_cols + [interaction_cols[i] for i in sampled]]
    return X_poly
```

with a new `train_fraction` constructor parameter (default `1.0` for backwards
compatibility; set to `0.7` for paper-correct behaviour with `test_prop=0.3`).

The `PolynomialExpansion` constructor would become:

```python
def __init__(self, degree, max_entries=35_000_000, train_fraction=1.0):
```

Setting `train_fraction=0.7` with the existing `max_entries=35_000_000`
reproduces the paper's intent exactly. The `NEURIPS2023_D2` / `NEURIPS2023_D3`
problem sets would then use `PolynomialExpansion(2, train_fraction=0.7)` etc.

This change would alter cached results, so it should be a deliberate opt-in
rather than a default change.
