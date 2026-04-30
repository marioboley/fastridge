# Polynomial Expansion Efficient Transform

## Motivation

`PolynomialExpansion.__call__` currently calls `fit_transform(X)` which materialises
all C(p+d, d) feature combinations for all n rows before subsampling down to P << C(p+d, d)
selected columns. This violates the O(nP) complexity that is achievable: the correct
approach is to determine the P feature combinations first, then materialise only those.

sklearn's `PolynomialFeatures.transform` always computes all C(p+d, d) features
internally regardless of what `powers_` contains (which is also a read-only property in
sklearn >= 1.6), so sklearn cannot be used to achieve this. The implementation must bypass
sklearn entirely.

## Design

Replace `PolynomialFeatures.fit_transform` with a direct implementation:

1. Enumerate all interaction combinations (degree >= 2) as tuples of column indices using
   `itertools.combinations_with_replacement`. This is cheap: only integer tuples, no
   data arrays.
2. If `n * (p + len(all_combos)) > max_entries`, randomly sample P interaction tuples as
   now, using the same budget logic and the same `rng`.
3. For each selected tuple, compute the feature column as the elementwise product of the
   corresponding input columns: one numpy multiply per index in the tuple, O(n) per
   column, O(nP) total.
4. Derive feature names from column names and the tuples using the same convention as
   sklearn: space-separated factors, `^exp` suffix for exponents > 1 (e.g. `'a^2 b'`).

A module-level helper `_poly_feature_name(col_names, combo)` derives the name from a
combination tuple.

## Behavior

The change is behavior-neutral: same output DataFrame, same column names and ordering,
same random selection logic when subsampling. The naming convention is verified by a
pytest that compares the bypass output against `sklearn.PolynomialFeatures.fit_transform`
on small inputs where no subsampling occurs.

## Naming

Internal variables for degree >= 2 terms are named `higher_degree_combos` /
`higher_degree_names`, not `interaction_*`. "Interaction" generally means
cross-product terms only (products of at least two distinct inputs, i.e.
`interaction_only=True`); degree >= 2 terms include both interactions and pure powers
(e.g. `a^2`). Aligning with sklearn's nomenclature avoids the misleading name in the
current implementation.

## Scope

`experiments/problems.py`: replace `PolynomialExpansion.__call__`, factor out `@staticmethod _feature_name` inside the class.
No interface or docstring changes to `PolynomialExpansion`.

## Future

Natural extensions once this is in place: an `interaction_only` flag (matching sklearn)
and skipping redundant powers of one-hot encoded columns. These are out of scope here.
