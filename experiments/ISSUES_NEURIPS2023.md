# Known Issues with NeurIPS 2023 Experiments

## Empirical Regression Problems

### Facebook

The prediction problem uses variable `Total Interactions` as target, defined as sum of variables `comment`, `like`, `share`, without dropping those variables. As a result, not only is the multi-target nature of the problem not reflected, but also a severe information leakage is introduced that renders the problem trivial.

## Polynomial Feature Subsampling

### Intent

Section 5.2 of the NeurIPS paper states:

> "For each experiment, we repeated the process 100 times and used a random 70/30 train-
> test split. Due to memory limitations, we limit our design matrix size to a maximum of
> 35 million entries. If the number of transformed predictors exceeded this limit, we uniformly
> sub-sampled the interaction variables to ensure that $p^* \leq 35 000 000 / (0.7n)$"

Given the intention of 70/30 train-test splits and bounding the size of the design (training) matrix, we can assume that the symbol $n$ in the bound for $p^*$ refers to the dataset size before splitting. In other words, the intent is to enforce

$$
p^* \leq 50 000 000 / n = 35 000 000 / (0.7n) = 35 000 000 / n_\mathrm{train}
$$

Further, $p^*$ is supposed to refer to the total number of columns of the design matrix, i.e., $p^*=p_\mathrm{new} + p$, resulting in the intention to limit the number of added columns $p_\mathrm{new}$ to

$$
p_\mathrm{new} \leq 50 000 000 / n - p
$$

For the dataset `twitter`, the number of columns reported in Tab. 2 of the paper ($86$), matches this formula after rounding, noting that the symbol $n$ in this table refers to the train size $n_\mathrm{train}=408275$ and assuming that the reported number of "raw features" $p=77$ refers indeed to the number of variables of the linear model (i.e., no one-hot-encoding).

### Legacy code

The original experiment runner implemented the feature limit with the following code fragment located *before* the training/test split (and after one-hot-encoding of columns with `dtype in ['object', 'category']`):

```python
poly = PolynomialFeatures(degree=polynomial, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

ppoly = X_poly.shape[1]
npoly = X_poly.shape[0]

if npoly*ppoly > 35000000 :
    X_poly = X_poly.drop(X.columns, axis=1)
    pnew = np.ceil(35000000/npoly)
    X_poly = X_poly.iloc[:, np.random.choice(X_poly.shape[1], size=int(pnew), replace=False)]
    X = pd.concat([X, X_poly], axis=1)                                                           
else:
    X = X_poly
```

This violates the stated intent in two ways:

**Bug 1: denominator uses full $n$ instead of $n_\mathrm{n_train} 0.7n$.**
Variable `npoly` refers to the full dataset size. The paper bounds the training matrix, so the
correct denominator is `0.7 * npoly`. Using `npoly` produces a smaller
per-column budget and therefore fewer interaction columns than described.

**Bug 2: budget applied to interaction columns, not total columns.**
After dropping the linear columns, `pnew = ceil(35M/n)` is used as the count
of interaction columns to keep. The final matrix has `p_linear + pnew` total
columns, which can substantially exceed the paper's bound of `35M/(0.7n)`.

For the Twitter dataset (n = 583 250, p_linear = 77, d = 2):

| | $p_\mathrm{new}$ | $p_*$ | training matrix entries |
|---|---|---|---|
| Paper intent | 9 | **86** | 35.1 M |
| Legacy code  | 61 | 138 | 56.3 M |

The paper's Table reports p* = 86 for Twitter at d = 2, which matches the
stated formula after rounding and indicates that the legacy code present in the repository was in fact **not** used to run the reported experiments. 

### Current implementation

The inconsistent behaviour of the legacy code is therefore not retained in the current
codebase and `PolynomialExpansion` instead, implement the paper's stated intent.

`PolynomialExpansion(degree, max_entries=50_000_000)` applies the corrected
formula. The default `max_entries = 50 000 000 = 35M / 0.7` is the full-dataset
equivalent of a 35M training-matrix budget at a 70/30 split. When
`n x p_full > max_entries`, the class keeps
`max(0, ceil(max_entries/n) - p_linear)` interaction columns, bounding the
total feature count to `ceil(max_entries/n)`.

## Redundant Higher-Order Features 

On a more conceptual level, the interaction of the polynomial feature expansion with one-hot-encoding means that a number of redundant features is created, because for $x \in \{0, 1\}$ resulting from one-hot-encoding we have $x=x^2=x^3$.

## Unreproduced Feature Counts

Comparison of the paper's reported p* values against the current implementation
at d = 2 reveals two datasets with discrepancies (all others checked matched).

**abalone** (paper 51, current ~53): The polynomial expansion of one-hot-encoded
binary columns produces squared terms that are algebraically identical to the
originals (x^2 = x for binary x). For abalone's Sex variable (3 levels,
drop-first OHE producing 2 binary columns), this yields 2 non-zero-variance
duplicate columns that survive the zero-variance filter. The paper apparently
excluded these. For datasets where some OHE categories are absent from the data,
the corresponding binary column and its square are both zero-variance and both
filtered, so no discrepancy arises for those datasets.

**automobile** (paper 1076, current ~1009): Discrepancy is in the opposite
direction. Not yet explained; the automobile dataset has many categorical columns
and several missing-value patterns that interact with the zero-variance filter
in ways that differ from the paper's reported count. The root cause has not been
identified.

## Seeding

In `EmpiricalDataExperiment.run()`, the experiment seed is applied to the global
numpy random state *after* `problem.get_X_y()` is called:

```python
X, y = problem.get_X_y()   # polynomial subsampling happens here
...
if self.seed is not None:
    np.random.seed(self.seed)
```

`PolynomialExpansion` uses `np.random.choice` (the global numpy random state)
for interaction column subsampling. Because `get_X_y()` is called before the
seed is set, the subsampling is not controlled by the experiment-level seed and
its outcome depends on whatever global random state happens to be in effect at
call time. Only large datasets that exceed `max_entries` are affected (e.g.
Twitter at d = 2 or d = 3); small datasets that do not trigger subsampling are
unaffected.
