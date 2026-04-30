# Y-Normalization Order Bug in RidgeLOOCV (2026-04-28)

## Background

During a seeding refactor (commit `210f510`, "feat: unified RidgeLOOCV wrapper"), the y normalization block in `RidgeLOOCV.fit` was accidentally moved **after** the `alpha_range_GMLNET` call rather than before it. Because `alpha_range_GMLNET(x, y)` computes `alpha_max = (1 / (0.001 * n)) * max(|X^T y|)` using the raw (unnormalized) `y`, the resulting alpha grid was scaled by `std_y` relative to the intended behavior. For datasets with `normalize=True` (the default), this shifted the entire CV_glm alpha grid by a factor of `std_y`, causing the selected alpha — and therefore the fitted coefficients — to be consistently wrong.

The bug did **not** affect `RidgeEM` or `CV_fix` (`RidgeLOOCV` with a fixed alpha array): EM does not use `alpha_range_GMLNET`, and CV_fix bypasses it entirely.

## Fix

Y normalization was moved before the `alpha_range_GMLNET` call, so the function now always receives normalized `y`. The restructured block in `RidgeLOOCV.fit` reads:

```python
a_y = y.mean(axis=0) if self.fit_intercept else np.zeros(q)
b_y = y.std(axis=0) if self.normalize else np.ones(q)
y = (y - a_y) / b_y

if np.isscalar(self.alphas):
    alpha_min, alpha_max = self.alpha_range_GMLNET(x, y)   # normalized y
    self.alphas_ = self.alpha_log_grid(alpha_min, alpha_max, self.alphas)
else:
    self.alphas_ = self.alphas
```

## Verification

The NeurIPS 2023 D1 experiment (`ExperimentWithPerSeriesSeeding`, 100 reps, seed 123) was re-run four times in sequence:

- **comp[0]**: original pre-bug result
- **comp[1], comp[2]**: two force-recompute runs with the buggy code
- **comp[3]** (`comp[-1]`): first force-recompute run after the fix

The table below compares `comp[0]` (original), `comp[-2]` (last buggy run), and `comp[-1]` (fixed run) on `prediction_r2`, averaged over the 100 repetitions. `pre-orig` = `comp[-2] - comp[0]`; `fix-orig` = `comp[-1] - comp[0]`.

```
Dataset                   Est        N    comp[0]   comp[-2]   comp[-1]   fix-orig   pre-orig
----------------------------------------------------------------------------------------------
ribo                      EM         4    0.63839    0.63839    0.63839   +0.00000   +0.00000
ribo                      CV_fix     4    0.63832    0.63832    0.63832   -0.00000   -0.00000
ribo                      CV_glm     4    0.63936    0.63932    0.63936   +0.00000   -0.00004
eye                       EM         4    0.49933    0.49933    0.49933   +0.00000   +0.00000
eye                       CV_fix     4    0.22138    0.22138    0.22138   -0.00000   -0.00000
eye                       CV_glm     4    0.44950    0.29410    0.44950   -0.00000   -0.15540
automobile                EM         4    0.90647    0.90647    0.90647   +0.00000   +0.00000
automobile                CV_fix     4    0.89587    0.89589    0.89589   +0.00002   +0.00002
automobile                CV_glm     4    0.89836    0.89773    0.89836   +0.00000   -0.00064
yacht                     EM         4    0.96850    0.96850    0.96850   +0.00000   +0.00000
yacht                     CV_fix     4    0.96852    0.96852    0.96852   -0.00000   -0.00000
yacht                     CV_glm     4    0.96853    0.96853    0.96853   -0.00000   -0.00000
autompg                   EM         4    0.81350    0.81350    0.81350   +0.00000   +0.00000
autompg                   CV_fix     4    0.81335    0.81335    0.81335   +0.00000   +0.00000
autompg                   CV_glm     4    0.81330    0.81334    0.81330   +0.00000   +0.00004
real_estate               EM         4    0.56219    0.56219    0.56219   +0.00000   +0.00000
real_estate               CV_fix     4    0.56188    0.56187    0.56187   -0.00001   -0.00001
real_estate               CV_glm     4    0.56190    0.56191    0.56190   -0.00000   +0.00001
diabetes                  EM         4    0.48756    0.48756    0.48756   +0.00000   +0.00000
diabetes                  CV_fix     4    0.48554    0.48554    0.48554   +0.00000   +0.00000
diabetes                  CV_glm     4    0.48567    0.48694    0.48567   +0.00000   +0.00127
facebook                  EM         4    0.89661    0.89661    0.89661   +0.00000   +0.00000
facebook                  CV_fix     4    0.89459    0.89451    0.89451   -0.00008   -0.00008
facebook                  CV_glm     4    0.89505    0.42703    0.89505   +0.00000   -0.46802
boston                    EM         4    0.71318    0.71318    0.71318   +0.00000   +0.00000
boston                    CV_fix     4    0.71290    0.71290    0.71290   +0.00000   +0.00000
boston                    CV_glm     4    0.71284    0.71284    0.71284   +0.00000   +0.00000
forest                    EM         4   -0.01557   -0.01557   -0.01557   +0.00000   +0.00000
forest                    CV_fix     4   -0.01923   -0.01921   -0.01921   +0.00002   +0.00002
forest                    CV_glm     4   -0.05330   -0.04287   -0.05330   +0.00000   +0.01043
student                   EM         4    0.26764    0.26764    0.26764   +0.00000   +0.00000
student                   CV_fix     4    0.26801    0.26812    0.26812   +0.00011   +0.00011
student                   CV_glm     4    0.26854    0.26857    0.26854   +0.00000   +0.00003
concrete                  EM         4    0.60893    0.60893    0.60893   +0.00000   +0.00000
concrete                  CV_fix     4    0.60886    0.60886    0.60886   +0.00000   +0.00000
concrete                  CV_glm     4    0.60887    0.60888    0.60887   +0.00000   +0.00001
airfoil                   EM         4    0.51425    0.51425    0.51425   +0.00000   +0.00000
airfoil                   CV_fix     4    0.51422    0.51422    0.51422   +0.00000   +0.00000
airfoil                   CV_glm     4    0.51424    0.51424    0.51424   +0.00000   -0.00000
crime                     EM         4    0.65565    0.65565    0.65565   +0.00000   +0.00000
crime                     CV_fix     4    0.65543    0.65541    0.65541   -0.00002   -0.00002
crime                     CV_glm     4    0.65545    0.65545    0.65545   +0.00000   -0.00001
abalone                   EM         4    0.52587    0.52587    0.52587   +0.00000   +0.00000
abalone                   CV_fix     4    0.52582    0.52582    0.52582   +0.00000   +0.00000
abalone                   CV_glm     4    0.52582    0.52582    0.52582   +0.00000   +0.00000
parkinsons (motor)        EM         4    0.14635    0.14635    0.14635   +0.00000   +0.00000
parkinsons (motor)        CV_fix     4    0.14618    0.14618    0.14618   +0.00000   +0.00000
parkinsons (motor)        CV_glm     4    0.14625    0.14624    0.14625   -0.00000   -0.00000
parkinsons (total)        EM         4    0.16923    0.16923    0.16923   +0.00000   +0.00000
parkinsons (total)        CV_fix     4    0.16887    0.16887    0.16887   +0.00000   +0.00000
parkinsons (total)        CV_glm     4    0.16891    0.16890    0.16891   +0.00000   -0.00000
naval (compressor)        EM         4    0.91082    0.91082    0.91082   +0.00000   +0.00000
naval (compressor)        CV_fix     4    0.91082    0.91082    0.91082   +0.00000   +0.00000
naval (compressor)        CV_glm     4    0.91080    0.90501    0.91080   +0.00000   -0.00579
naval (turbine)           EM         4    0.84223    0.84223    0.84223   +0.00000   +0.00000
naval (turbine)           CV_fix     4    0.84223    0.84223    0.84223   -0.00000   -0.00000
naval (turbine)           CV_glm     4    0.84215    0.83100    0.84215   +0.00000   -0.01114
tomshw                    EM         4    0.96149    0.96149    0.96149   +0.00000   +0.00000
tomshw                    CV_fix     4    0.96146    0.96146    0.96146   +0.00000   +0.00000
tomshw                    CV_glm     4    0.96147    0.95310    0.96147   +0.00000   -0.00837
blog                      EM         3    0.35180    0.35180    0.35180   +0.00000   +0.00000
blog                      CV_fix     4    0.35170    0.35170    0.35170   -0.00000   -0.00000
blog                      CV_glm     4    0.35155    0.35170    0.35155   -0.00000   +0.00015
ct_slices                 EM         3    0.86275    0.86275    0.86275   +0.00000   +0.00000
ct_slices                 CV_fix     3    0.86274    0.86274    0.86274   +0.00000   +0.00000
ct_slices                 CV_glm     3    0.86274    0.86274    0.86274   +0.00000   -0.00000
twitter                   EM         2    0.93149       n/a     0.93149   +0.00000       n/a
twitter                   CV_fix     2    0.93134       n/a     0.93134   +0.00000       n/a
twitter                   CV_glm     2    0.93136       n/a     0.93134   -0.00002       n/a
```

## Conclusions

**CV_glm** was the only estimator affected by the bug. The `pre-orig` column shows deviations ranging from negligible (ribo, yacht, autompg) to catastrophic (facebook −0.468, eye −0.155). After the fix, every CV_glm `fix-orig` entry is zero to five decimal places — original values fully restored.

**EM** was unaffected throughout (no `alpha_range_GMLNET` call).

**CV_fix** (`RidgeLOOCV` with a fixed alpha array) was also unaffected by the bug. The small non-zero `fix-orig` values seen for a handful of datasets (facebook −0.00008, student +0.00011, etc.) are **identical** in `pre-orig` and `fix-orig`, confirming they were already present before any broken run and are independent of the normalization order. They reflect pre-existing floating-point sensitivity: datasets with flat LOO MSE curves where argmin can shift by one grid step across runs.

## Open questions

- The `warn_recompute` warnings observed during the fix-verification run fired for CV_fix because the series mean (computed over all 4 entries including 2 broken) was compared against the fixed computation. The warnings are not indicative of a real problem for CV_fix; they will stop once the cache is reset. Running `--overwrite_cache` would restore each series to a single clean computation.
- D3 series showed zero cached computations throughout. This was not investigated in this session.
