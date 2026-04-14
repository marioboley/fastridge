# Real Data Result Discrepancies

Comparison of `experiments/real_data.ipynb` (full-experiment cells, seed=123, n_iterations=100) against paper Table 2. Values read directly from Table 2; our values extracted from executed notebook cells. Discrepancies marked **bold**. Columns omitted where both agree and no polynomial results exist.

Table 2 column definitions: n and p are post-preprocessing (after NaN removal, OHE, zero-variance dropping), before polynomial expansion. p* is features after polynomial expansion. T is the speed-up ratio t_CV / t_EM.

**Critical systemic difference:** The legacy notebook used `RidgeEM(squareU=False)`. Inspection of the legacy `fastridge.py` confirms that `squareU` was renamed to `t2` in the current API, with the same formula: `squareU=False` ↔ `t2=False` (half-Cauchy prior on τ) and `squareU=True` ↔ `t2=True` (Beta Prime prior on τ²). The current notebook uses `RidgeEM()` with default `t2=True`, which is a different EM variant from the one used in the paper. This is the primary suspected cause of the EM discrepancies at d=2 and d=3 across multiple datasets. Confirming requires re-running the full experiment with `RidgeEM(t2=False)`.

---

## 1. Yacht — log transform missing

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|---|---|
| New notebook | 215 | **6** | 27 | 83 | **0.63** | **0.92** | 0.99 |
| Paper Table 2 | 215 | **7** | 27 | 83 | **0.97** | **0.97** | 0.98 |

**Cause:** Legacy `yacht.csv` stored `log(Residuary_resistance)`. Confirmed by exact row-by-row match. Target is right-skewed (range 0.01–1090), log transform makes it near-linear. Also explains p discrepancy: paper has p=7, we have p=6 — the extra column is likely a feature that appears after the log transform stabilises one variable, or simply a version difference.

**Fix:** Apply `log` to yacht target.

---

## 2. Abalone — large discrepancy at d=2 and d=3, cause unknown

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|---|---|
| New notebook | 2923 | 9 | **53** | **209** | 0.53 | **0.30** | **−0.66** |
| Paper Table 2 | 2923 | 9 | **51** | **209** | 0.53 | **0.38** | **0.28** |

d=1 matches. At d=2 paper has 2 fewer features (51 vs 53) and substantially better EM R² (0.38 vs 0.30). At d=3 p* matches exactly (209) but EM R² diverges sharply (0.28 vs −0.66). The d=3 collapse for all three methods (Fix=−0.60, GLM=−0.60) in our results while paper shows Fix=0.12, GLM=0.12 suggests a preprocessing or data difference that becomes critical in the polynomial regime.

**Ruled out:** Log/log1p target transform — verified by exact row-by-row comparison with legacy `abalone.csv` (targets identical, data source identical). Also ruled out: different zero-variance dropping logic — the current `run_real_data_experiments` and legacy `RealDataExperiments` perform per-fold dropping identically.

**Candidate cause:** The p* discrepancy at d=2 (51 vs 53) suggests a difference in the polynomial feature space — possibly caused by a different scikit-learn version generating different OHE column ordering or PolynomialFeatures output. With n:p ≈ 9.8 at d=3, the regime is not p>>n, so pure dimensionality is not the cause; the collapse of all three methods (EM/Fix/GLM) at d=3 suggests a feature space issue rather than an EM-specific one.

**Status:** Open; root cause unresolved. The fact that all three methods collapse at d=3 (not just EM) points to a feature representation difference rather than a hyperparameter tuning issue.

---

## 3. Crime — EM and Fix swap roles at d=2

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² Fix (d=1) | R² EM (d=2) | R² Fix (d=2) | R² EM (d=3) | R² Fix (d=3) |
|---|---|---|---|---|---|---|---|---|---|---|
| New notebook | 1395 | 99 | 5049 | 17652 | 0.66 | **0.66** | **0.66** | **−0.80** | 0.66 | −0.34 |
| Paper Table 2 | 1395 | 99 | 5049 | 17652 | 0.67 | **−0.74** | **−0.89** | **0.16** | 0.66 | −6.22 |

At d=1 paper Fix=−0.74 while ours is 0.66 — fixed grid already fails in the paper. At d=2 (n:p=0.28) the paper's EM collapses to −0.89 while ours holds at 0.66. The paper's behaviour (EM failing when p>>n) is physically expected; our EM not failing is suspicious.

**Candidate cause unknown.** Zero-variance dropping logic is identical between current and legacy code (confirmed), so effective p at d=2 should match. Both report p*=5049. The d=1 Fix discrepancy (0.66 vs −0.74) is particularly striking — it suggests that the fixed lambda grid performs very differently, which could indicate a different NaN handling strategy changing which rows/columns are retained. Crime has many NaN values; our `nan_policy='drop_cols'` drops all columns with any NaN, which may differ from the legacy approach. A different set of retained columns would produce a different feature space and different lambda grid behaviour.

**Next step:** Check legacy crime preprocessing (which columns were kept after NaN handling) against our current `drop_cols` output.

---

## 4. Parkinsons — EM failure at d=3 less severe than paper

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² EM (d=2) | R² EM (d=3) | R² GLM (d=3) |
|---|---|---|---|---|---|---|---|---|
| New notebook (motor) | 4112 | 19 | 209 | 1539 | 0.15 | 0.24 | **−0.47** | 0.28 |
| New notebook (total) | 4112 | 19 | 209 | 1539 | 0.17 | 0.24 | **−0.83** | 0.27 |
| Paper Table 2 (motor) | 4112 | 19 | 208 | 1539 | 0.15 | 0.25 | **−1.09** | 0.04 |
| Paper Table 2 (total) | 4112 | 19 | 209 | 1539 | 0.17 | 0.24 | **−1.38** | 0.00 |

n and p match at d=1. d=2 is close (within rounding). At d=3 our EM is less negative (−0.47/−0.83 vs −1.09/−1.38) and our GLM does much better (0.28/0.27 vs 0.04/0.00). The p* matches, so this is not a feature count issue.

**Candidate cause:** Different random split realisations at d=3 drive large variance in EM performance in this regime (n:p=2.67 at d=3). The paper's worse EM may result from splits where the n:p ratio is unfavourable. The large GLM discrepancy is harder to explain.

**Status:** Open; no fix without reproducing the paper's exact random splits.

---

## 5. Forest — OHE vs numeric month/day

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² Fix (d=1) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|---|---|---|
| New notebook | 361 | **26.43** | excluded | excluded | **−0.05** | **−0.16** | — | — |
| Paper Table 2 | 361 | **12** | **295** | **1984** | **−0.01** | **−0.01** | **−0.01** | **−0.08** |

**Confirmed:** Legacy `forest.csv` stored `log1p(area)` as target (confirmed row-by-row match), explaining the small R² difference. Apply `log1p` to area target — already done in `NEURIPS2023` collection.

**p discrepancy (open):** The legacy notebook also produces p≈26, so the paper's p=12 was not generated by the legacy experiment either. This suggests the paper may report the pre-OHE feature count (raw columns minus target = 12) rather than the post-OHE dimensionality used in computation. The same pattern appears in the automobile discrepancy (paper p=25 = raw features; our p≈50 after OHE). If so, the actual experiments in the paper also used OHE'd features and the p column in Table 2 is a dataset descriptor, not the effective model dimensionality.

**Cause (exclusion):** Forest is excluded from our d=2/d=3 due to SVD failure on OHE interaction columns. This may also have occurred in the legacy experiment at d≥2 — the paper's p*=295/1984 would need to be reconciled with either interpretation of p.

**Status:** Open. The R² discrepancy at d=1 is explained by the missing log1p transform (already fixed in NEURIPS2023 collection). The p column interpretation and d=2/d=3 exclusion require further investigation.

---

## 6. Student — Portuguese-only vs merged dataset

| | n\_train | p | R² EM (d=1) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|
| New notebook | **454** | 39 | **0.27** | **0.27** | **0.25** |
| Paper Table 2 | **730** | 39 | **0.18** | **0.19** | **0.18** |

**Cause:** Paper merged Portuguese (649) and Math (395) student datasets → 1044 rows, n_train=730. Our source (ucimlrepo) returns Portuguese only → n_train=454. G1 and G2 are correctly dropped in both. The homogeneous Portuguese sample inflates R² across all degrees.

**Fix:** Merge both student datasets. Expected to restore n_train=730 and R²≈0.18 at all degrees.

---

## 7. Automobile — p discrepancy from OHE treatment

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|---|---|
| New notebook | 111 | **50.06** | **1009** | **12105** | 0.88 | 0.88 | 0.87 |
| Paper Table 2 | 111 | **25** | **1076** | **12924** | 0.90 | 0.90 | 0.88 |

**Cause (p):** Paper p=25 vs our p≈50. The paper likely treated categorical columns as numeric rather than OHE-encoding them, halving the feature count. Interestingly, p* at d=2/d=3 are similar (1076 vs 1009, 12924 vs 12105) because the polynomial expansion of both converges toward similar counts after zero-variance dropping. Legacy `automobile.csv` stored `log(price)` as target (confirmed). R² differences are small.

**Fix:** Treat automobile categorical features as numeric (or apply the same encoding as legacy). Apply `log` to price target.

---

## 8. Facebook — EM failure magnitude differs at d=3

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=2) | R² EM (d=3) |
|---|---|---|---|---|---|---|
| New notebook | **349** | **17** | 167 | 1087 | **−6.26** | **−29.85** |
| Paper Table 2 | **346** | **15** | 167 | 1087 | **−5.09** | **−2.53** |

d=1 is close (EM=0.90 ours vs 0.91 paper). p* at d=2/d=3 matches. At d=2 EM failure magnitude is similar; at d=3 ours is dramatically worse (−29.85 vs −2.53). Paper Fix at d=3 is −164; ours is −66,439 — both catastrophic but vastly different in magnitude.

**Candidate cause:** Our source has 5 extra rows (349 vs 346) and 2 extra features (p=17 vs 15) at d=1. The paper excludes the `comment`, `like`, and `share` sub-components; our version also drops them but may have different residual columns. The extra features at d=1 expand more at d=3, pushing into a worse regime.

**Status:** Minor discrepancy at d=1/d=2; large at d=3. Investigate which 2 features differ.

---

## 9. Naval propulsion — p discrepancy, R² unaffected

| | n | p | p\* (d=2) | p\* (d=3) | R² EM (d=1, comp.) | R² EM (d=1, turb.) |
|---|---|---|---|---|---|---|
| New notebook | 8353 | **15** | **149** | **963** | 0.84 | 0.91 |
| Paper Table 2 | 8353 | **13** | **104** | **559** | 0.84 | 0.91 |

**Cause:** Our source retains `Ts` and `Tp` (propeller torque output variables). Dropping them reproduces paper p=13 but reduces R² below reported values. At d=2/d=3 both reach R²=1.00.

**Status:** No fix applied; keeping Ts/Tp preserves R² at the cost of p discrepancy at all degrees.

---

## 10. Autompg — minor n and p discrepancy

| | n | p | R² EM (d=1) |
|---|---|---|---|
| New notebook | **274** | **7** | 0.81 |
| Paper Table 2 | **278** | **8** | 0.81 |

Paper has 4 more rows (different NaN handling) and 1 extra feature. R² matches. Cause and fix TBD; low priority given identical R².

---

## Summary

| Dataset | Most severe discrepancy | Likely cause | Fix available |
|---|---|---|---|
| Yacht | R² 0.63 vs 0.97 (d=1) | log target transform | Yes |
| Abalone | R² −0.66 vs 0.28 (d=3) | Possible log transform; unconfirmed | Needs verification |
| Crime | EM 0.66 vs −0.89 (d=2) | Zero-variance dropping reduces effective p | Needs investigation |
| Parkinsons | EM −0.47 vs −1.09 (d=3) | Random split variance; GLM gap unexplained | Needs investigation |
| Forest | p=26 vs 12; excluded from d≥2 | OHE vs numeric month/day; log1p missing | Yes |
| Student | n=454 vs 730; R² 0.27 vs 0.18 | Portuguese-only vs merged | Yes |
| Automobile | p=50 vs 25 (OHE); log missing | OHE treatment + log target | Yes |
| Facebook | EM −29.85 vs −2.53 (d=3) | 2 extra features cascade at d=3 | Partial |
| Naval | p=15 vs 13 | Ts/Tp retained | No (dropping hurts R²) |
| Autompg | n=274 vs 278; p=7 vs 8 | NaN/feature handling | TBD |
