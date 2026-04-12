---
name: Real Data p Discrepancies
description: Known gaps between current experiment setup and paper Table 2/5 p values — open investigation, not blocking figure reproduction
type: project
---

# Real Data p Discrepancies

## Background

Comparison of our current linear (d=1) experiment results against the paper's Table 2 (linear column p values) and Table 5 (raw feature counts) reveals several discrepancies in the number of predictors p. Datasets where p matches are listed with their expected p* values for d=2 and d=3 (from Table 2 / computed via PolynomialFeatures formula) to serve as verification targets when running polynomial experiments.

## Confirmed Matching — Expected p* Values

p* computed via PolynomialFeatures(degree=d, include_bias=False): d=2: p(p+3)/2; d=3: p + p(p+1)/2 + p(p+1)(p+2)/6. Values verified against Table 2 for non-subsampled cases.

| Dataset | p | p* (d=2) | p* (d=3) | Notes |
|---|---|---|---|---|
| Abalone | 9 | 54 | 219 | R² matches at seed=123; Table 5 raw p=8 reflects pre-OHE count |
| Airfoil | 5 | 20 | 55 | Table 2 confirms p=5 |
| Concrete | 8 | 44 | 164 | Table 5 raw p=9 includes target column |
| Boston | 13 | 104 | 559 | |
| Diabetes | 10 | 65 | 285 | sklearn source, no ambiguity |
| Eye | 200 | 20300 | sub-sampled | d=3 exceeds 35M cap; p* ≈ 35M/(0.7n) ≈ 416667, varies per run |
| Ribo | 4088 | sub-sampled | sub-sampled | d=2 already exceeds cap; p* ≈ 703622, varies per run |

## Known Discrepancies

| Dataset | Our p | Paper p (Table 2) | Likely cause |
|---|---|---|---|
| Forest | ~26 | 13 | We OHE `month` (12 cats → 11) and `day` (7 cats → 6); paper likely treated these as numeric |
| Crime | 99 | 128 | Our `drop_cols` NaN policy removes ~29 columns; paper kept them (different NaN threshold or imputation) |
| Parkinsons | 19 | 26 | Unknown — paper may not have dropped the other UPDRS target column, or used a different source |
| Naval propulsion | 15 | 13 (paper + legacy) | Paper/legacy dropped `Ts` and `Tp` (propeller torques — output variables not available at prediction time); dropping them + 1 zero-variance col gives p=13. Verified: dropping reduces R², decision deferred to user. |
| Yacht | 6 (ours + legacy) | 7 (paper) | No discrepancy with legacy; paper Table 5/2 shows p=7 but our dataset and legacy CSVs both have p=6. R²=0.63 (ours/legacy at seed=123) vs 0.97 (paper) remains unexplained. |
| Facebook | 17 | 19 (paper) | We drop `comment`, `like`, `share` (3 sub-components); paper drops 2. Or paper source had 2 additional columns beyond the 3 sub-components. |
| Student | p=39, n=454 | p=33, n=730 | Paper uses both Math + Portuguese datasets merged; we use Portuguese-only with G1/G2 dropped. Deferred pending UCI site recovery (see `issues/2026-04-11-student-dataset-structure.md`). |
| Automobile | p≈50 | p=26 | Paper p=26 suggests far fewer columns; our ucimlrepo source likely includes more categorical columns after OHE. |
| Real estate | 6 | 7 | One feature fewer in our source. |
| Autompg | 7 | 8 | Matches legacy p=7; paper has one extra feature. |

## Notes

- Discrepancies compound for polynomial experiments: Forest (p~26) expands to ~26*29/2=377 for d=2 vs paper's 13*16/2=104.
- Forest OHE is the most impactful for polynomial experiments; treating `month` and `day` as numeric would align with paper.
- The Yacht R²=0.97 vs 0.63 mystery is unresolved and may relate to a different version of the data file used by the paper.

## Status

Open investigation. Not blocking figure reproduction (Fig. 3). Revisit before producing final paper table equivalents.
