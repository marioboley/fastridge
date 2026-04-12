---
name: Real Data p Discrepancies
description: Known gaps between current experiment setup and paper Table 2/5 p values — open investigation, not blocking figure reproduction
type: project
---

# Real Data p Discrepancies

## Background

Comparison of our current linear (d=1) experiment results against the paper's Table 2 (linear column p values) and Table 5 (raw feature counts) reveals several discrepancies in the number of predictors p and occasionally in n_train or R². These are documented here for future alignment work.

## Known Discrepancies

| Dataset | Our p (linear) | Paper p (Table 2 linear) | Likely cause |
|---|---|---|---|
| Forest | ~26 | 13 | We OHE `month` (12 cats → 11) and `day` (7 cats → 6); paper likely treated these as numeric |
| Crime | 99 | 128 | Our `drop_cols` NaN policy removes ~29 columns with any NaN; paper appears to have kept them (possibly imputed or used a different NaN threshold) |
| Parkinsons | 19 | 26 | Unknown — paper may not have dropped the other UPDRS target column, or used a different source |
| Naval propulsion | 15 | 13 (legacy) | Paper/legacy dropped `Ts` (starboard propeller torque) and `Tp` (port propeller torque) as output variables not available at prediction time; dropping them + 1 zero-variance col gives p=13 |
| Yacht | 6 | 7 | Paper has one extra column; R²=0.63 (ours) vs 0.97 (paper) is unexplained — same seed, same p count doesn't reproduce |
| Facebook | 17 | 19 | We drop `comment`, `like`, `share` (3 sub-components of Total Interactions); paper drops only 2 (comment not mentioned in original comment) — or paper source had 2 additional columns |
| Student | p=39, n_train=454 | p=33, n_train=730 | Paper uses both Math + Portuguese datasets merged; we use Portuguese-only with G1/G2 dropped. Two-dataset design deferred pending UCI site recovery (see `issues/2026-04-11-student-dataset-structure.md`) |
| Abalone | 9 | 8 | We OHE `Sex` (M/F/I → 2 binary cols); paper may have used label encoding or dropped one category differently |

## Notes

- Discrepancies in p affect polynomial feature counts (p* grows as O(p^d)), so these become more significant for d=2 and d=3 experiments.
- The Naval propulsion case (Ts/Tp) is the most actionable: dropping those columns is principled (they are output variables) and would match legacy. Verified that dropping them reduces R², so the decision is left to the user.
- The Yacht discrepancy (R²=0.97 in paper vs 0.63 in ours) remains unexplained. No R source for the dataset was found; legacy CSVs are unavailable.
- Forest OHE behaviour is the most impactful discrepancy for polynomial experiments (p going from 13 to ~26 before expansion).

## Status

Open investigation. Not blocking figure reproduction (Fig. 3). Revisit before producing final paper table equivalents.
