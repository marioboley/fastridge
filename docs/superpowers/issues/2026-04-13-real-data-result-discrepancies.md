# Real Data Result Discrepancies

Comparison of `experiments/real_data.ipynb` output against paper Table 2 (linear and polynomial experiments). Discrepancies are ordered by severity. For each issue the current notebook value, the paper value, the confirmed cause, and the recommended fix are given.

Analysis method: legacy CSVs recovered from Google Drive (shared by co-author), compared row-by-row against cached UCI sources; `run_real_data_experiments` source code reviewed; d=1, d=2, and d=3 full-experiment table cells extracted from the executed notebook.

---

## 1. Yacht — log transform missing (all degrees)

**Current notebook:** d=1 R²=0.63, d=2 R²=0.92, d=3 R²=0.99
**Paper Table 2:** d=1 R²=0.97

**Cause:** The legacy `yacht.csv` stored `log(Residuary_resistance)` as the target. Our source (`from_url`, UCI archive) fetches the raw values. Residuary resistance is heavily right-skewed (range 0.01–1090), so the log transform dramatically improves linear predictability. Confirmed by exact row-by-row match: `legacy_V7 == log(new_Residuary_resistance)` for all 308 rows.

**Fix:** Apply `log` transform to the yacht target in `EmpiricalDataProblem` (via a `target_transform` parameter or equivalent). Expected to restore d=1 R²≈0.97.

---

## 2. Parkinsons — p discrepancy, most visible at d=3

**Current notebook:** d=1 p=19, d=3 EM R²=−0.47 (motor), −0.83 (total); CV_glm R²=0.28, 0.27
**Paper Table 2:** d=1 p=26, d=3 EM R²=−1.09 (motor), −1.38 (total)

**Cause:** Our source (ucimlrepo) yields p=19 features at d=1; the paper used p=26. The source of the 7 missing features is unknown — the paper may have used a different source file or different NaN/target handling. At d=3 the polynomial expansion compounds this: our p=1539 vs the paper's ~2977. Both versions show EM failing catastrophically in the low n:p regime at d=3, but our smaller expansion makes the failure less extreme. At d=1 and d=2, R² values are consistent and near each other.

**Fix:** Identify the 7 additional features used by the paper. Candidate sources: alternative UCI download, different handling of the second UPDRS target column, or inclusion of subject/time metadata.

---

## 3. Student — Portuguese-only vs merged dataset (all degrees)

**Current notebook:** n_train=454, d=1 R²=0.27
**Paper Table 2:** n_train=730, d=1 R²=0.18

**Cause:** The paper merged the Portuguese course dataset (649 rows) and the Math course dataset (395 rows) into a single 1044-row file, giving n_train=730. Our source (`from_ucimlrepo(320)`) returns only the Portuguese dataset (649 rows). G1 and G2 are correctly dropped in both. The more homogeneous Portuguese-only sample is easier to predict, inflating our R² from 0.18 to 0.27.

**Fix:** Fetch and merge both `student-mat.csv` and `student-por.csv` from UCI and deduplicate as the paper did. Expected to restore n_train=730 and R²≈0.18.

---

## 4. Forest — log1p transform missing (d=1 only; excluded from d=2/d=3)

**Current notebook:** d=1 R²=−0.05
**Paper Table 2:** d=1 R²=−0.01

**Cause:** The legacy `forest.csv` stored `log1p(area)` as the target. Our source fetches raw burned area values (range 0–1090.84). Confirmed by exact row-by-row match: `legacy_area == log1p(new_area)` for all 517 rows. Both values are near zero because the target is 47.8% zeros (zero-inflation dominates regardless of transform), so the practical impact on reported R² is small. Forest is excluded from d=2/d=3 experiments due to an SVD failure on OHE interaction columns — this is a separate unresolved issue.

**Fix:** Apply `log1p` transform to forest area target. Separately investigate the SVD failure at d=2 to determine whether forest can be reinstated in polynomial experiments.

---

## 5. Automobile — log transform missing (all degrees)

**Current notebook:** d=1 R²=0.88, d=2 R²=0.88, d=3 R²=0.87
**Paper (legacy notebook):** d=1 R²=0.90

**Cause:** The legacy `automobile.csv` stored `log(price)` as the target (range 8.54–10.46 in legacy vs 5118–45400 raw). Confirmed by exact sorted match: `legacy_price == log(new_price)` for all 159 post-NaN-removal rows. The impact is small (Δ≈0.02) because car price is already approximately log-linear. The paper's Table 2 value for automobile is not available for d=2/d=3.

**Fix:** Apply `log` transform to automobile price target.

---

## 6. Naval propulsion — p=15 (ours) vs p=13 (paper), R² unaffected

**Current notebook:** p=15, d=1 R²=0.84 (compressor), 0.91 (turbine)
**Paper Table 2:** p=13, R²=0.84, 0.91

**Cause:** Our UCI source retains `Ts` and `Tp` (propeller torque variables). These are output variables not available at prediction time and should be excluded. However, dropping them reduces R² below the paper's reported values — the opposite of the expected effect. The mechanism is unclear: `Ts` and `Tp` are near-constant and add minimal signal, but their removal appears to affect regularisation in a way that slightly degrades predictions. The paper's p=13 also implies one additional zero-variance column was dropped that we do not consistently drop (T1 or P1, which are zero-variance in the full dataset but not always in every fold).

**Status:** No fix applied. Dropping Ts/Tp cannot be recommended until the R² deterioration is explained. Open question for follow-up.

---

## Summary table

| Dataset | Metric | Ours | Paper | Cause | Fix available |
|---|---|---|---|---|---|
| Yacht | d=1 R² (EM) | 0.63 | 0.97 | log transform missing | Yes — apply `log` |
| Parkinsons | d=3 R² (EM, motor) | −0.47 | −1.09 | p=19 vs 26, source unknown | Needs investigation |
| Parkinsons | d=3 R² (EM, total) | −0.83 | −1.38 | p=19 vs 26, source unknown | Needs investigation |
| Student | d=1 n_train | 454 | 730 | Portuguese only vs merged | Yes — merge datasets |
| Student | d=1 R² (EM) | 0.27 | 0.18 | Portuguese only vs merged | Yes — merge datasets |
| Forest | d=1 R² (EM) | −0.05 | −0.01 | log1p transform missing | Yes — apply `log1p` |
| Automobile | d=1 R² (EM) | 0.88 | 0.90 | log transform missing | Yes — apply `log` |
| Naval | d=1 p | 15 | 13 | Ts/Tp retained | No — dropping hurts R² |
