---
name: Figure 3 Polynomial Experiment
description: Design for reproducing Fig. 3 from the paper appendix — polynomial feature experiments (d=1,2,3) and 2×3 scatter figure in real_data.ipynb
type: project
---

# Figure 3 Polynomial Experiment Design

## Background

Section 5.2 of the paper describes three linear regression settings evaluated on the real-world datasets:

1. **d=1 (linear)**: standard linear regression — existing preview and full experiment cells
2. **d=2 (second-order)**: polynomial regression with all pairwise interactions and squared terms
3. **d=3 (third-order)**: polynomial regression with all three-way interactions and cubic terms

Key parameters from section 5.2:
- 100 iterations for d=1; **30 iterations** for d=2 and d=3
- 70/30 train/test split (`test_prop=0.3`)
- Design matrix capped at 35 million entries: if `n * p* > 35_000_000`, interaction variables are sub-sampled uniformly so that `p* ≤ 35_000_000 / (0.7 * n)` — already implemented in `run_real_data_experiments` via the `polynomial` parameter
- Original (main-effects) variables are always retained; only interaction variables are sub-sampled
- The `polynomial` parameter in `run_real_data_experiments` passes `degree` to `sklearn.PolynomialFeatures(degree=d, include_bias=False)`

Fig. 3 (appendix p. 20) is a 2×3 scatter plot comparing EM predictive performance against LOOCV across the three polynomial degrees. The color encodes the speed-up ratio (CV time / EM time) for the specific CV variant shown in that row.

**Note:** The `polynomial` parameter is currently passed to `PolynomialFeatures` at the experiment level. A planned future refactoring will push polynomial expansion into the estimator objects; this design deliberately defers that change.

## Notebook Cell Structure

Changes apply to both the **Preview** and **Full Experiment** sections of `experiments/real_data.ipynb`, following the existing pattern of one experiment cell + one table cell per setting.

### Preview section additions (after existing d=1 cells)

| Cell | Content |
|---|---|
| d=2 experiment | `results_d2 = run_real_data_experiments(problems, estimators, n_iterations=30, polynomial=2, seed=1, verbose=True)` |
| d=2 table | Same structure as existing d=1 table cell, reading from `results_d2` |
| d=3 experiment | `results_d3 = run_real_data_experiments(problems, estimators, n_iterations=30, polynomial=3, seed=1, verbose=True)` |
| d=3 table | Same structure, reading from `results_d3` |
| Figure | 2×3 matplotlib figure (see below); saved to `output/paper2023_figure3_preview.pdf` |

`estimators` is reused from the d=1 cell (same dict, no redefinition).

### Full Experiment section additions

Mirror of the above using `problems_full` and `estimators_full`, storing into `results_full_d2` and `results_full_d3`. Experiment cells (d=2 and d=3) are tagged `skip-execution`. The figure cell reads `results_full`, `results_full_d2`, `results_full_d3` and `problems_full`. Output saved to `output/paper2023_figure3.pdf`.

## Figure Design

A 2×3 matplotlib figure reproducing Fig. 3 from the paper appendix.

**Layout:**
- **Columns** (left → right): d=1, d=2, d=3
- **Rows** (top → bottom): CV_fix on y-axis; CV_glm on y-axis
- **x-axis** (all panels): EM R², negative values capped at 0
- **y-axis** (all panels): CV variant R², negative values capped at 0
- **Axis range**: [0, 1] × [0, 1]
- **Reference line**: dashed diagonal y=x in each panel

**Color encoding:**
- Color = log(speed-up ratio), where speed-up is CV-variant-specific:
  - Top row: `log(CV_fix_time / EM_time)` per dataset
  - Bottom row: `log(CV_glm_time / EM_time)` per dataset
- Sequential green colormap; shared colorbar across the full figure
- Rationale: all current speed-up ratios are positive; log scale compresses the large range

**Implementation:**
- Figure code defined inline in the figure cell — no new module-level functions
- Inputs: `results` (d=1), `results_d2`, `results_d3`, and `problems` (for dataset count)
- Each result list is a parallel list of per-dataset dicts with keys `r2`, `time` per estimator name

## Variable Naming Convention

| Variable | Contents |
|---|---|
| `results` / `results_full` | d=1 results (existing) |
| `results_d2` / `results_full_d2` | d=2 results (new) |
| `results_d3` / `results_full_d3` | d=3 results (new) |

## Out of Scope

- Pushing polynomial expansion into estimator objects (future refactoring)
- Large datasets (Twitter, Blog, CT Slices, TomsHw) — excluded from preview and full experiment cells; relevant only for the main text table
- Matching paper's exact p* values (tracked separately in the discrepancies issue doc)
