---
name: Sparse Designs Notebook
description: Add sparse_designs.ipynb to experiments/ following the established pattern
type: project
---

# Sparse Designs Notebook Design

## Goal

Create `experiments/sparse_designs.ipynb` from `Analysis/Simulated_data/sparse_designs.ipynb`, following the established pattern of `double_asymptotic_trends.ipynb`. Includes a light `exp0` for pytest coverage and a full `exp1` tagged `skip-execution` that produces `output/paper2023_figure1.pdf`.

## Cell Structure

```
Cell 0  markdown          Problem description
Cell 1  markdown          ## Preview Experiment
Cell 2  code              exp0 — light params, seed=1, runs in pytest
Cell 3  code              exp0 visualization — runs in pytest
Cell 4  markdown          ## Full Experiment
Cell 5  code  skip-exec   exp1 — full params, seed=1
Cell 6  code  skip-exec   exp1 visualization + plt.savefig
```

Scratch cells from original (cells 2, 4, 5, 6, 7) are not copied.

## Cell 0: Markdown

Analyses the effect of increasing noise variance $\sigma^2$ on estimator performance for random sparse factor designs with $p=100$ and increasing sample size $n$. Produces Figure 1 of the paper.

For fixed $p=100$, a random prediction problem is drawn via:
$$
\beta \sim \mathcal{N}(0, I_p), \qquad x_i \sim \mathrm{Bernoulli}(1/p) \text{ i.i.d.}
$$
Then for a designated number of repetitions, training datasets are drawn via $n$ i.i.d. samples:
$$
y \mid x, \beta \sim \mathcal{N}(\beta^\top x,\, \sigma^2)
$$
Note: a single fixed $\beta$ is used across all repetitions per problem instance.

## Cell 2: exp0

```python
sigmas0 = [1.0, 3.0, 5.0]
ns0 = [100, 200, 400, 800, 1600]
10 reps, seed=1
estimators = [ridgeEM, ridgeCV_fixed]  # normalize=False
est_names = ['EM', 'LOOCV']
```

## Cell 3: exp0 visualization

4 metrics × 3 sigma values: `[parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time]`
x-axis log scale. Titles: `$\sigma = {sigmas0[j]}$`.

## Cell 5: exp1

```python
sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ns1 = [100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800]
100 reps, seed=1
```

## Cell 6: exp1 visualization

4 metrics × 5 sigma values (`sigmas_idx_plot = [1, 2, 3, 4, 5]`), `figsize=(12, 8)`.
`plt.savefig('../output/paper2023_figure1.pdf', dpi=600, bbox_inches="tight", pad_inches=0)`

## pytest.ini

Add `--nbmake experiments/sparse_designs.ipynb` to `addopts`.

## Out of Scope

- Changes to `problems.py`, `experiments.py`, `plotting.py`
- Any cleanup of `Analysis/Simulated_data/`
