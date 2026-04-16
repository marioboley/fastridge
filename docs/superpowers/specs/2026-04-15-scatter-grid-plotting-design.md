# Scatter Grid Plotting Design

## Goal

Add two reusable functions to `experiments/plotting.py` that replace the duplicated `make_figure3` definitions in `real_data.ipynb` and `real_data_neurips2023.ipynb`. The functions have no dependency on `EmpiricalDataExperiment` — callers extract data from experiments themselves in notebook cells and pass plain arrays.

---

## Background: Current State

Both notebooks define an identical (or near-identical) `make_figure3` function locally that:
- Creates a 2×3 figure
- Draws clipped scatter plots with a globally normalised colour scale
- Saves to PDF if `output_path` is given

Problems with the current code:
- Duplicated in two notebooks
- Monolithic — no reusable unit for a single scatter panel
- Saving to PDF is the function's responsibility (violates the pattern in `plot_metric`)
- Hard-coded to 2×3 and to specific estimator names/indices

---

## Design

### Function 1: `scatter_clipped`

```python
def scatter_clipped(x, y, c, norm, cmap,
                    clip_min=-0.1, clip_max=1.0,
                    ref_lines=(0.0,), pad=0.03,
                    ax=None):
```

Draws one scatter panel on `ax` (defaults to `plt.gca()`).

**Parameters:**
- `x`, `y`: 1D array-like of floats — one point per problem
- `c`: 1D array-like of floats — colour values (e.g. speed-up ratios), same length as `x` and `y`
- `norm`: a `matplotlib.colors.Normalize` instance — applied to `c` to get colours; must be pre-computed by the caller over all data in the grid so the colour scale is globally consistent
- `cmap`: a matplotlib colormap
- `clip_min`: float (default `-0.1`) — values below this are clipped and drawn with dashed edges to signal they are out of range
- `clip_max`: float (default `1.0`) — values above this are clipped and drawn with dashed edges
- `ref_lines`: sequence of floats (default `(0.0,)`) — positions of grey horizontal and vertical reference lines (`color='0.8'`, `lw=0.5`)
- `pad`: float (default `0.03`) — margin added beyond `[clip_min, clip_max]` on all sides; axis limits are set to `[clip_min - pad, clip_max + pad]`
- `ax`: `matplotlib.axes.Axes` or `None` — defaults to `plt.gca()`

**Behaviour:**
- Points where neither coordinate is clipped are drawn with solid edges (`linewidths=0.6`)
- Points where either coordinate falls outside `[clip_min, clip_max]` are clipped to the nearest bound and drawn with dashed edges (`linewidths=0.8`, `linestyle='--'`)
- Draws a diagonal reference line from `(clip_min - pad, clip_min - pad)` to `(clip_max + pad, clip_max + pad)` in black dashed style, so it reaches exactly to the corners of the axes
- Draws grey horizontal and vertical lines at each position in `ref_lines`
- Sets axis limits to `[clip_min - pad, clip_max + pad]` on both axes
- Returns `ax`

**Does not set:** axis labels, ticks, titles, or colorbars — those are the caller's responsibility (usually `grid_with_colourbar`).

---

### Function 2: `grid_with_colourbar`

```python
def grid_with_colourbar(nrows, ncols, norm, cmap,
                        y_labels=None, col_titles=None,
                        x_labels='', cbar_label='',
                        cbar_fraction=0.56, figsize=None):
```

Creates a figure with an `nrows × ncols` grid of axes and a shared colorbar. Returns `(fig, axes)` where `axes` has shape `(nrows, ncols)`. The caller populates each axis however they like — `scatter_clipped` is one option but the function is not scatter-specific.

**Parameters:**
- `nrows`, `ncols`: grid dimensions
- `norm`: `matplotlib.colors.Normalize` — used to draw the colorbar; must be pre-computed by the caller
- `cmap`: matplotlib colormap
- `y_labels`: `str` or `list[str]` of length `nrows` or `None` — y-axis labels, applied to the leftmost column only. A single string is repeated for all rows (symmetric with `x_labels`)
- `col_titles`: `list[str]` of length `ncols` or `None` — column titles, applied to the top row only
- `x_labels`: `str` or `list[str]` of length `ncols` — x-axis label(s), applied to bottom-row axes only. A single string is repeated for all columns
- `cbar_label`: `str` — label for the colorbar
- `cbar_fraction`: float (default `0.56`) — height of the colorbar as a fraction of the axes area height (the vertical span set by `subplots_adjust`). The colorbar is centred on the axes midpoint. Default reproduces the original figure layout
- `figsize`: passed to `plt.subplots`; defaults to `(3 * ncols, 2.7 * nrows)`

**Behaviour:**
- Calls `plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)`
- Applies `fig.subplots_adjust` for tight layout with room for the colorbar
- Sets x-axis ticks to `[0.0, 0.5, 1.0]` on all bottom-row axes
- Sets y-axis ticks to `[0.0, 0.5, 1.0]` on all left-column axes
- Adds colorbar in a new axes to the right of the grid using `fig.add_axes`
- Does **not** call `scatter_clipped` — the caller does that after receiving `axes`
- Does **not** save — caller handles `fig.savefig(...)` if desired
- Returns `(fig, axes)`

---

## Notebook Pattern After Refactoring

```python
from plotting import scatter_clipped, grid_with_colourbar

# 1. extract data (notebook cell, experiment-specific)
em_idx  = exp_d1.est_names.index('EM')
glm_idx = exp_d1.est_names.index('CV_glm')
fix_idx = exp_d1.est_names.index('CV_fix')

# 2. compute global norm
all_c = [
    np.nanmean(exp.fitting_time_[:, i, 0, cv], axis=0)
    / np.nanmean(exp.fitting_time_[:, i, 0, em_idx], axis=0)
    for exp in [exp_d1, exp_d2, exp_d3]
    for cv in [glm_idx, fix_idx]
    for i in range(len(exp.problems))
]
norm = mcolors.Normalize(vmin=min(all_c), vmax=max(all_c))

# 3. create grid infrastructure
fig, axes = grid_with_colourbar(
    2, 3, norm, plt.cm.Greens,
    y_labels=['CV GLM Grid', 'CV Fixed Grid'],
    col_titles=['$d=1$', '$d=2$', '$d=3$'],
    x_labels='BayesEM $R^2$',
    cbar_label='speed-up ratio',
)

# 4. populate each axis
for col, exp in enumerate([exp_d1, exp_d2, exp_d3]):
    for row, cv_idx in enumerate([glm_idx, fix_idx]):
        x = np.nanmean(exp.prediction_r2_[:, :, 0, em_idx], axis=0)
        y = np.nanmean(exp.prediction_r2_[:, :, 0, cv_idx], axis=0)
        c = (np.nanmean(exp.fitting_time_[:, :, 0, cv_idx], axis=0)
             / np.nanmean(exp.fitting_time_[:, :, 0, em_idx], axis=0))
        scatter_clipped(x, y, c, norm, plt.cm.Greens, ax=axes[row, col])
```

To save: `fig.savefig('../output/realdata_r2_by_degree.pdf', bbox_inches='tight')`.

---

## Files

- Modify: `experiments/plotting.py` — add `scatter_clipped` and `grid_with_colourbar`
- Modify: `experiments/real_data.ipynb` — replace `make_figure3` definition and calls with `grid_with_colourbar` + `scatter_clipped` pattern
- Modify: `experiments/real_data_neurips2023.ipynb` — same replacement

---

## Future Possibilities

`scatter_clipped` could be registered as a method on `matplotlib.axes.Axes` via monkey-patching at import time of `plotting.py`:

```python
matplotlib.axes.Axes.scatter_clipped = scatter_clipped
```

This would allow `axes[row, col].scatter_clipped(x, y, c, norm, cmap)` in notebook cells. Matplotlib has no formal extension mechanism for adding Axes methods (the projection registration system is for full coordinate-system replacements, not utility methods), so monkey-patching is the only realistic path. Downsides: IDEs cannot infer the method without explicit type stubs, the patch is invisible unless `plotting` is imported, and it mutates a third-party class globally. Left as a future option for when the readability benefit outweighs the implicitness cost.
