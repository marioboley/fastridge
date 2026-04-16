# Scatter Grid Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scatter_clipped` and `grid_with_colourbar` to `experiments/plotting.py`, then replace the duplicated `make_figure3` definitions in both real-data notebooks with the new functions.

**Architecture:** Two new functions in `plotting.py` (no experiment dependency). Notebooks keep all data-extraction logic inline; the functions handle only visual layout. `real_data.ipynb` is done first and reviewed before touching the neurips notebook. The neurips notebook is not in pytest/nbmake so it is only tested by manual inspection.

**Tech Stack:** `matplotlib`, `numpy`, `NotebookEdit` for notebook edits (cell_id required — always Read before editing; close notebooks in VSCode first).

**Key constraint:** `real_data.ipynb` cells 10 and 11 are NOT tagged `skip-execution` and are run by `pytest --nbmake`. After refactoring they must execute without error. Per project policy, get user approval on figure output (cell 11) before running pytest.

---

## Files

- Modify: `experiments/plotting.py` — add `scatter_clipped` and `grid_with_colourbar`
- Modify: `experiments/real_data.ipynb` — cells `65e15066` (10), `0fccfcde` (11), `07437ade` (21), `f36b7874` (28)
- Modify: `experiments/real_data_neurips2023.ipynb` — cell `a013`

---

### Task 1: Implement `scatter_clipped`

**Files:** `experiments/plotting.py`

- [ ] **Step 1: Add `scatter_clipped` to `plotting.py`**

Append the following after the existing `plot_metrics` function (before `plot_lambda_risks`):

```python
_SCATTER_PAD = 0.03


def scatter_clipped(x, y, c, norm, cmap, clip_min=-0.1, clip_max=1.0,
                    ref_lines=(0.0,), pad=_SCATTER_PAD, ax=None):
    """Scatter plot with out-of-range points clipped and marked with dashed edges.

    Points where either coordinate falls outside [clip_min, clip_max] are clipped
    to the nearest bound and drawn with dashed edges. A diagonal reference line
    runs from corner to corner of the axes. Operates on ax (default: plt.gca()).

    Parameters
    ----------
    x, y : array-like of float
        Coordinates, one entry per point.
    c : array-like of float
        Colour values, same length as x and y. Mapped via norm and cmap.
    norm : matplotlib.colors.Normalize
        Pre-computed normalisation — must cover all data in the grid for a
        globally consistent colour scale.
    cmap : matplotlib colormap
    clip_min : float, default -0.1
    clip_max : float, default 1.0
    ref_lines : sequence of float, default (0.0,)
        Positions of grey horizontal and vertical reference lines.
    pad : float, default 0.03
        Margin added beyond [clip_min, clip_max] on all sides.
    ax : matplotlib.axes.Axes or None
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)

    lim = [clip_min - pad, clip_max + pad]
    clipped = (x < clip_min) | (x > clip_max) | (y < clip_min) | (y > clip_max)
    x_disp = np.clip(x, clip_min, clip_max)
    y_disp = np.clip(y, clip_min, clip_max)
    colors = cmap(norm(c))

    if (~clipped).any():
        ax.scatter(x_disp[~clipped], y_disp[~clipped], c=colors[~clipped],
                   s=50, zorder=3, edgecolors='k', linewidths=0.6)
    if clipped.any():
        sc = ax.scatter(x_disp[clipped], y_disp[clipped], c=colors[clipped],
                        s=50, zorder=4, edgecolors='k', linewidths=0.8)
        sc.set_linestyle('--')

    ax.plot(lim, lim, 'k--', lw=0.8, zorder=2)
    for ref in ref_lines:
        ax.axhline(ref, color='0.8', lw=0.5, zorder=1)
        ax.axvline(ref, color='0.8', lw=0.5, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    return ax
```

- [ ] **Step 2: Smoke-test `scatter_clipped`**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import sys
sys.path.insert(0, 'experiments')
from plotting import scatter_clipped

rng = np.random.default_rng(0)
x = rng.uniform(-0.5, 1.2, 20)
y = rng.uniform(-0.5, 1.2, 20)
c = rng.uniform(1, 10, 20)
norm = mcolors.Normalize(vmin=1, vmax=10)
import matplotlib.pyplot as plt
ax = scatter_clipped(x, y, c, norm, plt.cm.Greens)
assert ax is not None
# check axis limits
xlim = ax.get_xlim()
assert abs(xlim[0] - (-0.1 - 0.03)) < 1e-9
assert abs(xlim[1] - (1.0 + 0.03)) < 1e-9
print('scatter_clipped OK')
"
```

Expected: `scatter_clipped OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && git add experiments/plotting.py && git commit -m "feat: add scatter_clipped to plotting.py"
```

---

### Task 2: Implement `grid_with_colourbar`

**Files:** `experiments/plotting.py`

- [ ] **Step 1: Add `grid_with_colourbar` to `plotting.py`**

Append the following immediately after `scatter_clipped`:

```python
def grid_with_colourbar(nrows, ncols, norm, cmap,
                        y_labels=None, col_titles=None,
                        x_labels='', cbar_label='',
                        cbar_fraction=0.56, figsize=None):
    """Create an nrows x ncols grid of axes with a shared colorbar on the right.

    Returns (fig, axes) where axes has shape (nrows, ncols). The caller populates
    each axis — scatter_clipped is one option but the function is not scatter-specific.

    Parameters
    ----------
    nrows, ncols : int
    norm : matplotlib.colors.Normalize
        Used to draw the colorbar; pre-compute over all data before calling.
    cmap : matplotlib colormap
    y_labels : str or list of str of length nrows, or None
        Y-axis label(s) applied to the leftmost column only.
        A single string is repeated for all rows (symmetric with x_labels).
    col_titles : list of str, length ncols, or None
        Column titles applied to the top row only.
    x_labels : str or list of str of length ncols
        X-axis label(s) applied to bottom-row axes. Single string applies to all.
    cbar_label : str
        Label for the colorbar.
    cbar_fraction : float, default 0.56
        Height of the colorbar as a fraction of the axes area height
        (the vertical span set by subplots_adjust). Colorbar is centred on the
        axes midpoint. Default reproduces the original figure layout.
    figsize : tuple or None
        Passed to plt.subplots. Default: (3 * ncols, 2.7 * nrows).
    """
    if figsize is None:
        figsize = (3 * ncols, 2.7 * nrows)

    bottom_adj, top_adj = 0.11, 0.93
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=bottom_adj, top=top_adj,
                        hspace=0.06, wspace=0.04)

    if col_titles:
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title)

    y_labels_list = ([y_labels] * nrows if isinstance(y_labels, str)
                     else (y_labels or []))
    for i, label in enumerate(y_labels_list):
        axes[i, 0].set_ylabel(label)

    x_labels_list = [x_labels] * ncols if isinstance(x_labels, str) else list(x_labels)
    for j, label in enumerate(x_labels_list):
        axes[nrows - 1, j].set_xlabel(label)

    for j in range(ncols):
        axes[nrows - 1, j].set_xticks([0.0, 0.5, 1.0])
    for i in range(nrows):
        axes[i, 0].set_yticks([0.0, 0.5, 1.0])

    cbar_h = cbar_fraction * (top_adj - bottom_adj)
    cbar_b = (top_adj + bottom_adj) / 2 - cbar_h / 2
    cbar_ax = fig.add_axes([0.86, cbar_b, 0.02, cbar_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label(cbar_label, rotation=90, labelpad=-30)

    return fig, axes
```

- [ ] **Step 2: Smoke-test `grid_with_colourbar`**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && python -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import sys
sys.path.insert(0, 'experiments')
from plotting import grid_with_colourbar

norm = mcolors.Normalize(vmin=1, vmax=10)
fig, axes = grid_with_colourbar(2, 3, norm, __import__('matplotlib.pyplot', fromlist=['cm']).cm.Greens,
                                y_labels=['Row A', 'Row B'],
                                col_titles=['C1', 'C2', 'C3'],
                                x_labels='x axis',
                                cbar_label='ratio')
assert axes.shape == (2, 3)
assert axes[1, 0].get_xlabel() == 'x axis'
assert axes[0, 0].get_ylabel() == 'Row A'
assert axes[0, 0].get_title() == 'C1'
print('grid_with_colourbar OK')
"
```

Expected: `grid_with_colourbar OK`

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -10
```

Expected: all tests pass (plotting.py is not in pytest doctest list; nbmake notebooks are unaffected so far).

- [ ] **Step 4: Commit**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && git add experiments/plotting.py && git commit -m "feat: add grid_with_colourbar to plotting.py"
```

---

### Task 3: Update `real_data.ipynb`

**Files:** `experiments/real_data.ipynb`

Make sure the notebook is **not open in VSCode** before editing. Use `Read` before each `NotebookEdit`.

- [ ] **Step 1: Replace cell `65e15066` (imports + `make_figure3` definition) with imports only**

New content — drops the `make_figure3` definition, keeps the imports and adds the new import:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotting import scatter_clipped, grid_with_colourbar
```

- [ ] **Step 2: Replace cell `0fccfcde` (preview figure call) with declarative pattern**

New content:

```python
em_idx, glm_idx, fix_idx = (exp.est_names.index(n) for n in ['EM', 'CV_glm', 'CV_fix'])
cv_pairs = [(glm_idx, 'CV GLM Grid'), (fix_idx, 'CV Fixed Grid')]
exps_by_col = [exp, exp_d2, exp_d3]

su_all = [np.nanmean(e.fitting_time_[:, i, 0, cv]) / np.nanmean(e.fitting_time_[:, i, 0, em_idx])
          for e in exps_by_col for cv, _ in cv_pairs for i in range(len(e.problems))]
norm = mcolors.Normalize(vmin=min(su_all), vmax=max(su_all))

fig, axes = grid_with_colourbar(2, 3, norm, plt.cm.Greens,
                                y_labels=['CV GLM Grid $R^2$', 'CV Fixed Grid $R^2$'],
                                col_titles=['$d=1$', '$d=2$', '$d=3$'],
                                x_labels='BayesEM $R^2$', cbar_label='speed-up ratio')

for col, e in enumerate(exps_by_col):
    for row, (cv_idx, _) in enumerate(cv_pairs):
        scatter_clipped(np.nanmean(e.prediction_r2_[:, :, 0, em_idx], axis=0),
                        np.nanmean(e.prediction_r2_[:, :, 0, cv_idx], axis=0),
                        np.nanmean(e.fitting_time_[:, :, 0, cv_idx], axis=0)
                        / np.nanmean(e.fitting_time_[:, :, 0, em_idx], axis=0),
                        norm, plt.cm.Greens, ax=axes[row, col])
```

- [ ] **Step 3: Run the preview cells manually to check figure output**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=300 --output /tmp/real_data_preview_check.ipynb experiments/real_data.ipynb 2>&1 | tail -5
```

Then open `/tmp/real_data_preview_check.ipynb` and **visually inspect the figure produced by the refactored cell 11**. It should match the previous figure (2×3 scatter, green colour scale, diagonal reaching corners — the corners fix is the only visual change from before). Get user approval before continuing.

- [ ] **Step 4: Replace cell `07437ade` (full experiment figure, skip-execution)**

New content:

```python
em_idx, glm_idx, fix_idx = (exp_full.est_names.index(n) for n in ['EM', 'CV_glm', 'CV_fix'])
cv_pairs = [(glm_idx, 'CV GLM Grid'), (fix_idx, 'CV Fixed Grid')]
exps_by_col = [exp_full, exp_full_d2, exp_full_d3]

su_all = [np.nanmean(e.fitting_time_[:, i, 0, cv]) / np.nanmean(e.fitting_time_[:, i, 0, em_idx])
          for e in exps_by_col for cv, _ in cv_pairs for i in range(len(e.problems))]
norm = mcolors.Normalize(vmin=min(su_all), vmax=max(su_all))

fig, axes = grid_with_colourbar(2, 3, norm, plt.cm.Greens,
                                y_labels=['CV GLM Grid $R^2$', 'CV Fixed Grid $R^2$'],
                                col_titles=['$d=1$', '$d=2$', '$d=3$'],
                                x_labels='BayesEM $R^2$', cbar_label='speed-up ratio')

for col, e in enumerate(exps_by_col):
    for row, (cv_idx, _) in enumerate(cv_pairs):
        scatter_clipped(np.nanmean(e.prediction_r2_[:, :, 0, em_idx], axis=0),
                        np.nanmean(e.prediction_r2_[:, :, 0, cv_idx], axis=0),
                        np.nanmean(e.fitting_time_[:, :, 0, cv_idx], axis=0)
                        / np.nanmean(e.fitting_time_[:, :, 0, em_idx], axis=0),
                        norm, plt.cm.Greens, ax=axes[row, col])
```

- [ ] **Step 5: Replace cell `f36b7874` (combined large + small figure with save, skip-execution)**

New content — note `col_exps` is a list of lists; arrays are concatenated across experiments per column:

```python
em_idx, glm_idx, fix_idx = (exp_full.est_names.index(n) for n in ['EM', 'CV_glm', 'CV_fix'])
cv_pairs = [(glm_idx, 'CV GLM Grid'), (fix_idx, 'CV Fixed Grid')]
col_exps = [[exp_full, exp_large], [exp_full_d2, exp_large_d2], [exp_full_d3]]

su_all = [np.nanmean(e.fitting_time_[:, i, 0, cv]) / np.nanmean(e.fitting_time_[:, i, 0, em_idx])
          for exps in col_exps for e in exps for cv, _ in cv_pairs for i in range(len(e.problems))]
norm = mcolors.Normalize(vmin=min(su_all), vmax=max(su_all))

fig, axes = grid_with_colourbar(2, 3, norm, plt.cm.Greens,
                                y_labels=['CV GLM Grid $R^2$', 'CV Fixed Grid $R^2$'],
                                col_titles=['$d=1$', '$d=2$', '$d=3$'],
                                x_labels='BayesEM $R^2$', cbar_label='speed-up ratio')

for col, exps in enumerate(col_exps):
    for row, (cv_idx, _) in enumerate(cv_pairs):
        scatter_clipped(
            np.concatenate([np.nanmean(e.prediction_r2_[:, :, 0, em_idx], axis=0) for e in exps]),
            np.concatenate([np.nanmean(e.prediction_r2_[:, :, 0, cv_idx], axis=0) for e in exps]),
            np.concatenate([np.nanmean(e.fitting_time_[:, :, 0, cv_idx], axis=0)
                            / np.nanmean(e.fitting_time_[:, :, 0, em_idx], axis=0) for e in exps]),
            norm, plt.cm.Greens, ax=axes[row, col])

fig.savefig('../output/realdata_r2_by_degree.pdf', bbox_inches='tight')
```

- [ ] **Step 6: Run full test suite (includes nbmake on `real_data.ipynb`)**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -10
```

Expected: all tests pass. The nbmake run of `real_data.ipynb` will execute the refactored cells 10 and 11; the skip-execution cells (21, 28) are skipped.

- [ ] **Step 7: Commit**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && git add experiments/real_data.ipynb && git commit -m "refactor: replace make_figure3 in real_data.ipynb with scatter_clipped/grid_with_colourbar"
```

---

### Task 4: Update `real_data_neurips2023.ipynb`

**Files:** `experiments/real_data_neurips2023.ipynb`

This notebook is not in `pytest.ini` nbmake, so there is no automated test. Changes are verified by inspection only. Make sure the notebook is **not open in VSCode** before editing.

- [ ] **Step 1: Read current cell `a013` to confirm structure**

Read `experiments/real_data_neurips2023.ipynb` and confirm cell `a013` is at index 13, tagged `skip-execution`, and contains the `make_figure3` definition and call.

- [ ] **Step 2: Replace cell `a013` with imports + declarative pattern + save**

New content — the `exp_d1`, `exp_d2`, `exp_d3` variables are set in preceding skip-execution cells; `np` is imported in cell `a003`:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from plotting import scatter_clipped, grid_with_colourbar

em_idx, glm_idx, fix_idx = (exp_d1.est_names.index(n) for n in ['EM', 'CV_glm', 'CV_fix'])
cv_pairs = [(glm_idx, 'CV GLM Grid'), (fix_idx, 'CV Fixed Grid')]
exps_by_col = [exp_d1, exp_d2, exp_d3]

su_all = [np.nanmean(e.fitting_time_[:, i, 0, cv]) / np.nanmean(e.fitting_time_[:, i, 0, em_idx])
          for e in exps_by_col for cv, _ in cv_pairs for i in range(len(e.problems))]
norm = mcolors.Normalize(vmin=min(su_all), vmax=max(su_all))

fig, axes = grid_with_colourbar(2, 3, norm, plt.cm.Greens,
                                y_labels=['CV GLM Grid $R^2$', 'CV Fixed Grid $R^2$'],
                                col_titles=['$d=1$', '$d=2$', '$d=3$'],
                                x_labels='BayesEM $R^2$', cbar_label='speed-up ratio')

for col, e in enumerate(exps_by_col):
    for row, (cv_idx, _) in enumerate(cv_pairs):
        scatter_clipped(np.nanmean(e.prediction_r2_[:, :, 0, em_idx], axis=0),
                        np.nanmean(e.prediction_r2_[:, :, 0, cv_idx], axis=0),
                        np.nanmean(e.fitting_time_[:, :, 0, cv_idx], axis=0)
                        / np.nanmean(e.fitting_time_[:, :, 0, em_idx], axis=0),
                        norm, plt.cm.Greens, ax=axes[row, col])

fig.savefig('../output/paper2023_figure3.pdf', bbox_inches='tight')
```

- [ ] **Step 3: Run pytest to confirm nothing else broke**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -10
```

Expected: all tests pass (neurips notebook is not in nbmake).

- [ ] **Step 4: Commit**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && git add experiments/real_data_neurips2023.ipynb && git commit -m "refactor: replace make_figure3 in real_data_neurips2023.ipynb with scatter_clipped/grid_with_colourbar"
```
