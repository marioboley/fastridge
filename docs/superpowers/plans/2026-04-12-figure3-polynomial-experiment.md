# Figure 3 Polynomial Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add d=2 and d=3 polynomial experiment cells to `experiments/real_data.ipynb` and produce a 2×3 scatter figure reproducing Fig. 3 from the paper appendix.

**Architecture:** Insert pairs of (experiment, table) cells for d=2 and d=3 in both preview and full sections, following the existing d=1 pattern. A shared `make_figure3` helper is defined in one cell and called from two figure cells (preview + full). All new polynomial experiment/table/figure cells are tagged `skip-execution` (polynomial expansion is too slow for CI). The make_figure3 definition cell is not skip-execution.

**Tech Stack:** Jupyter notebook (NotebookEdit tool), matplotlib, numpy, existing `run_real_data_experiments(problems, estimators, n_iterations, polynomial, seed, verbose)` from `experiments/experiments.py`.

---

## Notebook cell structure reference (current state)

```
cell-0  markdown  title
cell-1  markdown  ## Preview Experiment
cell-2  code      d=1 preview experiment  → results
cell-3  code      d=1 preview table
cell-4  markdown  ## Full Experiment ...
cell-5  code      d=1 full experiment     → results_full  [skip-execution]
cell-6  code      d=1 full table          [skip-execution]
```

Target state after this plan:

```
cell-2  d=1 preview experiment
cell-3  d=1 preview table
[new]   d=2 preview experiment     [skip-execution] → results_d2
[new]   d=2 preview table          [skip-execution]
[new]   d=3 preview experiment     [skip-execution] → results_d3
[new]   d=3 preview table          [skip-execution]
[new]   make_figure3 definition    (NOT skip-execution)
[new]   preview figure             [skip-execution]
cell-4  markdown header
cell-5  d=1 full experiment        [skip-execution]
cell-6  d=1 full table             [skip-execution]
[new]   d=2 full experiment        [skip-execution] → results_full_d2
[new]   d=2 full table             [skip-execution]
[new]   d=3 full experiment        [skip-execution] → results_full_d3
[new]   d=3 full table             [skip-execution]
[new]   full figure                [skip-execution]
```

## Skip-execution tagging helper

After each NotebookEdit insertion you must Read the notebook to find the auto-generated cell ID, then tag it. Use this Bash command (substitute the actual ID):

```bash
python3 -c "
import json
cell_id = 'REPLACE_ME'
with open('experiments/real_data.ipynb') as f: nb = json.load(f)
for cell in nb['cells']:
    if cell.get('id') == cell_id:
        cell.setdefault('metadata', {}).setdefault('tags', [])
        if 'skip-execution' not in cell['metadata']['tags']:
            cell['metadata']['tags'].append('skip-execution')
        break
with open('experiments/real_data.ipynb', 'w') as f: json.dump(nb, f, indent=1)
print('tagged', cell_id)
"
```

---

### Task 1: Preview d=2 experiment and table cells

**Files:** `experiments/real_data.ipynb` — insert 2 cells after cell-3

- [ ] **Step 1: Insert preview d=2 experiment cell after cell-3**

```python
results_d2 = run_real_data_experiments(problems, estimators, n_iterations=10, polynomial=2, seed=1, verbose=True)
print()
```

Use NotebookEdit with `cell_id="cell-3"`, `edit_mode="insert"`, `cell_type="code"`.

- [ ] **Step 2: Read notebook, find d=2 experiment cell ID, tag it skip-execution**

Read `experiments/real_data.ipynb`. Identify the cell immediately after cell-3. Run the tagging helper with its ID.

- [ ] **Step 3: Insert preview d=2 table cell after the experiment cell**

Insert after the cell ID found in Step 2:

```python
rows_d2 = []
for problem, data_result in zip(problems, results_d2):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = data_result['EM']['p']
    row['n_train']  = data_result['EM']['n_train']
    row['n:p']      = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_d2.append(row)
pd.DataFrame(rows_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 4: Read notebook, find d=2 table cell ID, tag it skip-execution**

- [ ] **Step 5: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add preview d=2 polynomial experiment and table cells"
```

---

### Task 2: Preview d=3 experiment and table cells

**Files:** `experiments/real_data.ipynb` — insert 2 cells after the d=2 table cell from Task 1

- [ ] **Step 1: Insert preview d=3 experiment cell after the d=2 table cell**

```python
results_d3 = run_real_data_experiments(problems, estimators, n_iterations=10, polynomial=3, seed=1, verbose=True)
print()
```

- [ ] **Step 2: Read notebook, find d=3 experiment cell ID, tag it skip-execution**

- [ ] **Step 3: Insert preview d=3 table cell after the experiment cell**

```python
rows_d3 = []
for problem, data_result in zip(problems, results_d3):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed-Up'] = cv_time / em_time
    row['p']        = data_result['EM']['p']
    row['n_train']  = data_result['EM']['n_train']
    row['n:p']      = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_d3.append(row)
pd.DataFrame(rows_d3).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 4: Read notebook, find d=3 table cell ID, tag it skip-execution**

- [ ] **Step 5: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add preview d=3 polynomial experiment and table cells"
```

---

### Task 3: make_figure3 helper and preview figure cell

**Files:** `experiments/real_data.ipynb` — insert 2 cells after the d=3 table cell from Task 2

The helper cell is **not** skip-execution (defines a function, no side effects). The preview figure cell **is** skip-execution (depends on results_d2, results_d3 which are from skipped cells).

- [ ] **Step 1: Insert make_figure3 definition cell after the d=3 preview table cell**

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_figure3(results_d1, results_d2, results_d3, output_path=None):
    """2x3 scatter of EM vs CV R² for d=1,2,3. Color = log(speed-up ratio).

    Top row: CV_fix on y-axis. Bottom row: CV_glm on y-axis.
    Speed-up for each panel is CV_variant_time / EM_time per dataset.
    Negative R² values are capped at 0. Shared colorbar across all panels.
    """
    results_list = [results_d1, results_d2, results_d3]
    cv_rows = [('CV_fix', 'CV Fixed Grid'), ('CV_glm', 'CV GLM')]

    all_log_su = [
        np.log(dr[cv]['time'] / dr['EM']['time'])
        for results in results_list
        for dr in results
        for cv, _ in cv_rows
    ]
    norm = mcolors.Normalize(vmin=min(all_log_su), vmax=max(all_log_su))
    cmap = plt.cm.Greens

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    fig.subplots_adjust(right=0.85, hspace=0.35, wspace=0.35)

    for col, results in enumerate(results_list):
        for row, (cv_name, cv_label) in enumerate(cv_rows):
            ax = axes[row, col]
            em_r2  = [max(0.0, dr['EM']['r2'])       for dr in results]
            cv_r2  = [max(0.0, dr[cv_name]['r2'])     for dr in results]
            log_su = [np.log(dr[cv_name]['time'] / dr['EM']['time']) for dr in results]
            ax.scatter(em_r2, cv_r2, c=cmap(norm(np.array(log_su))), s=50, zorder=3)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            if row == 1:
                ax.set_xlabel('BayesEM $R^2$')
            if col == 0:
                ax.set_ylabel(f'{cv_label}\n$R^2$')
            if row == 0:
                ax.set_title(f'd = {col + 1}')

    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label='log(Speed-Up)')

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
    return fig
```

Do **not** tag this cell skip-execution.

- [ ] **Step 2: Insert preview figure cell after the helper cell**

```python
make_figure3(results, results_d2, results_d3,
             output_path='../output/paper2023_figure3_preview.pdf')
```

- [ ] **Step 3: Read notebook, find preview figure cell ID, tag it skip-execution**

- [ ] **Step 4: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add make_figure3 helper and preview figure cell"
```

---

### Task 4: Full d=2 experiment and table cells

**Files:** `experiments/real_data.ipynb` — insert 2 cells after cell-6

Full experiment uses `problems_full`, `estimators_full`, `n_iterations=30`, `seed=123`.

- [ ] **Step 1: Insert full d=2 experiment cell after cell-6**

```python
results_full_d2 = run_real_data_experiments(problems_full, estimators_full, n_iterations=30, polynomial=2, seed=123, verbose=True)
print()
```

- [ ] **Step 2: Read notebook, find cell ID, tag skip-execution**

- [ ] **Step 3: Insert full d=2 table cell after the experiment cell**

```python
rows_full_d2 = []
for problem, data_result in zip(problems_full, results_full_d2):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = data_result['EM']['p']
    row['n_train']        = data_result['EM']['n_train']
    row['n:p']            = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_full_d2.append(row)
pd.DataFrame(rows_full_d2).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 4: Read notebook, find table cell ID, tag skip-execution**

- [ ] **Step 5: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add full d=2 polynomial experiment and table cells"
```

---

### Task 5: Full d=3 experiment and table cells

**Files:** `experiments/real_data.ipynb` — insert 2 cells after the full d=2 table cell

- [ ] **Step 1: Insert full d=3 experiment cell**

```python
results_full_d3 = run_real_data_experiments(problems_full, estimators_full, n_iterations=30, polynomial=3, seed=123, verbose=True)
print()
```

- [ ] **Step 2: Read notebook, find cell ID, tag skip-execution**

- [ ] **Step 3: Insert full d=3 table cell**

```python
rows_full_d3 = []
for problem, data_result in zip(problems_full, results_full_d3):
    em_time = data_result['EM']['time']
    cv_time = (data_result['CV_glm']['time'] + data_result['CV_fix']['time']) / 2
    row = {'dataset': problem.dataset, 'target': problem.target}
    row.update({est: data_result[est]['r2'] for est in data_result})
    row['Speed Up Ratio'] = cv_time / em_time
    row['p']              = data_result['EM']['p']
    row['n_train']        = data_result['EM']['n_train']
    row['n:p']            = data_result['EM']['n_train'] / data_result['EM']['p']
    rows_full_d3.append(row)
pd.DataFrame(rows_full_d3).sort_values('n_train', ascending=False).round(2)
```

- [ ] **Step 4: Read notebook, find table cell ID, tag skip-execution**

- [ ] **Step 5: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add full d=3 polynomial experiment and table cells"
```

---

### Task 6: Full figure cell

**Files:** `experiments/real_data.ipynb` — insert 1 cell after the full d=3 table cell

- [ ] **Step 1: Insert full figure cell**

```python
make_figure3(results_full, results_full_d2, results_full_d3,
             output_path='../output/paper2023_figure3.pdf')
```

- [ ] **Step 2: Read notebook, find cell ID, tag skip-execution**

- [ ] **Step 3: Commit**

```bash
git add experiments/real_data.ipynb
git commit -m "feat: add full figure cell for paper2023_figure3"
```

---

### Task 7: Run tests, push to dev

- [ ] **Step 1: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge && source .venv/bin/activate && pytest 2>&1 | tail -20
```

Expected: all tests pass. The make_figure3 definition cell executes in nbmake (defines a function only); all polynomial experiment/table/figure cells are skipped.

- [ ] **Step 2: Push to dev and wait for CI**

```bash
git push origin dev
```

- [ ] **Step 3: When CI passes, proceed to finishing-a-development-branch**

Use `superpowers:finishing-a-development-branch` to merge to main.
