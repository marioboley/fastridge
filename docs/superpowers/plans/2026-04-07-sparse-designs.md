# Sparse Designs Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/sparse_designs.ipynb` following the established pattern of `double_asymptotic_trends.ipynb`, with a pytest-covered preview experiment and a full experiment producing `output/paper2023_figure1.pdf`.

**Architecture:** New notebook created from scratch (not copied from original) using the refactored `experiments/` modules. Light `exp0` runs in pytest; heavy `exp1` is tagged `skip-execution`. `pytest.ini` is updated to include the new notebook.

**Tech Stack:** numpy, scipy, matplotlib, nbmake, pytest

---

### Task 1: Create sparse_designs.ipynb

**Files:**
- Create: `experiments/sparse_designs.ipynb`

- [ ] **Step 1: Create the notebook via Python script**

Run from the project root:

```bash
python3 - << 'EOF'
import json

with open('experiments/double_asymptotic_trends.ipynb') as f:
    ref = json.load(f)

def code_cell(source, tags=None):
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}
    if tags:
        cell["metadata"]["tags"] = tags
    return cell

def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

nb = dict(ref)
nb['cells'] = [

    markdown_cell(
        '# Sparse Factor Design\n\n'
        'Analyses the effect of increasing noise variance $\\sigma^2$ on estimator performance '
        'for random sparse factor designs with $p=100$ and increasing sample size $n$. '
        'Produces Figure 1 of the paper.\n\n'
        'For fixed $p=100$, a random prediction problem is drawn via:\n'
        '$$\n'
        '\\begin{aligned}\n'
        '\\beta &\\sim \\mathrm{N}(0, I_p)\\\\\n'
        'x_i &\\sim \\mathrm{Bernoulli}(1/p) \\text{ i.i.d.}, \\quad i = 1,\\ldots,p\n'
        '\\end{aligned}\n'
        '$$\n'
        'Then for a designated number of repetitions, training datasets are drawn via $n$ i.i.d. samples:\n'
        '$$\n'
        'y \\mid x, \\beta \\sim \\mathrm{N}(\\beta^\\top x,\\, \\sigma^2)\n'
        '$$\n'
        'Note: a single fixed $\\beta$ is used across all repetitions per problem instance.'
    ),

    markdown_cell('## Preview Experiment'),

    code_cell(
        'import numpy as np\n'
        'import problems\n'
        'from experiments import Experiment, parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time\n'
        'from matplotlib import pyplot as plt\n'
        'from plotting import plot_metrics\n'
        'from fastridge import RidgeEM, RidgeLOOCV\n'
        '\n'
        'sigmas0 = [1.0, 3.0, 5.0]\n'
        'rng = np.random.default_rng(1)\n'
        'probs0 = [problems.random_sparse_factor_problem(100, sigma_eps=sig, rng=rng) for sig in sigmas0]\n'
        'ns0 = [100, 200, 400, 800, 1600]\n'
        '\n'
        'ridgeEM = RidgeEM(fit_intercept=False, normalize=False)\n'
        'ridgeCV_fixed = RidgeLOOCV(alphas=np.logspace(-10, 10, 100, endpoint=True, base=10), fit_intercept=False, normalize=False)\n'
        '\n'
        'estimators = [ridgeEM, ridgeCV_fixed]\n'
        'est_names = [\'EM\', \'LOOCV\']\n'
        '\n'
        'exp0 = Experiment(probs0, estimators, ns0, 10, est_names, seed=1)\n'
        'exp0.run()'
    ),

    code_cell(
        'prob_idx0 = list(range(len(sigmas0)))\n'
        'fig, axs = plot_metrics(exp0, [parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time], prob_idx=prob_idx0, figsize=(9, 8))\n'
        'axs[0, 0].set_xscale(\'log\')\n'
        'for j in range(len(sigmas0)):\n'
        '    axs[0, j].set_title(f\'$\\\\sigma = {sigmas0[j]}$\')\n'
        'axs[0, 0].legend()\n'
        'plt.show()'
    ),

    markdown_cell('## Full Experiment'),

    code_cell(
        'sigmas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n'
        'rng1 = np.random.default_rng(1)\n'
        'probs1 = [problems.random_sparse_factor_problem(100, sigma_eps=sig, rng=rng1) for sig in sigmas]\n'
        'ns1 = [100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 4800, 6400, 9600, 12800]\n'
        '\n'
        'exp1 = Experiment(probs1, estimators, ns1, 100, est_names, seed=1)\n'
        'exp1.run()',
        tags=['skip-execution']
    ),

    code_cell(
        'sigmas_idx_plot = [1, 2, 3, 4, 5]\n'
        'fig, axs = plot_metrics(exp1, [parameter_mean_squared_error, regularization_parameter, number_of_iterations, fitting_time], prob_idx=sigmas_idx_plot, figsize=(12, 8))\n'
        'axs[0, 0].set_xscale(\'log\')\n'
        'for j in range(len(sigmas_idx_plot)):\n'
        '    axs[0, j].set_title(f\'$\\\\sigma = {sigmas[sigmas_idx_plot[j]]}$\')\n'
        'axs[0, 0].legend()\n'
        'plt.subplots_adjust(wspace=0, hspace=0)\n'
        'plt.savefig(\'../output/paper2023_figure1.pdf\', dpi=600, bbox_inches="tight", pad_inches=0)\n'
        'plt.show()',
        tags=['skip-execution']
    ),
]

with open('experiments/sparse_designs.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Done -', len(nb['cells']), 'cells')
for i, c in enumerate(nb['cells']):
    print(f'  Cell {i} ({c["cell_type"]}) tags={c.get("metadata",{}).get("tags",[])}')
EOF
```

Expected output:
```
Done - 7 cells
  Cell 0 (markdown) tags=[]
  Cell 1 (markdown) tags=[]
  Cell 2 (code) tags=[]
  Cell 3 (code) tags=[]
  Cell 4 (markdown) tags=[]
  Cell 5 (code) tags=['skip-execution']
  Cell 6 (code) tags=['skip-execution']
```

- [ ] **Step 2: Commit**

```bash
git add experiments/sparse_designs.ipynb
git commit -m "feat: add sparse_designs notebook to experiments/"
```

---

### Task 2: Update pytest.ini and verify

**Files:**
- Modify: `pytest.ini`

- [ ] **Step 1: Update pytest.ini**

Current:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake experiments/double_asymptotic_trends.ipynb
```

New:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake experiments/double_asymptotic_trends.ipynb --nbmake experiments/sparse_designs.ipynb
```

- [ ] **Step 2: Run pytest and verify**

```bash
source .venv/bin/activate
pytest
```

Expected: all tests pass including both notebooks. Runtime under ~30 seconds.

- [ ] **Step 3: Commit and push**

```bash
git add pytest.ini
git commit -m "feat: add sparse_designs to pytest nbmake"
git push origin dev
```
