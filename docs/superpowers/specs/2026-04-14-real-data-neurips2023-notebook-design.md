# real_data_neurips2023 Notebook Design

## Goal

Create `experiments/real_data_neurips2023.ipynb` to reproduce the NeurIPS 2023 paper Table 2 and Figure 2 results using the `NEURIPS2023` named collections and `RidgeEM(t2=False)` (matching the legacy `squareU=False` prior).

## Structure

All code cells are tagged `skip-execution`. No preview experiment. One section per polynomial degree, plus a final summary section.

### Section 1 — d=1

**Cell 1 (run):** First cell in the notebook. Imports everything needed for d=1 only.

```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV
from experiments import run_real_data_experiments
from problems import NEURIPS2023
from data import DATASETS

estimators = {
    'EM':     RidgeEM(t2=False),
    'CV_glm': RidgeLOOCV(alphas=100),
    'CV_fix': RidgeLOOCV(alphas=np.logspace(-10, 10, 100, base=10)),
}

problems_d1 = sorted(NEURIPS2023, key=lambda p: DATASETS[p.dataset]['n'])
results_d1 = run_real_data_experiments(
    problems_d1, estimators, n_iterations=100, seed=123, verbose=True)
print()
```

**Cell 2 (table):** First use of pandas. Builds and displays the d=1 results table.

```python
import pandas as pd

rows = []
for problem, result in zip(problems_d1, results_d1):
    rows.append({
        'dataset': problem.dataset,
        'n': DATASETS[problem.dataset]['n'],
        'p': DATASETS[problem.dataset]['p'],
        'd': 1,
        'R² EM':    round(result['EM']['r2'], 2),
        'R² CV_glm': round(result['CV_glm']['r2'], 2),
        'R² CV_fix': round(result['CV_fix']['r2'], 2),
        'T': round(result['CV_glm']['time'] / result['EM']['time'], 1),
    })
df_d1 = pd.DataFrame(rows)
df_d1
```

### Section 2 — d=2

**Cell 3 (run):** First import of `NEURIPS2023_D2`.

```python
from problems import NEURIPS2023_D2

problems_d2 = sorted(NEURIPS2023_D2, key=lambda p: DATASETS[p.dataset]['n'])
results_d2 = run_real_data_experiments(
    problems_d2, estimators, n_iterations=100, polynomial=2, seed=123, verbose=True)
print()
```

**Cell 4 (table):** Same row-building pattern as d=1, with `d=2`.

### Section 3 — d=3

**Cell 5 (run):** First import of `NEURIPS2023_D3`.

```python
from problems import NEURIPS2023_D3

problems_d3 = sorted(NEURIPS2023_D3, key=lambda p: DATASETS[p.dataset]['n'])
results_d3 = run_real_data_experiments(
    problems_d3, estimators, n_iterations=100, polynomial=3, seed=123, verbose=True)
print()
```

**Cell 6 (table):** Same pattern, `d=3`.

### Section 4 — Summary

**Cell 7 (combined table):** Concatenates df_d1, df_d2, df_d3; sorts by n then d; displays as DataFrame.

**Cell 8 (figure):** First import of `matplotlib`. Figure code replicated from `real_data.ipynb` (`make_figure3`-equivalent), saves to `../output/paper2023_figure2.pdf`.

**Cell 9 (CSV):** Writes combined table to `../output/paper2023_table2.csv`.

## Files

- Create: `experiments/real_data_neurips2023.ipynb`
