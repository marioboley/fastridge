# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`fastridge` is a Python research library (NeurIPS 2023) implementing two fast ridge regression approaches:
- **`RidgeEM`**: Bayesian ridge regression via Expectation-Maximization — simultaneously estimates model coefficients and hyperparameters (`tau_square`, `sigma_square`) without cross-validation
- **`RidgeLOOCV`**: Fast Leave-One-Out Cross-Validation ridge — vectorized computation over an alpha grid using SVD

The entire library lives in a single module: [fastridge.py](fastridge.py).

## Setup

See the **Project Setup** section in `README.md` for environment setup instructions (adapt depending on system using venv or conda).

## Development Workflow

- Day-to-day work goes on the `dev` branch; `main` is kept green
- CI runs on push to both `dev` and `main` via `.github/workflows/ci.yml` — two jobs: `package-test` (package deps only, doctest) and `project-test` (full requirements, all tests)
- Run all tests via `pytest` before commit
- **Testing philosophy:** doctests are the primary test vehicle for all user-facing interfaces — they serve as executable documentation and are run in the `package-test` CI job. Add standard `pytest` tests for technical requirements that are of secondary or no concern to the user, or to keep docstrings concise by moving coverage that would otherwise bloat them.
- All local Python processs use dedicated virtual environment
- When finishing a development branch: push to `dev` first, wait for CI, then merge to `main`

## Research Reproducibility

Exact numerical reproduction of published paper results is a first-class concern. Currently this applies to the NeurIPS 2023 results (reproduced by `real_data_neurips2023.ipynb`); the same principle extends to any future publications. Designs that would reduce reproducibility should not be proposed in the first place — any deviation is a conscious, explicitly justified decision taken long before implementation, not something discovered at merge time.

The experiment infrastructure (`experiments/`) is a candidate for extraction into a standalone package. This makes principled, general solutions doubly important: shortcuts or repo-specific hacks will become technical debt the moment the code is published independently.

## Architecture

Both classes follow the scikit-learn estimator API (`fit(X, y)` / `predict(X)`) and share the same preprocessing pattern: optional intercept centering and feature normalization applied before SVD, then reversed when storing `coef_` and `intercept_`.

**Core computation pattern** (both classes):
1. Center/normalize `X` and `y` if configured
2. Compute thin SVD: `U, s, V^T = svd(X)`
3. Project `y` onto singular vectors: `c = U^T y * s`
4. Solve in the compressed space, avoiding explicit matrix inversion

**`RidgeEM` internals**: Iterates EM updates — E-step computes sufficient statistics `w` (expected squared coefficient norm) and `z` (expected RSS); M-step has a closed-form solution for `tau_square` and `sigma_square` under a Beta Prime prior on `tau_square` (`t2=True`, default) or a half-Cauchy prior on `tau` (`t2=False`). Convergence is checked on relative RSS change (`epsilon`). The `alpha_` attribute is a derived convenience alias for `1/tau_square`.

**`RidgeLOOCV` internals**: Vectorizes LOOCV over the alpha grid. Hat matrix diagonal `h` is computed without forming the full hat matrix: `h_i = sum_j (U_ij * s_j)^2 / (s_j^2 + alpha)`. LOO MSE is `mean((residual / (1 - h))^2)`.

## Experiments Module

`experiments/` contains the reproducible experiment infrastructure:

- **`problems.py`**: problem classes — `linear_problem` (synthetic, has `rvs()`) and `EmpiricalDataProblem` (real data, has `get_X_y()`). These are intended to be unified in a future refactor.
- **`experiments.py`**: experiment runners — `Experiment` (synthetic), `RidgePathExperiment`, `EmpiricalDataExperiment` (real data, stores results as numpy arrays).
- **`data.py`**: dataset registry (`DATASETS`) and retrieval (`get_dataset(name)`). Sources: `from_sklearn`, `from_ucimlrepo`, `from_url`, `from_zip`. Tier-1 only — raw DataFrames, no target selection.
- **Notebooks** (`real_data.ipynb`, `tutorial.ipynb`, `sparse_designs.ipynb`, `double_asymptotic_trends.ipynb`): run via `nbmake` in CI. Cells tagged `skip-execution` are skipped — used for expensive full-experiment cells.

**Superpowers docs** live in `docs/superpowers/` with three subdirectories: `specs/` (approved designs), `plans/` (implementation plans), `issues/` (open questions and investigations — some may eventually become specs, others may not). All files are date-stamped (`YYYY-MM-DD-<topic>.md`).

**Design documents** must open with a concise statement of the core motivation — the problem being solved and why it matters. All design decisions should be explicitly derived from that motivation; decisions that cannot be traced back to it are a signal the scope has drifted.

`Analysis/` contains legacy research notebooks. These are not part of CI and may depend on local files not in the repository.

**Editing notebooks:** Always use the `Read` tool (not Bash) before `NotebookEdit`, and specify `cell_id` from the Read output. The notebook must not be open in VSCode at the same time — the extension modifies the file on save, causing write conflicts. The `NotebookEdit` tool does not show a diff preview; announce the intended change in plain text before applying it.

**Notebook tables:** Prefer returning a DataFrame as the last expression in a cell (pandas/notebook native rendering) over `tabulate` or `print`. No extra dependency, interactive in JupyterLab.

**Notebook code style:** Notebooks represent documented scientific studies. Their focus in on presenting and analysing results of clearly defined experiments in sequential and highly readable manner. As such, they have a distinct codestyle that is very different from a Python module and rather mimics a small scientific paper or presentation. Every cell provides one comprehensible step of a scientific narrative that therefore needs to be compact and declarative. In other words, a cell should express *what* is being computed or displayed, not walk through *how* step by step. Prefer list comprehensions and expressive function calls over explicit loops and intermediate variables. Heavy logic belongs in module functions, not inline in cells. Still cells need to be self-contained to be understandable. Also, there should be no leading cells with bulk imports and function definitions. Instead, names are defined (or imported) along the way of the argument where they are first used. While names (e.g., of estimators or problem sets) are often imported from central modules according to the single source of truth principle, the corresponding values should typically be presented in the notebooks to facilitate the reader's understanding of the conducted experiment. These guidelines will often create tensions that can be resolved by extracting general purpose functions in plotting and other modules. You can actively propose such extractions. However, such functions must be truly general and reusable in various contexts.

**Single source of truth:** Every named constant, configuration value, or shared artefact must have exactly one definition that all consumers import. A value that is defined in two places (module and notebook, two modules, module and test) will silently diverge. If the only current consumer is a notebook or a script, the definition still belongs in whichever module owns that concept — consumers import, they do not redefine.

## Coding Conventions

**Class names:** UpperCamelCase / PascalCase (Python standard — each word capitalised, no separators). `linear_problem` in `experiments/problems.py` is a legacy exception — do not replicate it; new classes use PascalCase.

**Function and method names:** snake_case.

**Module names:** single word, no underscores (e.g. `data`, `problems`, `experiments`).

**Docstrings and comments:** Use only standard keyboard characters — no Unicode math or typographic symbols (e.g. avoid `×`, `∗`, `≤`, `²`). Use plain ASCII equivalents (`*`, `<=`, `**2`, etc.).

**Function specs:** Every function must have a docstring that states its contract unambiguously reflecting a deliberate design rationale. If a function's behaviour cannot be stated clearly in one or two sentences, treat that as a signal the design needs rethinking.

**Test granularity:** Prefer doctest over separate pytest as long as the tested behaviour is not an implementation detail and the docstring does not become too bloated. Further pytests are justified for important additional cases and behaviour that is hard to cover in a doctest (e.g. I/O, fixtures). Meaningful integration tests are preferred over fine-grained unit tests; especially for pytests.

**Imports:** Default to top-level imports. Inline imports (`from x import y` inside a function) are acceptable only for optional external dependencies that may not be installed in all environments (e.g. `ucimlrepo` is not a package dependency). When using an inline import, add a module-level comment documenting the optional dependency so it is visible without reading function bodies. Raise the question with the user if a new case arises — the rule may evolve.

**Helper functions:** At module level, prefer documented public helpers with a general contract over private helpers with an overly specific one — specificity shows in a function's contract as much as its name. A module-level helper that is genuinely reusable needs a docstring and doctest and needs to be placed in the proper module (often general utility module); a leading underscore is not a substitute for thinking about where a function belongs. Private methods within a class are fine when they exist to improve readability of public methods and cannot be adequalely expressed as general purpose helper.

**Code entropy:** Never silently introduce duplication or complexity. When the clean solution requires more thought or refactoring, raise it rather than taking the shortcut. Duplication and complexity are sometimes accepted after explicit discussion — but that is a conscious decision, not a default. Watch especially for: leaving a copy of extracted logic in the caller for "legacy" or "edge case" variants; adding a parallel implementation because integration is harder; growing a function's scope to avoid touching a second file; and adding alternative code paths (e.g. a `verbose` branch). Every additional branch is a maintenance surface — raise it as a design question rather than writing it silently.

## Production Capacity

Prefer changes that improve future editability and tooling over silent workarounds. When you encounter a limitation that has a clean fix — a format upgrade, a missing abstraction, a tool incompatibility — flag it and offer the fix rather than working around it silently. Similarly, when code being touched has grown complex enough to impede future changes (deeply nested logic, overloaded functions, unclear boundaries), note it and offer a targeted simplification. The goal is not just to solve the immediate problem but to leave the project in a state where the next task is easier. A concrete example: notebooks in this repo may be nbformat 4.2 (no cell IDs), which prevents `NotebookEdit` from locating cells by ID. The right response is to offer upgrading to 4.5 (`jupyter nbconvert --to notebook --inplace <file>`), not to silently fall back to JSON manipulation.

## Refactoring Rules

When integrating or refactoring existing code (e.g. moving analysis scripts into `experiments/`), every behaviour change must be explicitly identified, communicated to the user, and approved before implementation — even changes that appear neutral or beneficial.

**Before completing any refactoring step:**
- Audit every public method or function being replaced: do the new implementations preserve exact input/output behaviour?
- Identify all differences, including subtle ones such as normalization conventions, attribute naming, data flow through notebook session state, and global matplotlib rc state.
- Present each difference to the user with options (preserve or change) before proceeding.

This applies even to "obviously correct" fixes — the user decides whether to fix or preserve existing behaviour. Undiscovered behaviour changes in notebooks are particularly costly because they are only detectable by visual inspection of figure output.
