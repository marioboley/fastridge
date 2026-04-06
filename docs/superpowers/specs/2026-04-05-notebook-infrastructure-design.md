---
name: Notebook Infrastructure
description: Make notebooks runnable in the project venv and testable via pytest
type: project
---

# Notebook Infrastructure Design

## Goal

Make `Analysis/Simulated_data/increasing_p.ipynb` runnable in the project `.venv` via VSCode, and verifiable via `pytest` as part of the local and CI `project-test` workflow.

## Context

Notebooks are research/exploration code. The long-term plan is to move them to a separate project; this setup should therefore be minimal. Output regression testing is not needed — execution verification is sufficient.

## Approach

**Tool: `pytest-notebook`** with all output diffs ignored globally. Chosen over `nbmake` because it is comparable in simplicity but leaves the option open to enable output regression on specific cells in the future.

## Changes

### `requirements.txt`

Add:
- `ipykernel>=6.0` — required for VSCode to discover and use the `.venv` kernel (no manual registration needed)
- `pytest-notebook>=0.10` — pytest plugin for notebook execution testing

### `pytest.ini`

Add `--nb-test-files Analysis/Simulated_data/increasing_p.ipynb` to `addopts`.

Add configuration to ignore all cell outputs (execution-only mode):

```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nb-test-files Analysis/Simulated_data/increasing_p.ipynb
nb_diff_ignore =
    /cells/*/outputs
    /cells/*/execution_count
```

### Cell tagging (deferred)

No cell tags added upfront. After measuring local runtime, heavy cells (the `exp1.run()` invocation and its downstream plotting) may be tagged `skip-execution` if the full notebook run is too slow for routine testing.

### CI (`project-test` job)

No explicit change needed — `pytest.ini` is used automatically by the existing `pytest` call in `project-test`. The new packages in `requirements.txt` are picked up via `pip install -r requirements.txt`.

### `pytest-package.ini`

No change — notebook testing uses project dependencies, not package dependencies.

## Out of Scope

- Notebook output storage / regression baselines
- Testing notebooks in `Analysis/` subdirectories other than `increasing_p.ipynb`
- Any changes to `pytest-package.ini` or the `package-test` CI job
