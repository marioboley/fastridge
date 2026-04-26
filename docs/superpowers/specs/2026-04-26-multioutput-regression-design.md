# Multioutput Regression Design

## Motivation

`RidgeEM` and `RidgeLOOCV` currently only accept 1D target vectors. Supporting 2D target matrices
(multiple output regression) is needed to make the library useful as a drop-in for scikit-learn
pipelines and multi-output benchmarks without requiring users to loop externally. The design must
preserve exact backward compatibility for 1D inputs and must not introduce any overhead on the
single-target path.

---

## Approach

Extract the core EM iteration and LOOCV sweep into public free functions that operate on the
SVD-projected quantities `c` and `s`. The class wrappers `RidgeEM` and `RidgeLOOCV` handle
preprocessing (centering, normalization, SVD) and dispatch to the appropriate free function based
on `y.ndim`. This separates numerical kernel from pre/postprocessing, allows Numba JIT compilation
of the kernels without jitclass limitations, and gives the free functions a clean, testable public
interface.

---

## Free Functions

### `ridge_em_fit(c, s, n, p, epsilon=1e-8, t2=True)`

Single-target EM kernel. Operates entirely in SVD-projected space.

- `c`: 1D array of length `r = min(n, p)` — projected observations (`U^T y * s`)
- `s`: 1D array of singular values, length `r`
- `n`, `p`: original dataset dimensions
- `epsilon`: convergence threshold on relative RSS change
- `t2`: prior parametrization (Beta Prime on tau^2 if True, half-Cauchy on tau if False)

Returns `(tau, sigma, iterations)` as a plain tuple of Python scalars.

This function is the direct extraction of the current EM body in `RidgeEM.fit`, with no change to
the numerical algorithm.

### `ridge_em_fit_multi_target(c, s, n, p, epsilon=1e-8, t2=True)`

Per-target multi-output variant. Each target column runs an independent EM, producing an
independent `(tau_t, sigma_t)` pair.

- `c`: 2D array of shape `(r, n_targets)`
- `s`: 1D array of singular values, length `r`

Returns `(tau_arr, sigma_arr, iters_arr)` — 1D arrays of length `n_targets`.

Internally implemented as a loop over targets that calls `ridge_em_fit(c[:, t], ...)` for each
`t`. This structure:

1. Contains no EM logic of its own (no duplication).
2. Adds zero overhead to the 1D path (this function is never called for 1D `y`).
3. Is straightforwardly parallelisable with `prange` for Numba.
4. Leaves room for a future `ridge_em_fit_joint` (shared tau across targets, pooled M-step
   statistics) as a separate function with its own algorithm — not a parametric variant of this one.

### `ridge_loocv_sweep(c, s, y_sqnorm, n, alphas)`

Single-target LOOCV kernel, extracted from the current `RidgeLOOCV.fit` loop.

- Returns `loo_mse_` array of length `len(alphas)`.

Multi-target extension: `ridge_loocv_sweep_multi_target(c, s, y_sqnorm, n, alphas)` where `c` is
`(r, n_targets)` and `y_sqnorm` is `(n_targets,)`. Internally a loop over targets calling
`ridge_loocv_sweep`. Returns `loo_mse_` of shape `(n_targets, len(alphas))`.

---

## Class Wrapper: `RidgeEM`

Constructor signature is unchanged. The `fit` method:

1. Preprocesses `X` and `y` (center, normalize) — unchanged.
2. Computes thin SVD — unchanged.
3. Computes `c = U^T y * s` — for 2D `y`, this is `(U^T @ y) * s[:, None]`, giving `(r, n_targets)`.
4. Dispatches:
   - `y.ndim == 1` → `ridge_em_fit(c, s, n, p, epsilon, t2)` → scalar `tau`, `sigma`
   - `y.ndim == 2` → `ridge_em_fit_multi_target(c, s, n, p, epsilon, t2)` → 1D `tau`, `sigma`
5. Postprocesses into `coef_`, `intercept_`, `sigma_square_`, `tau_square_`, `alpha_`:
   - 1D `y`: shapes unchanged — `coef_` is `(p,)`, `alpha_` is scalar (backward compatible).
   - 2D `y`: `coef_` is `(n_targets, p)`, `alpha_` is `(n_targets,)`.
6. `trace` and `verbose` remain class-level concerns handled in the wrapper (not in free functions).

## Class Wrapper: `RidgeLOOCV`

Same dispatch pattern. `coef_` shape follows `y.ndim` as above. `alpha_` is scalar for 1D,
`(n_targets,)` for 2D.

---

## Backward Compatibility

- 1D `y` input: all output attribute shapes are identical to current behaviour.
- 2D `y` input with `n_targets == 1`: `coef_` is `(1, p)`, `alpha_` is `(1,)` — consistent with
  sklearn's `alpha_per_target=True` convention for `RidgeCV`.

---

## Docstrings and Public API

Both free functions are public module-level functions with docstrings stating their contract and a
doctest. They become part of the `fastridge` package's public interface and are importable directly:

```python
from fastridge import RidgeEM, RidgeLOOCV, ridge_em_fit, ridge_em_fit_multi_target
```

---

## Future Work

- **Joint-tau multi-target**: a separate `ridge_em_fit_joint` function where tau is shared across
  targets (pooled ESN/ESS in M-step). This is algorithmically distinct and belongs in a new
  function, not as a flag on `ridge_em_fit_multi_target`.
- **Numba JIT**: `ridge_em_fit` and `ridge_em_fit_multi_target` are designed to be JIT-compiled.
  The `trace` and `verbose` paths in the class wrapper cannot be JIT-compiled and are deliberately
  kept outside the free functions. If iteration history is needed in the future, a `max_iter`
  parameter with pre-allocated output arrays is the Numba-compatible approach.
- **Numerical M-step**: the non-closed-form (`closed_form_m_step=False`) path using `scipy.minimize`
  stays in the class wrapper or a separate non-JIT function — it is not part of the free-function
  kernel.
