# Multioutput Regression Design

## Motivation

`RidgeEM` and `RidgeLOOCV` currently only accept 1D target vectors. Supporting 2D target matrices
(multiple output regression) is needed to make the library useful as a drop-in for scikit-learn
pipelines and multi-output benchmarks without requiring users to loop externally. The design must
preserve exact backward compatibility for 1D inputs and must not introduce any overhead on the
single-target path.

---

## EM Free Functions

The EM iteration is extracted into two public free functions that operate on SVD-projected
quantities `c` and `s`. Within these functions all computation is implicitly relative to the
orthogonalised input, so `beta` unambiguously refers to the coefficient vector in that space.
The class wrappers handle preprocessing (centering, normalization, SVD) and delegate to these
functions. This separation allows future Numba JIT compilation of the kernels without jitclass
limitations and gives the functions a clean, independently testable public interface.

There are no analogous free functions for `RidgeLOOCV`: the LOOCV sweep is already a vectorized
numpy operation with no Numba motivation, and multi-target support is handled inline in the class
wrapper.

### `em_max_marginal_posterior_ridge(c, s, n, p, epsilon=1e-8, t2=True, trace=False)`

Single-target EM kernel. All computation is in the space of the orthogonalised input.

- `c`: 1D array of length `r = min(n, p)` — projected observations (`U^T y * s`)
- `s`: 1D array of singular values, length `r`
- `n`, `p`: original dataset dimensions
- `epsilon`: convergence threshold on relative RSS change
- `t2`: prior parametrization (Beta Prime on tau^2 if True, half-Cauchy on tau if False)
- `trace`: if True, additionally returns per-iteration history (see below)

Returns `(sigma_square, tau_square, beta, n_iter)`:
- `sigma_square`, `tau_square`: converged hyperparameter estimates (Python scalars)
- `beta`: coefficient vector `c / (s**2 + 1/tau_square)` at convergence (1D array, length `r`)
- `n_iter`: number of EM iterations (int)

When `trace=True`, additionally returns `(sigma_hist, tau_hist, beta_hist)` — lists of values at
each iteration. This path is Python-only. The class wrapper stores `sigma_hist` and `tau_hist`
directly as `sigma_squares_` and `tau_squares_`, and reconstructs `coefs_` from `beta_hist` by
rotating each entry back to the original feature space and applying output rescaling.

This function is a direct extraction of the current EM body in `RidgeEM.fit`, with no change to
the numerical algorithm.

### `em_max_marginal_posterior_ridge_multi_target(c, s, n, p, epsilon=1e-8, t2=True)`

Per-target multi-output variant. Each target column runs an independent EM, producing independent
`sigma_square_t` and `tau_square_t` estimates.

- `c`: 2D array of shape `(r, n_targets)`
- `s`: 1D array of singular values, length `r`

Returns `(sigma_arr, tau_arr, beta_mat, n_iter_arr)`:
- `sigma_arr`, `tau_arr`, `n_iter_arr`: 1D arrays of length `n_targets`
- `beta_mat`: 2D array of shape `(r, n_targets)`

Internally a loop over targets calling
`em_max_marginal_posterior_ridge(c[:, t], s, n, p, epsilon, t2)` for each `t`. This structure
contains no EM logic of its own (no duplication), adds zero overhead to the 1D path (this
function is never called for 1D `y`), and is straightforwardly parallelisable with `prange` for
Numba. Trace is not supported in the multi-target variant.

A future `em_max_marginal_posterior_ridge_joint` (shared tau across targets, pooled M-step
statistics) is a distinct algorithm and will be a separate function — not a parametric variant of
this one.

---

## Class Wrappers

### `RidgeEM`

Constructor signature is unchanged. The `fit` method:

1. Preprocesses `X` and `y` (center, normalize) — unchanged for 1D `y`. For 2D `y`,
   `a_y` and `b_y` become 1D arrays of length `n_targets` (per-target mean and std).
2. Computes thin SVD — unchanged.
3. Computes projected observations `c`:
   - 1D `y`: `c = U.T @ y * s` (length `r`) — unchanged
   - 2D `y`: `c = (U.T @ y) * s[:, None]` (shape `(r, n_targets)`)
4. Dispatches:
   - `y.ndim == 1` → `em_max_marginal_posterior_ridge(c, s, n, p, epsilon, t2, trace=self.trace)`
   - `y.ndim == 2` → `em_max_marginal_posterior_ridge_multi_target(c, s, n, p, epsilon, t2)`
5. Postprocesses into fitted attributes:
   - 1D `y` (unchanged shapes): `coef_` is `(p,)`, `alpha_` is scalar, `sigma_square_` is scalar.
   - 2D `y`: `coef_` is `(n_targets, p)`, `alpha_` is `(n_targets,)`, `sigma_square_` is
     `(n_targets,)`.
6. When `self.trace` is True (1D `y` only):
   - `sigma_squares_` and `tau_squares_` are taken directly from the returned `sigma_hist` and
     `tau_hist`.
   - `coefs_` is reconstructed by rotating each `beta_hist[t]` back to the original feature space
     (`v_trans.T @ beta_t`) and rescaling (`* b_y / b_x`).
   - `trace=True` with 2D `y` is not supported; the wrapper raises `ValueError`.

### `RidgeLOOCV`

No free functions. Multi-target support is added inline in `fit`: when `y.ndim == 2`, compute
`c` as above, then loop over targets to independently select `alpha_t` via LOOCV. Returns
`coef_` of shape `(n_targets, p)` and `alpha_` of shape `(n_targets,)`. The 1D `y` code path is
unmodified and incurs zero overhead. Single alpha shared across targets is noted as a future
extension analogous to joint-tau in EM.

---

## Backward Compatibility

1D `y` input: all output attribute shapes and values are identical to current behaviour. No
existing code path is altered.

2D `y` input with `n_targets == 1`: `coef_` is `(1, p)`, `alpha_` is `(1,)` — consistent with
scikit-learn's `RidgeCV(alpha_per_target=True)` convention.

---

## Public API

Both free functions are module-level public functions with docstrings stating their contract and a
doctest, importable directly:

```python
from fastridge import (RidgeEM, RidgeLOOCV,
                       em_max_marginal_posterior_ridge,
                       em_max_marginal_posterior_ridge_multi_target)
```

---

## Future Work

- **Joint-tau multi-target**: `em_max_marginal_posterior_ridge_joint` where tau is shared across
  targets (pooled ESN/ESS in M-step). Algorithmically distinct from the per-target variant; the
  class wrapper would dispatch to it via a new parameter (e.g., `shared_tau=False`).
- **Single alpha for LOOCV multi-target**: analogous to joint-tau; shared alpha selected to
  minimise combined LOOCV loss across targets.
- **Numba JIT**: the free functions can be annotated with `@njit`. This requires (a) factoring out
  or reimplementing the numerical M-step (`scipy.minimize`) in a JIT-compatible form, and (b)
  introducing a `max_iter` parameter to pre-allocate the trace matrix instead of appending to
  dynamic lists.
