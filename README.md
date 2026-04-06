# <p align="center"> fastridge </p>
<h2 align="center"> Fast and Accurate Ridge Regression via Expectation Maximization </h2>

<p align="center">
    <a href="https://neurips.cc/virtual/2023/poster/72106"><img src="https://img.shields.io/badge/NeurIPS%202023-916ba4"></a>
    <a href="https://arxiv.org/abs/2310.18860"><img src="https://img.shields.io/badge/arXiv-2310.18860-b31b1b"></a>
    <a href="https://github.com/marioboley/fastridge/actions/workflows/ci.yml"><img src="https://github.com/marioboley/fastridge/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
</p>

<p align="center"> by Shu Yu Tew, Mario Boley, Daniel F. Schmidt </p>

---

The statistical performance of the ridge regression estimate for linear regression parameters fitted to a training dataset $\boldsymbol{X}\in\mathbb{R}^{n \times p}$, $\boldsymbol{y} \in \mathbb{R}^n$, i.e., 

$$
\hat{\boldsymbol{\beta}}_\alpha = \arg\min_{\boldsymbol{\beta} \in \mathbb{R}^p} \{\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\beta}\|^2 + \alpha\|\boldsymbol{\beta}\|^2\}
$$

strongly depends on the choice of the regularisation parameter $\alpha \in \mathbb{R}_+$. The commonly used approach to estimate the optimal value for this parameter is by leave-one-out cross-validation.

This package provides an alternative iterative algorithm based on the Bayesian formulation of ridge regression:
```math
\begin{aligned}
\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{\beta}, \sigma^2, \tau^2 &\sim \mathrm{N}(\boldsymbol{X}\boldsymbol{\beta}, \sigma^2 \boldsymbol{I}_n)\\
\boldsymbol{\beta} &\sim \mathrm{N}(0, \tau^{-2}\sigma^{-2}\boldsymbol{I}_p)\\
\sigma^2 &\sim \sigma^{-2}\,\mathrm{d}\sigma^2\\
\tau^2 &\sim \pi(\tau^2)\,\mathrm{d}\tau^2
\end{aligned}
```

In particular, the package implements an expectation maximisation (EM) approach that approximates the marginal posterior mode $\arg\max_{\sigma^2, \tau^2} p(\sigma^2, \tau^2 \mid \boldsymbol{X}, \boldsymbol{y})$ by iterating the equation
$$
\sigma^2_{t+1}, \tau^2_{t+1} = \arg\min_{\sigma^2, \tau^2} \mathbb{E}_{\boldsymbol{\beta} \mid \sigma^2_t, \tau^2_t}\!\left[-\log p(\boldsymbol{\beta}, \sigma^2, \tau^2)\right]
$$
until a convergence criterion is met.

## Usage
```python
import numpy as np
from fastridge import RidgeEM, RidgeLOOCV

# generate synthetic regression data
rng = np.random.default_rng(0)
beta = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
x_train = rng.standard_normal((50, 5))
y_train = x_train @ beta + 0.1 * rng.standard_normal(50)
x_test = rng.standard_normal((1000, 5))
y_test = x_test @ beta + 0.1 * rng.standard_normal(1000)

# fit using EM algorithm (no cross-validation required)
em = RidgeEM()
em.fit(x_train, y_train)
print(f'RidgeEM    train RMSE: {np.sqrt(np.mean((y_train - em.predict(x_train))**2)):.4f}')
print(f'RidgeEM    test RMSE:  {np.sqrt(np.mean((y_test - em.predict(x_test))**2)):.4f}')

# fit using fast LOOCV
loocv = RidgeLOOCV()
loocv.fit(x_train, y_train)
print(f'RidgeLOOCV train RMSE: {np.sqrt(np.mean((y_train - loocv.predict(x_train))**2)):.4f}')
print(f'RidgeLOOCV test RMSE:  {np.sqrt(np.mean((y_test - loocv.predict(x_test))**2)):.4f}')
```


## Package Installation

To install the package from pypi use

```bash skip
pip install fastridge
```

or to install directly from this repository use

```bash skip
pip install git+https://github.com/marioboley/fastridge.git
```

(`pip` or `pip3` depending on the local Python setup.)

## Project Setup

To alter the package or to run and modify the analysis code, run

```bash skip
pip3 install -r requirements.txt
pip3 install -e .
```

at the root of the repository after cloning.

The second step (local editable installation) is required so that `import fastridge` works for the analysis notebooks in subdirectories.

It is recommended to install package and dependencies into a dedicated virtual environment by running at the project root before the above steps:

```bash skip
python3 -m venv .venv
source .venv/bin/activate   # or: conda create/activate for Anaconda
```

To test the project setup, run the test suite:

```bash skip
pytest
```

## Citation
Should you find this repository helpful, please consider citing the associated paper:
```
@article{tew2023bayes,
  title={Bayes beats cross validation: Efficient and accurate ridge regression via expectation maximization},
  author={Tew, Shu Yu and Boley, Mario and Schmidt, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={19749--19768},
  year={2023}
}
```
