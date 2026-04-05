# <p align="center"> fastridge: Fast and Accurate Ridge Regression via Expectation Maximization </p>

<p align="center">
    <a href="https://neurips.cc/virtual/2023/poster/72106"><img src="https://img.shields.io/badge/NeurIPS%202023-916ba4"></a>
    <a href="https://arxiv.org/abs/2310.18860"><img src="https://img.shields.io/badge/arXiv-2310.18860-b31b1b"></a>
    <a href="https://github.com/marioboley/fastridge/actions/workflows/ci.yml"><img src="https://github.com/marioboley/fastridge/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
</p>

<p align="center"> by Shu Yu Tew, Mario Boley, Daniel F. Schmidt </p>

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
