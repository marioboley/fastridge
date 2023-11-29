# <p align="center"> Bayes beats Cross Validation: Fast and Accurate Ridge Regression via Expectation Maximization </p>
 
<p align="center"> by Shu Yu Tew, Mario Boley, Daniel F. Schmidt </p>

## Usage
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from fastridge import RidgeEM, RidgeLOOCV

# load data
diabetes = load_diabetes()
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)

# fitting ridge regression using the proposed EM algorithm
ridgeEM = RidgeEM(trace = True)
ridgeEM.fit(x_train, y_train)

# fitting ridge regression using the fast implementation of LOOCV
N = 400
alphas =np.logspace(-15, 10, N, endpoint=True, base=10)
ridgeCV = RidgeLOOCV(alphas=alphas, fit_intercept=True, normalize=True)
ridgeCV.fit(x_train, y_train)

# return the coefficients of the respective regression models (refer to the code documentation for more details on additional return values)
print(ridgeEM.coef_)
print(ridgeCV.coef_)

```


## Citation
Should you find this repository helpful, please consider citing the associated paper:
```
@article{tew2023bayes,
  title={Bayes beats Cross Validation: Efficient and Accurate Ridge Regression via Expectation Maximization},
  author={Tew, Shu Yu and Boley, Mario and Schmidt, Daniel F},
  journal={arXiv preprint arXiv:2310.18860},
  year={2023}
}
```
