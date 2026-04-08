# Unimodality Convexity Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `Analysis/Unimodality_Convexity/` into `experiments/` by adding `RidgePathExperiment` to `experiments.py`, creating `experiments/plotting2d.py`, extending `experiments/plotting.py`, and creating `experiments/unimodality_convexity.ipynb`.

**Architecture:** Test harness (notebook + pytest.ini) established first. `RidgePathExperiment` added to `experiments.py`. Landscape plotting functions in new `plotting2d.py`. Risk pathway plotting functions added to `plotting.py`. Notebook created last, after all supporting code is in place.

**Tech Stack:** numpy, sklearn, matplotlib, scipy, nbmake, pytest

---

### Task 1: Add RidgePathExperiment to experiments.py

**Files:**
- Modify: `experiments/experiments.py`

- [ ] **Step 1: Add RidgePathExperiment class**

First, extend the import block at the top of `experiments/experiments.py`:

```python
# existing imports stay; add:
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
```

Then append to the bottom of `experiments/experiments.py`:

```python
class RidgePathExperiment:

    def __init__(self, x_train, y_train, x_test, y_test, alphas,
                 fit_intercept=True, normalize=True):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def run(self):
        n, p = self.x_train.shape

        a_x = self.x_train.mean(axis=0) if self.fit_intercept else np.zeros(p)
        a_y = self.y_train.mean() if self.fit_intercept else 0.0
        b_x = self.x_train.std(axis=0) if self.normalize else np.ones(p)
        b_y = self.y_train.std() if self.normalize else 1.0

        x_tr = (self.x_train - a_x) / b_x
        y_tr = (self.y_train - a_y) / b_y
        x_te = (self.x_test - a_x) / b_x
        y_te = (self.y_test - a_y) / b_y

        self.alphas_ = np.asarray(self.alphas)
        self.coef_path_ = np.zeros((p, len(self.alphas_)))
        self.true_risk_ = np.zeros(len(self.alphas_))

        for i, alpha in enumerate(self.alphas_):
            rr = Ridge(alpha=alpha, fit_intercept=False)
            rr.fit(x_tr, y_tr)
            self.coef_path_[:, i] = rr.coef_
            self.true_risk_[i] = mean_squared_error(y_te, rr.predict(x_te))

        lr = LinearRegression(fit_intercept=False)
        lr.fit(x_tr, y_tr)
        self.ols_coef_ = lr.coef_

        return self
```

- [ ] **Step 2: Verify import works**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'experiments')
from experiments import RidgePathExperiment
import numpy as np
rng = np.random.default_rng(1)
x_tr = rng.normal(size=(30, 5))
y_tr = rng.normal(size=30)
x_te = rng.normal(size=(20, 5))
y_te = rng.normal(size=20)
exp = RidgePathExperiment(x_tr, y_tr, x_te, y_te, np.logspace(-3, 3, 10)).run()
print('alphas_', exp.alphas_.shape)
print('coef_path_', exp.coef_path_.shape)
print('true_risk_', exp.true_risk_.shape)
print('ols_coef_', exp.ols_coef_.shape)
"
```

Expected:
```
alphas_ (10,)
coef_path_ (5, 10)
true_risk_ (10,)
ols_coef_ (5,)
```

- [ ] **Step 3: Commit**

```bash
git add experiments/experiments.py
git commit -m "feat: add RidgePathExperiment to experiments.py"
```

---

### Task 2: Create experiments/plotting2d.py

**Files:**
- Create: `experiments/plotting2d.py`

- [ ] **Step 1: Create plotting2d.py**

Copy the landscape functions verbatim from `Analysis/Unimodality_Convexity/plotting.py`. The content is:

```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
from scipy.linalg import inv, det
from matplotlib import ticker, cm
import matplotlib


def plot_marg_profile(x_train, y_train, t2, ax=None, text="", dpi=300):
    p, L = profile_marg(x_train, y_train, t2)

    if ax is None:
        ax = plt.gca()

    ax.plot(t2, p, color='forestgreen')
    ax.figure.set_dpi(dpi)

    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    plt.rcParams['text.usetex'] = True

    ax.axvline(t2[np.argmax(p)], ls=':', color='forestgreen', linewidth=2.5)
    ax.margins(x=0.01)

    ax.set_xlabel('$\\tau^2$', size=24)
    ax.set_ylabel('', size=18)
    ax.set_xscale('log')

    ax.text(t2.min() * 1.5, 1, text, color="black", fontweight='bold',
            horizontalalignment="left", verticalalignment="top", size=16)


def profile_marg(X, y, t2):
    n, p = X.shape
    L = np.zeros(len(t2))

    for i in range(len(t2)):
        tau2 = t2[i]
        A = tau2 * np.dot(X, X.T) + np.eye(n)
        sigma2_hat = np.dot(y.T, np.dot(inv(A), y)) / (n + 2)
        L[i] = (n / 2) * np.log(sigma2_hat) + (1 / 2) * np.log(det(A)) + np.dot(y.T, np.dot(inv(A), y)) / (2 * sigma2_hat) + (1 / 2) * np.log(tau2) + np.log(1 + tau2) + np.log(sigma2_hat)

    L = L - np.min(L)
    p = np.exp(-L)

    return p, L


def Q_function(x, y, sigma2, tau2):
    n, p = x.shape
    A = tau2 * x @ x.T + np.eye(n)
    sign, logabsdet = np.linalg.slogdet(A)
    marginal_likelihood = (n * np.log(sigma2)) / 2 + 1 / 2 * logabsdet + y.T @ np.linalg.inv(A) @ y / 2 / sigma2
    prior = np.log(sigma2) + np.log(1 + tau2) + np.log(tau2) / 2

    return marginal_likelihood + prior


def compute_marginal_likelihood(x, y, sig2, t2, sigma2, tau2):
    a_x, a_y = (x.mean(axis=0), y.mean())
    b_x, b_y = (x.std(axis=0), y.std())
    x = (x - a_x) / b_x
    y = (y - a_y) / b_y

    Qf = np.zeros((len(t2), len(sig2)))
    for i in range(len(t2)):
        for j in range(len(sig2)):
            z = Q_function(x, y, sig2[j], t2[i])
            Qf[i, j] = z

    return Qf


def plot_EM_step(z, sig2, t2, sigma2, tau2, levels, sigma_squares=None, tau_squares=None,
                 log=True, save_file=None, title='', figsize=(8, 9.5), dpi=300):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    z_ = np.log(z) if log else z
    cs = plt.contourf(sig2, t2, z_, cmap='viridis_r', levels=levels)

    if sigma_squares is not None:
        plt.scatter(sigma_squares, tau_squares, s=88, color='white')
        plt.scatter(sigma2, tau2, color="black", marker='x', s=70)

        u = np.diff(sigma_squares[0:4])
        v = np.diff(tau_squares[0:4])
        pos_x = sigma_squares[0:3] + u / 2
        pos_y = tau_squares[0:3] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy",
                  linestyle='dashed', width=0.006, pivot="mid", color="white")

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    plt.rcParams['text.usetex'] = True

    ax.text(sig2.min() * 1.01, t2.min() * 1.01, title, color="white", fontweight='bold',
            horizontalalignment="left", verticalalignment="bottom", size=18)

    ax.text(sig2.min() * 1.01, t2.max() * 0.7, '$-\\log p(\\tau^2, \\sigma^2 | y, X)$', color="white",
            fontweight='bold', horizontalalignment="left", verticalalignment="bottom", size=18)

    plt.xlabel('$\\sigma^2$', weight="bold", size=18)
    plt.ylabel('$\\tau^2$', rotation=360, weight="bold", size=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    cbar = plt.colorbar(cs, location='top', pad=0.025, aspect=50)
    cbar.ax.locator_params(nbins=4)

    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')

    plt.show()
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'experiments')
from plotting2d import plot_marg_profile, profile_marg, Q_function, compute_marginal_likelihood, plot_EM_step
print('plotting2d imported ok')
"
```

Expected: `plotting2d imported ok`

- [ ] **Step 3: Commit**

```bash
git add experiments/plotting2d.py
git commit -m "feat: add plotting2d module with landscape visualization functions"
```

---

### Task 3: Add plot_lambda_risks and plot_pathway_risk to plotting.py

**Files:**
- Modify: `experiments/plotting.py`

- [ ] **Step 1: Append functions to plotting.py**

Append to the bottom of `experiments/plotting.py`:

```python
from scipy.signal import argrelmin
import matplotlib


def plot_lambda_risks(ridgeCV, ridgeCV_test=None, ridgeEM=None, ax=None, axis_labels=True, title=None, dpi=300):
    ax1 = plt.gca() if ax is None else ax
    ax1.figure.set_size_inches(8.4, 4.8)
    ax1.figure.set_dpi(dpi)

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    plt.rcParams['text.usetex'] = True

    ax1.plot(ridgeCV.alphas_, ridgeCV.loo_mse_, label='LOOCV')

    local_minima = argrelmin(ridgeCV.loo_mse_, axis=0, order=1)[0]
    if len(local_minima) > 1:
        for local_min in ridgeCV.alphas_[local_minima]:
            ax1.axvline(local_min, ls='--', color='lightgrey')

    ax1.axvline(ridgeCV.alphas_[np.argmin(ridgeCV.loo_mse_)], ls=':', color='blue', label='$\\lambda^*_{CV}$')

    if ridgeCV_test is not None:
        ax1.plot(ridgeCV_test.alphas_, ridgeCV_test.true_risk_, label="True")
        ax1.axvline(ridgeCV_test.alphas_[np.argmin(ridgeCV_test.true_risk_)], ls=':', color='orange', label='$\\lambda^*$')

    plt.subplots_adjust(hspace=0.05)
    ax1.set_title(title, loc='right')
    ax1.set_xscale('log')
    ax1.margins(x=0.01)

    if ridgeEM is not None:
        ax1.axvline(1 / ridgeEM.tau_square_, ls=':', color='forestgreen', label='$\\lambda^*_{EM}$', linewidth=2)

        handles, labels = ax1.get_legend_handles_labels()
        order = [2, 0, 3, 1, 4]
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize="13")
    else:
        handles, labels = ax1.get_legend_handles_labels()
        order = [2, 0, 3, 1]
        ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize="13")

    if axis_labels:
        ax1.set_ylabel('Risk', size=18)
        ax1.set_xlabel('$\\lambda$', size=18)


def plot_pathway_risk(ridge, title=None, best_lambda=True, variable_names=None, figsize=(8, 9.5), dpi=300):
    plt.rcParams['text.usetex'] = True

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize, dpi=dpi)

    ax1.plot(ridge.alphas_, ridge.coef_path_.T, label=variable_names)
    ls = ax1.scatter(ridge.alphas_.min() * np.ones(len(ridge.ols_coef_)), ridge.ols_coef_,
                     color="black", marker='x', s=20, zorder=10)
    ax1.set_xscale("log")
    ax1.set_ylabel('$\\hat{\\beta}_{\\lambda}$', size=18)

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    ax2.plot(ridge.alphas_, ridge.true_risk_)
    ax2.set_xscale("log")
    ax2.set_xlabel('$\\lambda$', size=18)
    ax2.set_ylabel('Prediction Risk', size=18)

    ax1.text(ridge.alphas_.max(), ridge.ols_coef_.min() * 1.01, title, color="black",
             horizontalalignment="right", verticalalignment="bottom")

    if best_lambda:
        ax1.axvline(ridge.alphas_[np.argmin(ridge.true_risk_)], ls=':', color='blue')
        ax2.axvline(ridge.alphas_[np.argmin(ridge.true_risk_)], ls=':', color='blue')

    ax1.legend(ncol=2, fontsize="13")
    plt.subplots_adjust(hspace=0.05)

    plt.show()
```

Note: `ridge.coef_path_.T` is used since `coef_path_` is `(n_features, n_alphas)` — transposing gives `(n_alphas, n_features)` for matplotlib to plot each alpha as a line.

- [ ] **Step 2: Verify import**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'experiments')
from plotting import plot_metrics, plot_lambda_risks, plot_pathway_risk
print('plotting imported ok')
"
```

Expected: `plotting imported ok`

- [ ] **Step 3: Commit**

```bash
git add experiments/plotting.py
git commit -m "feat: add plot_lambda_risks and plot_pathway_risk to plotting.py"
```

---

### Task 4: Create experiments/unimodality_convexity.ipynb and update pytest.ini

**Files:**
- Create: `experiments/unimodality_convexity.ipynb`
- Modify: `pytest.ini`

- [ ] **Step 1: Create the notebook via Python script**

Run from project root:

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
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
        '# LOOCV Risk Function vs Bayesian Ridge Posterior\n\n'
        'This notebook investigates the shape of the LOOCV risk estimates as a function of $\\lambda$, '
        'and explores the posterior distribution of the Bayesian ridge hierarchy. '
        'In particular, we focus on understanding the quality and quantity of the local minima.\n\n'
        'The code presented here is used to generate the plots featured in the NeurIPS poster and presentation slides.'
    ),

    code_cell(
        'import numpy as np\n'
        'from matplotlib import pyplot as plt\n'
        'from sklearn.datasets import load_diabetes\n'
        'from sklearn.model_selection import train_test_split\n'
        'from experiments import RidgePathExperiment\n'
        'from plotting import plot_lambda_risks, plot_pathway_risk\n'
        'from plotting2d import plot_marg_profile, compute_marginal_likelihood, plot_EM_step\n'
        'from fastridge import RidgeEM, RidgeLOOCV\n'
        '\n'
        'diabetes = load_diabetes()\n'
        'x, y = load_diabetes(return_X_y=True)'
    ),

    markdown_cell('## Ridge Regression'),

    markdown_cell(
        'Given training data ${\\bf X} \\in \\mathbb{R}^{n \\times p}$ and ${\\bf y} \\in \\mathbb{R}^n$, '
        'ridge regression finds the linear regression coefficients $\\hat{\\boldsymbol{\\beta}}_\\lambda$ '
        'that minimize the $\\ell_2$-regularized sum of squared errors, i.e.,\n\n'
        '\\begin{equation}\n'
        '\\hat{\\boldsymbol{\\beta}}_\\lambda = \\underset{\\boldsymbol{\\beta}}{\\operatorname{arg\\,min}} '
        '\\left\\{ || {\\bf y}-{\\bf X}\\boldsymbol{\\beta} ||^2 + \\lambda ||\\boldsymbol{\\beta}||^2 \\right\\}.\\tag{1}\n'
        '\\end{equation}\n\n'
        'In practice, using ridge regression additionally involves estimating the value for the tuning parameter '
        '$\\lambda$ that minimizes the expected squared error $\\mathbb{E}({\\bf X}^{\\rm T}\\!\\hat{\\boldsymbol{\\beta}}_\\lambda - y)^2$ '
        'for new data ${\\bf X}$ and $y$ sampled from the same distribution as the training data.'
    ),

    code_cell(
        'x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=20, shuffle=True, random_state=180)\n'
        '\n'
        'alphas = np.logspace(-5, 5, 400, endpoint=True, base=10)\n'
        '\n'
        'path_exp = RidgePathExperiment(x_train, y_train, x_test, y_test, alphas,\n'
        '                               fit_intercept=True, normalize=True).run()\n'
        '\n'
        'plot_pathway_risk(path_exp, variable_names=[x.upper() for x in diabetes.feature_names],\n'
        '                  best_lambda=True, dpi=100)'
    ),

    markdown_cell(
        'When $\\lambda = 0$, the ridge estimates recover the regular least squares solutions '
        '(marked by the x points in the plot above). Looking at the pathway plot above, you can see that '
        'different $\\lambda$ values gives different set of regression estimates $\\boldsymbol{\\beta}_\\lambda$, '
        'which in turns, yields different predictive performance on the test dataset. Some estimates of '
        '$\\boldsymbol{\\beta}_\\lambda$ gives overall better predictive performance than others. Consequently, '
        'the objective is to estimate the optimal $\\lambda$ (dotted blue horizontal line) from the training data.\n\n'
        'This problem is usually approached via the leave-one-out cross-validation (LOOCV) estimator. However, '
        'as discussed in the follwing section, this approach exhibits two weaknessess: (i) its risk function can '
        'be multimodal, and (ii) the global minima can be far away from the true optimal $\\lambda$, resulting '
        'in high prediction error.'
    ),

    markdown_cell('## LOOCV Risk'),

    code_cell(
        'N = 400\n'
        'alphas = np.logspace(-15, 10, N, endpoint=True, base=10)\n'
        '\n'
        'ridgeCV = RidgeLOOCV(alphas=alphas, fit_intercept=True, normalize=True)\n'
        'ridgeCV.fit(x_train, y_train)\n'
        '\n'
        'path_exp2 = RidgePathExperiment(x_train, y_train, x_test, y_test, alphas,\n'
        '                                fit_intercept=True, normalize=True).run()\n'
        '\n'
        'ridgeEM = RidgeEM(trace=True)\n'
        'ridgeEM.fit(x_train, y_train)\n'
        '\n'
        'plot_lambda_risks(ridgeCV, path_exp2, ridgeEM=ridgeEM, title=None, localmin=True, dpi=100)\n'
        'plt.show()'
    ),

    markdown_cell(
        'In the plot above, we observe multiple local minima (grey dotted lines) in the LOOCV risk function, '
        'with the global minimum (blue dotted line) notably far from the true optimal $\\lambda$ (orange dotted line). '
        'In contrast, the proposed EM procedure identifies an optimal $\\lambda$ that is close to the true optimal. '
        'Subsequent visuals in the next section illustrates the theoreom presented in the paper, highlighting that '
        'for sufficiently large $n$ and $\\tau^2$ being far enough away from zero, the joint posterior of the '
        'Bayesian ridge is unimodal.'
    ),

    markdown_cell('## Bayesian Ridge'),

    markdown_cell(
        'The ridge estimator has a well-known Bayesian interpretation; specifically, if we assume that the '
        'coefficients are <em>a priori</em> normally distributed with mean zero and common variance $\\tau^2 \\sigma^2$ '
        'we obtain a Bayesian version of the usual ridge regression procedure, i.e.,\n\n'
        '\\begin{align*}\n'
        '\\begin{split}\n'
        '    {\\bf y}\\,|\\,{\\bf X}, \\boldsymbol{\\beta}, \\sigma^2 \\; &\\sim \\; N_n\\left({\\bf X}\\boldsymbol{\\beta}, \\; \\sigma^2{\\bf I}_n\\right),\\\\\n'
        '    \\boldsymbol{\\beta}\\,|\\,\\tau^2, \\sigma^2 \\; &\\sim \\; N_p\\left(0, \\; \\tau^2 \\sigma^2{\\bf I}_p\\right),\\\\\n'
        '    \\sigma^2 &\\sim \\sigma^{-2} d\\sigma^2, \\\\\n'
        '    \\tau^2 \\; &\\sim \\; \\pi(\\tau^2)d\\tau^2.\n'
        '\\end{split}\n'
        '\\end{align*}\n\n'
        'With $\\tau > 0$ and $\\sigma>0$, the conditional posterior distribution of $\\boldsymbol{\\beta}$ is normal\n\n'
        '\\begin{align*}\n'
        '\\begin{split}\n'
        '    \\boldsymbol{\\beta}\\,|\\,\\tau^2,\\sigma^2,{\\bf y} \\; &\\sim \\; N_p(\\hat{\\boldsymbol{\\beta}}_{\\tau}, \\; \\sigma^2{\\bf A}_{\\tau}^{-1}),\\\\\n'
        '    \\hat{\\boldsymbol{\\beta}}_{\\tau} &= {\\bf A}_{\\tau}^{-1}{\\bf X}^{\\rm T}{\\bf y}, \\\\\n'
        '    {\\bf A}_{\\tau} \\; &= \\; ({\\bf X}^{\\rm T} {\\bf X} +  \\tau^{-2} {\\bf I}_p),\n'
        '\\end{split}\n'
        '\\end{align*}\n\n'
        'where the posterior mode (and mean) $\\hat{\\boldsymbol{\\beta}}_{\\tau}$ is equivalent to the ridge estimate '
        '([1](#mjx-eqn-eq:ridge_optim)) with penalty $\\lambda = 1/\\tau^2$.'
    ),

    markdown_cell(
        'The key idea now is to maximize the marginal posterior for $\\tau^2$ and $\\sigma^2$:\n\n'
        '\\begin{align*}\n'
        '\\begin{split}\n'
        '    &\\underset{\\tau^2, \\sigma^2}{\\operatorname{arg\\,min}}\\left[-\\log \\pi(\\tau^2, \\sigma^2)'
        '\\int p({\\bf y}| {\\bf X}, \\boldsymbol{\\beta}, \\tau^2, \\sigma^2) \\pi(\\boldsymbol{\\beta}| \\tau^2, \\sigma^2) d\\boldsymbol{\\beta} \\right] \\\\\n'
        '    &= \\underset{\\tau^2, \\sigma^2}{\\operatorname{arg\\,min}}[-\\log p(\\tau^2, \\sigma^2| {\\bf y}, {\\bf X})]\n'
        '\\end{split}\n'
        '\\end{align*}\n\n'
        'And one approach to do this is by using the EM algorithm to iteratively estimate the hyperparameters '
        '$\\tau^2$ and $\\sigma^2$ by minimizing the expected log posterior, with the regression coefficients '
        '$\\boldsymbol{\\beta}$ treated as latent variable:\n'
        '$$\n'
        '    \\{\\hat{\\tau}^2_{t+1}, \\hat{\\sigma}^{2}_{t+1} \\} = \\underset{\\tau^2, \\sigma^2}{\\operatorname{arg\\,min}} \\,  '
        '\\mathbb{E}_{\\boldsymbol{\\beta}}[-\\log p(\\boldsymbol{\\beta}, \\tau^2, \\sigma^2| {\\bf y}, {\\bf X}) | \\hat{\\tau}^2_{t}, \\hat{\\sigma}^{2}_{t} ] .\n'
        '$$'
    ),

    code_cell(
        'ridgeEM2 = RidgeEM(trace=True)\n'
        'ridgeEM2.fit(x_train, y_train)\n'
        '\n'
        'N = 100\n'
        'sig2 = np.logspace(-2, 0.5, N)\n'
        't2 = np.logspace(-2, 0.5, N)\n'
        '\n'
        'z = compute_marginal_likelihood(x_train, y_train, sig2, t2,\n'
        '                                ridgeEM2.sigma_square_, ridgeEM2.tau_square_)\n'
        '\n'
        'plot_EM_step(z, sig2, t2, ridgeEM2.sigma_square_, ridgeEM2.tau_square_,\n'
        '             levels=30, sigma_squares=ridgeEM2.sigma_squares_, tau_squares=ridgeEM2.tau_squares_,\n'
        '             log=True, title=\'Diabetes, $p=10$, $n=20$\', dpi=80)'
    ),

    markdown_cell(
        'In the paper, we presented the following theorem:\n\n'
        '**Theorem 3.1** *Let $\\epsilon > 0$, and let $\\gamma_n$ be the smallest eigenvalue of '
        '${\\bf X}^{\\rm T}{\\bf X}/n$. If $\\gamma_n > 0$ and $\\epsilon > 4/(n\\gamma_n)$ then the joint posterior\n'
        '$p(\\boldsymbol{\\beta}, \\sigma^2, \\tau^2 | {\\bf y})$ has a unique mode with $\\tau^2 \\geq \\epsilon$.\n'
        'In particular, if $\\gamma_n \\geq cn^{-\\alpha}$ with $\\alpha < 1$ and $c>0$ then there is a unique mode '
        'with $\\tau^2 \\geq \\epsilon$ if $n > (4/(c\\epsilon))^{1/(1-\\alpha)}$.*\n\n'
        'This suggest that all sub-optimal non-zero posterior modes vanish for large enough $n$ if the smallest '
        'eigenvalue of ${\\bf X}^{\\rm T} {\\bf X}$ grows at least proportionally to some positive power of $n$. '
        'This is a very mild assumption that is typically satisfied in fixed as well as random design settings, '
        'e.g., with high probability when the smallest marginal covariate variance is bounded away from zero.\n\n'
        'Here we attempt to visualise this fact by plotting the **marginal posterior profile** for $\\tau^2$.\n\n'
        '<sup>(Note) *We are guided by the notion that if the joint distribution exhibits unimodality in a given domain, '
        'then the individual marginal distributions will be unimodal as well. This hypothesis forms the basis of our '
        'investigation, as we endeavor to validate or challenge its validity through future analysis.*</sup>'
    ),

    markdown_cell('### Marginal posterior profile'),

    code_cell(
        'def make_patch_spines_invisible(ax):\n'
        '    ax.set_frame_on(True)\n'
        '    ax.patch.set_visible(False)\n'
        '    for sp in ax.spines.values():\n'
        '        sp.set_visible(False)\n'
        '\n'
        'N = 400\n'
        '\n'
        'n_size = [20, 40, 80]\n'
        't2_lowerlimit = [-3, -8, -17]\n'
        't2_upperlimit = [1, 6, 10]\n'
        '\n'
        '_, axs = plt.subplots(1, len(n_size), tight_layout=True, figsize=(12, 4), sharey=True, dpi=100)\n'
        '\n'
        'for i, s in enumerate(n_size):\n'
        '    x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(\n'
        '        x, y, train_size=s, shuffle=True, random_state=180)\n'
        '    alphas_s = 1 / np.logspace(t2_lowerlimit[i], t2_upperlimit[i], N, endpoint=True, base=10)\n'
        '\n'
        '    a_x = x_train_s.mean(axis=0)\n'
        '    a_y = y_train_s.mean()\n'
        '    b_x = x_train_s.std(axis=0)\n'
        '    b_y = y_train_s.std()\n'
        '    x_tr_norm = (x_train_s - a_x) / b_x\n'
        '    y_tr_norm = (y_train_s - a_y) / b_y\n'
        '\n'
        '    path_s = RidgePathExperiment(x_train_s, y_train_s, x_test_s, y_test_s, alphas_s,\n'
        '                                 fit_intercept=True, normalize=True).run()\n'
        '    t2_grid = 1 / path_s.alphas_\n'
        '\n'
        '    axs[i].plot(t2_grid, path_s.true_risk_, label="True", color=\'#ff7f0e\')\n'
        '    axs[i].axvline(t2_grid[np.argmin(path_s.true_risk_)], ls=\':\', color=\'#ff7f0e\',\n'
        '                   label=\'$\\\\lambda^*$\', linewidth=2.5)\n'
        '\n'
        '    if i > 0:\n'
        '        axs[i].get_yaxis().set_visible(False)\n'
        '\n'
        '    ax2 = axs[i].twinx()\n'
        '    plot_marg_profile(x_tr_norm, y_tr_norm, t2_grid, ax=ax2,\n'
        '                      text="(n =" + str(s) + ")", dpi=100)\n'
        '\n'
        '    axs[i].set_xlabel(\'$\\\\tau^2$\', size=24)\n'
        '\n'
        '    if i == 2:\n'
        '        ax2.tick_params(axis=\'y\', colors=\'black\')\n'
        '    else:\n'
        '        ax2.set_yticks([])'
    ),

    markdown_cell(
        'The left y axis is the true risk function (yellow) and the right y-axis is the marginal posterior '
        'distribution of $\\tau^2$ (green) up to a constant - '
        '$\\propto \\underset{\\sigma^2}{\\mathrm{max}}\\, p(\\tau^2, \\sigma^2 \\,|\\, {\\bf y}, {\\bf X})$. '
        'There seems to be a consistent presence of a peak close to $\\tau^2 = 0$; however, this peak becomes '
        'exceedingly narrow, eventually approaching zero probability, and progressively converges towards zero as '
        'the sample size $n$ increases. Additionally, these visualizations suggests that the marginal distribution '
        'is unimodal for $\\tau^2$ sufficiently larger than zero (as discussed in Thereom 3.1).'
    ),
]

with open('experiments/unimodality_convexity.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Done -', len(nb['cells']), 'cells')
for i, c in enumerate(nb['cells']):
    print(f"  Cell {i} ({c['cell_type']}) tags={c.get('metadata', {}).get('tags', [])}")
EOF
```

Expected output:
```
Done - 16 cells
  Cell 0 (markdown) tags=[]
  Cell 1 (code) tags=[]
  Cell 2 (markdown) tags=[]
  Cell 3 (markdown) tags=[]
  Cell 4 (code) tags=[]
  Cell 5 (markdown) tags=[]
  Cell 6 (markdown) tags=[]
  Cell 7 (code) tags=[]
  Cell 8 (markdown) tags=[]
  Cell 9 (markdown) tags=[]
  Cell 10 (markdown) tags=[]
  Cell 11 (markdown) tags=[]
  Cell 12 (code) tags=[]
  Cell 13 (markdown) tags=[]
  Cell 14 (markdown) tags=[]
  Cell 15 (code) tags=[]
  Cell 16 (markdown) tags=[]
```

- [ ] **Step 2: Update pytest.ini**

Current `pytest.ini`:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake experiments/double_asymptotic_trends.ipynb --nbmake experiments/sparse_designs.ipynb
```

New `pytest.ini`:
```ini
[pytest]
addopts = --doctest-modules fastridge.py --codeblocks README.md --nbmake experiments/double_asymptotic_trends.ipynb --nbmake experiments/sparse_designs.ipynb --nbmake experiments/unimodality_convexity.ipynb
```

- [ ] **Step 3: Run pytest**

```bash
cd /Users/marioboley/Documents/GitHub/fastridge
source .venv/bin/activate
pytest
```

Expected: all tests pass. If the contour plot cell fails due to `text.usetex` requiring a LaTeX installation, set `plt.rcParams['text.usetex'] = False` in `plotting2d.py` as a fallback, or verify LaTeX is available with `python3 -c "import matplotlib; matplotlib.checkdep_usetex(True)"`.

- [ ] **Step 4: Commit**

```bash
git add experiments/unimodality_convexity.ipynb pytest.ini
git commit -m "feat: add unimodality_convexity notebook and update pytest.ini"
```

---

### Task 5: Push and merge to main

- [ ] **Step 1: Push dev and merge**

```bash
git push origin dev
git checkout main
git merge dev
git push origin main
git checkout dev
```
