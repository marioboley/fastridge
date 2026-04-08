import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import inv, det
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
        L[i] = ((n / 2) * np.log(sigma2_hat) + (1 / 2) * np.log(det(A))
                 + np.dot(y.T, np.dot(inv(A), y)) / (2 * sigma2_hat)
                 + (1 / 2) * np.log(tau2) + np.log(1 + tau2) + np.log(sigma2_hat))

    L = L - np.min(L)
    p = np.exp(-L)

    return p, L


def Q_function(x, y, sigma2, tau2):
    n, p = x.shape
    A = tau2 * x @ x.T + np.eye(n)
    sign, logabsdet = np.linalg.slogdet(A)
    marginal_likelihood = ((n * np.log(sigma2)) / 2 + 1 / 2 * logabsdet
                           + y.T @ np.linalg.inv(A) @ y / 2 / sigma2)
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
            Qf[i, j] = Q_function(x, y, sig2[j], t2[i])

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

    ax.text(sig2.min() * 1.01, t2.max() * 0.7, '$-\\log p(\\tau^2, \\sigma^2 | y, X)$',
            color="white", fontweight='bold',
            horizontalalignment="left", verticalalignment="bottom", size=18)

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
