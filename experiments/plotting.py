import shutil
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
import matplotlib

matplotlib.rcParams['text.usetex'] = shutil.which('latex') is not None

def plot_metric(metric, exp, p=0, est_idx=None, plot_intervals=True, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    est_idx = range(len(exp.estimators)) if est_idx is None else est_idx
        
    for i in est_idx:
        err = exp.__dict__[str(metric)+'_'][:, p, :, i]
        if plot_intervals:
            ax.errorbar(exp.ns[p], err.mean(axis=0), yerr=1.96*err.std(axis=0)/exp.reps**0.5, label=exp.est_names[i])
        else:
            ax.plot(exp.ns[p], err.mean(axis=0), label=exp.est_names[i])
    ax.set_xlabel('$n$')
    ax.set_ylabel(metric.symbol())
    ax.margins(x=0.005)
    # ax.legend()
    return ax

def plot_metrics(exp, metrics, prob_idx=None, figsize=None, **params):
    prob_idx = list(range(len(exp.problems))) if prob_idx is None else prob_idx
    r = len(metrics)
    s = len(prob_idx)
    figsize = (0.5+s*3.5, r*3.5) if figsize is None else figsize
    fig, axs = plt.subplots(r, s, figsize=figsize, tight_layout=True, sharex=True, sharey='row', squeeze=False)
    for i in range(r):
        for j in range(s):
            plot_metric(metrics[i], exp, prob_idx[j], ax=axs[i][j], **params)
            if j > 0:
                axs[i][j].set_ylabel(None)
            if i < r-1:
                axs[i][j].set_xlabel(None)
    return fig, axs


def plot_lambda_risks(ridgeCV, ridgeCV_test=None, ridgeEM=None, ax=None, axis_labels=True, title=None, localmin=True, dpi=300):
    ax1 = plt.gca() if ax is None else ax
    ax1.figure.set_size_inches(8.4, 4.8)
    ax1.figure.set_dpi(dpi)

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)


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


def plot_pathway_risk(ridge, title=None, best_lambda=True, variable_names=None, figsize=(8, 9.5), dpi=300, save_file=None):
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize, dpi=dpi)

    ax1.plot(ridge.alphas_, ridge.coef_path_.T, label=variable_names)
    ax1.scatter(ridge.alphas_.min() * np.ones(len(ridge.ols_coef_)), ridge.ols_coef_,
                color="black", marker='x', s=20, zorder=10)
    ax1.set_xscale("log")
    ax1.set_ylabel('$\\hat{\\beta}_{\\lambda}$', size=18)

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

    if save_file is not None:
        plt.savefig(save_file, dpi=600, bbox_inches="tight", pad_inches=0)

    plt.show()