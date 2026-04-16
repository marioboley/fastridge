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


_SCATTER_PAD = 0.03


def scatter_clipped(x, y, c, norm, cmap, clip_min=-0.1, clip_max=1.0,
                    ref_lines=(0.0,), pad=_SCATTER_PAD, ax=None):
    """Scatter plot with out-of-range points clipped and marked with dashed edges.

    Points where either coordinate falls outside [clip_min, clip_max] are clipped
    to the nearest bound and drawn with dashed edges. A diagonal reference line
    runs from corner to corner of the axes. Operates on ax (default: plt.gca()).

    Parameters
    ----------
    x, y : array-like of float
        Coordinates, one entry per point.
    c : array-like of float
        Colour values, same length as x and y. Mapped via norm and cmap.
    norm : matplotlib.colors.Normalize
        Pre-computed normalisation — must cover all data in the grid for a
        globally consistent colour scale.
    cmap : matplotlib colormap
    clip_min : float, default -0.1
    clip_max : float, default 1.0
    ref_lines : sequence of float, default (0.0,)
        Positions of grey horizontal and vertical reference lines.
    pad : float, default 0.03
        Margin added beyond [clip_min, clip_max] on all sides.
    ax : matplotlib.axes.Axes or None
    """
    if ax is None:
        ax = plt.gca()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)

    lim = [clip_min - pad, clip_max + pad]
    clipped = (x < clip_min) | (x > clip_max) | (y < clip_min) | (y > clip_max)
    x_disp = np.clip(x, clip_min, clip_max)
    y_disp = np.clip(y, clip_min, clip_max)
    colors = cmap(norm(c))

    if (~clipped).any():
        ax.scatter(x_disp[~clipped], y_disp[~clipped], c=colors[~clipped],
                   s=50, zorder=3, edgecolors='k', linewidths=0.6)
    if clipped.any():
        sc = ax.scatter(x_disp[clipped], y_disp[clipped], c=colors[clipped],
                        s=50, zorder=4, edgecolors='k', linewidths=0.8)
        sc.set_linestyle('--')

    ax.plot(lim, lim, 'k--', lw=0.8, zorder=2)
    for ref in ref_lines:
        ax.axhline(ref, color='0.8', lw=0.5, zorder=1)
        ax.axvline(ref, color='0.8', lw=0.5, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    return ax


def grid_with_colourbar(nrows, ncols, norm, cmap,
                        y_labels=None, col_titles=None,
                        x_labels='', cbar_label='',
                        cbar_fraction=0.56, figsize=None):
    """Create an nrows x ncols grid of axes with a shared colorbar on the right.

    Returns (fig, axes) where axes has shape (nrows, ncols). The caller populates
    each axis — scatter_clipped is one option but the function is not scatter-specific.

    Parameters
    ----------
    nrows, ncols : int
    norm : matplotlib.colors.Normalize
        Used to draw the colorbar; pre-compute over all data before calling.
    cmap : matplotlib colormap
    y_labels : str or list of str of length nrows, or None
        Y-axis label(s) applied to the leftmost column only.
        A single string is repeated for all rows (symmetric with x_labels).
    col_titles : list of str, length ncols, or None
        Column titles applied to the top row only.
    x_labels : str or list of str of length ncols
        X-axis label(s) applied to bottom-row axes. Single string applies to all.
    cbar_label : str
        Label for the colorbar.
    cbar_fraction : float, default 0.56
        Height of the colorbar as a fraction of the axes area height
        (the vertical span set by subplots_adjust). Colorbar is centred on the
        axes midpoint. Default reproduces the original figure layout.
    figsize : tuple or None
        Passed to plt.subplots. Default: (3 * ncols, 2.7 * nrows).
    """
    if figsize is None:
        figsize = (3 * ncols, 2.7 * nrows)

    bottom_adj, top_adj = 0.11, 0.93
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.84, bottom=bottom_adj, top=top_adj,
                        hspace=0.06, wspace=0.04)

    if col_titles:
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title)

    y_labels_list = ([y_labels] * nrows if isinstance(y_labels, str)
                     else (y_labels or []))
    for i, label in enumerate(y_labels_list):
        axes[i, 0].set_ylabel(label)

    x_labels_list = [x_labels] * ncols if isinstance(x_labels, str) else list(x_labels)
    for j, label in enumerate(x_labels_list):
        axes[nrows - 1, j].set_xlabel(label)

    for j in range(ncols):
        axes[nrows - 1, j].set_xticks([0.0, 0.5, 1.0])
    for i in range(nrows):
        axes[i, 0].set_yticks([0.0, 0.5, 1.0])

    cbar_h = cbar_fraction * (top_adj - bottom_adj)
    cbar_b = (top_adj + bottom_adj) / 2 - cbar_h / 2
    cbar_ax = fig.add_axes([0.86, cbar_b, 0.02, cbar_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_label_position('right')
    cbar.set_label(cbar_label, rotation=90, labelpad=-30)

    return fig, axes


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