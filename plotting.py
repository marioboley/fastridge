from matplotlib import pyplot as plt

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
    ax.legend()
    return ax

def plot_metrics(metrics, exp, **params):
    r = len(metrics)
    s = len(exp.problems)
    fig, axs = plt.subplots(r, s, figsize=(0.5+s*3.5, r*3.5), tight_layout=True, sharex=True, sharey='row', squeeze=False)
    for i in range(r):
        for j in range(s):
            plot_metric(metrics[i], exp, j, ax=axs[i][j], **params)
            if j > 0:
                axs[i][j].set_ylabel(None)
        if i < r-1:
            axs[i][j].set_xlabel(None)
    return fig, axs