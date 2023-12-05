import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
from scipy.linalg import inv, det
#from fastridge import RidgeEM, RidgeLOOCV, RidgeTrueRisk
from matplotlib import ticker, cm
import matplotlib 

def plot_marg_profile(x_train, y_train, t2, ax = None, text = "", dpi = 300):
    p,L = profile_marg(x_train, y_train, t2)

    if ax is None:
        ax = plt.gca()
    
    ax.plot(t2, p, color = 'forestgreen')

    #ax.figure.set_size_inches(8.4, 4.8)
    ax.figure.set_dpi(dpi)

    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18) 
    plt.rcParams['text.usetex'] = True
    
    #print(t2[np.argmax(p)])

    ax.axvline(t2[np.argmax(p)], ls=':', color='forestgreen', linewidth=2.5)
    ax.margins(x=0.01)

    ax.set_xlabel('$\\tau^2$', size = 24)
    ax.set_ylabel('', size = 18)
    ax.set_xscale('log')
    
    ax.text(t2.min()*1.5, 1, text, color="black", fontweight = 'bold',
            horizontalalignment="left", verticalalignment="top", size = 16)


def profile_marg(X, y, t2):
    n, p = X.shape
    L = np.zeros(len(t2))

    for i in range(len(t2)):
        tau2 = t2[i]
        A = tau2 * np.dot(X, X.T) + np.eye(n)
        sigma2_hat = np.dot(y.T, np.dot(inv(A), y)) / (n + 2)
        L[i] = (n / 2) * np.log(sigma2_hat) + (1 / 2) * np.log(det(A)) + np.dot(y.T, np.dot(inv(A), y)) / (2 * sigma2_hat) + (1 / 2) * np.log(tau2) + np.log(1 + tau2) + np.log(sigma2_hat)

    # Normalize
    L = L - np.min(L)
    p = np.exp(-L)

    return p, L

def plot_lambda_risks(ridgeCV, ridgeCV_test = None, ridgeEM = None, ax = None, axis_labels=True, title = None, localmin = True, dpi = 300):
    
    ax1 = plt.gca() if ax is None else ax
    ax1.figure.set_size_inches(8.4, 4.8)
    ax1.figure.set_dpi(dpi)
    
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16)
    
    plt.rcParams['text.usetex'] = True
    
    ax1.plot(ridgeCV.alphas_, ridgeCV.loo_mse_, label = 'LOOCV')
    
    if localmin:
        local_minima = argrelmin(ridgeCV.loo_mse_, axis=0, order=1)[0]
        if len(local_minima) > 1:
            for local_min in ridgeCV.alphas[local_minima]:
                ax1.axvline(local_min, ls='--', color='lightgrey')
                
    ax1.axvline(ridgeCV.alphas_[np.argmin(ridgeCV.loo_mse_)], ls=':', color='blue', label='$\lambda^*_{CV}$')
    
    if ridgeCV_test is not None:
        ax1.plot(ridgeCV_test.alphas_, ridgeCV_test.true_risk, label = "True")
        #local_minima = argrelmin(ridgeCV_test.true_risk, axis=0, order=1)[0]
        #if len(local_minima) > 1:
            #for local_min in ridgeCV_test.alphas[local_minima]:
                #ax1.axvline(local_min, ls=':', color='lightgreen')
        ax1.axvline(ridgeCV_test.alphas_[np.argmin(ridgeCV_test.true_risk)], ls=':', color='orange', label='$\lambda^*$')

    
    plt.subplots_adjust(hspace=0.05)
    ax1.set_title(title, loc='right')
    ax1.set_xscale('log')
    ax1.margins(x=0.01)
    #ax2.margins(x=0.01)
    
    if ridgeEM is not None:
        ax1.axvline(1/ridgeEM.tau_square_, ls=':', color='forestgreen', label='$\lambda^*_{EM}$',linewidth=2)
        
        handles, labels = ax1.get_legend_handles_labels()
        order = [2,0,3,1,4]
        ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize="13")
        
    else:
        handles, labels = ax1.get_legend_handles_labels()
        order = [2,0,3,1]
        ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize="13")
    
    if axis_labels:
        ax1.set_ylabel('Risk', size = 18)
        ax1.set_xlabel('$\lambda$', size = 18)
            
            
            
def plot_pathway_risk(ridge, title = None, best_lambda = True, variable_names = None, figsize = (8,9.5), dpi = 300):
    plt.rcParams['text.usetex'] = True
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize, dpi=dpi)
    
    ax1.plot(ridge.alphas_, ridge.coefs, label = variable_names)
    ls = ax1.scatter(ridge.alphas_.min() * np.ones(len(ridge.lr_coef)), ridge.lr_coef, color="black", 
                marker = 'x', s = 20, zorder=10)
    ax1.set_xscale("log")
    #ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    #plt.xlabel('$ \lambda $')
    ax1.set_ylabel('$ \hat{\\beta}_{\lambda} $', size = 18)
    
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    
    ax2.plot(ridge.alphas_, ridge.true_risk)
    ax2.set_xscale("log")
    ax2.set_xlabel('$\lambda$', size = 18)
    ax2.set_ylabel('Prediction Risk', size = 18)
    
    ax1.text(ridge.alphas_.max(), ridge.lr_coef.min()*1.01, title, color="black",
            horizontalalignment="right", verticalalignment="bottom")
    
    if best_lambda:
        ax1.axvline(ridge.alphas_[np.argmin(ridge.true_risk)], ls=':', color='blue')
        ax2.axvline(ridge.alphas_[np.argmin(ridge.true_risk)], ls=':', color='blue')
    
    #ax1.legend([ls], ['Least squares,  $\hat{\\beta}_{\lambda = 0}$'])
    ax1.legend(ncol=2, fontsize="13")
    plt.subplots_adjust(hspace=0.05)
    
    plt.show()
    
    
    
def Q_function(x, y, sigma2, tau2):
    
    n, p = x.shape
    A = tau2 * x@x.T + np.eye(n)
    sign, logabsdet = np.linalg.slogdet(A)
    marginal_likelihood = (n*np.log(sigma2))/2 + 1/2 * logabsdet + y.T@np.linalg.inv(A)@y/2/sigma2 
    prior = np.log(sigma2) + np.log(1+tau2) + np.log(tau2)/2
    
    return marginal_likelihood + prior


def compute_marginal_likelihood(x, y, sig2, t2, sigma2, tau2):
    
    a_x, a_y = (x.mean(axis=0), y.mean()) 
    b_x, b_y = (x.std(axis=0), y.std())
    x = (x - a_x)/b_x
    y = (y - a_y)/b_y
    
    Qf = np.zeros((len(t2), len(sig2)))
    for i in range(len(t2)):
        for j in range(len(sig2)):
            z = Q_function(x, y, sig2[j], t2[i])
            Qf[i,j] = z 
    
    #print("maxQf = ", Qf.max(), " minQf = ", Qf.min())
    
    epsilon = 1e-8
    #print("gradients")
    #print(Q_function(x, y, sigma2, tau2))
    #print((Q_function(x, y, sigma2, tau2 + epsilon) - Q_function(x, y, sigma2, tau2)) / epsilon)
    #print((Q_function(x, y, sigma2 + epsilon, tau2) - Q_function(x, y, sigma2, tau2)) / epsilon)
    #print(Qf)
    
    return Qf


def plot_EM_step(z, sig2, t2, sigma2, tau2, levels, sigma_squares = None, tau_squares = None, log = True, save_file = None, title = '', figsize = (8,9.5), dpi = 300):
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    z_ = np.log(z) if log else z
    cs = plt.contourf(sig2, t2, z_, cmap='viridis_r', 
                      levels = levels)
    
    if sigma_squares is not None:
        
        plt.scatter(sigma_squares, tau_squares, s = 88, color = 'white')
        plt.scatter(sigma2, tau2, color="black", marker = 'x', s = 70) 
    
        u = np.diff(sigma_squares[0:4])
        v = np.diff(tau_squares[0:4])
        pos_x = sigma_squares[0:3] + u/2
        pos_y = tau_squares[0:3] + v/2
        norm = np.sqrt(u**2+v**2) 
 
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", 
                  linestyle='dashed', width=0.006, pivot="mid", color = "white")
    
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    
    #eq1 = ("$-\log p(\\tau^2, \sigma^2 | y, X)$")
    
    plt.rcParams['text.usetex'] = True
    eq1 = (r"\begin{eqnarray*}"
       r"&-&\log p(\tau^2, \sigma^2 | y, X)\\"
       r"&=& \left( \frac{n+p+2}{2} \right) \log \sigma^2 + \frac{{\rm ESS}}{2 \sigma^2} + \frac{p+1}{2} \log \tau^2 + \frac{{\rm ESN}}{2 \sigma^2 \tau^2} + \log(1+\tau^2) "
       r"\end{eqnarray*}")
    
    ax.text(sig2.min()*1.01, t2.min()*1.01, title, color="white", fontweight = 'bold',
            horizontalalignment="left", verticalalignment="bottom", size = 18)
    
    ax.text(sig2.min()*1.01, t2.max()*0.7, '$-\log p(\\tau^2, \sigma^2 | y, X)$', color="white", fontweight = 'bold',
            horizontalalignment="left", verticalalignment="bottom", size = 18)
    
    plt.xlabel('$\sigma^2$', weight = "bold", size = 18)
    plt.ylabel('$ \\tau^2 $', rotation=360, weight = "bold", size = 18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    cbar = plt.colorbar(cs, location='top', pad = 0.025, aspect=50)
    cbar.ax.locator_params(nbins=4)
    #cbar.set_ticks(np.linspace(round(z_.min()), round(z_.max()), 4))
    #cbar.ax.get_xaxis().labelpad = 10
    #cbar.ax.set_xlabel('$-\log p(\\tau^2, \sigma^2 | y, X)$')
    #plt.title(title, fontsize=15)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight')
    
    plt.show() 
    
    
    
