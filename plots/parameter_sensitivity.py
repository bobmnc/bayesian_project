import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from fit import sample_gbfry


def plot_proportion_of_clusters(eta=None, sigma=None, c=None, tau=None, T=None,n=None, n_values=None, sigma_values=None, eta_values=None, tau_values=None,var_name = 'eta'):
    if n_values is not None:
        results = [sample_gbfry(eta, sigma, c, tau, T)]
    elif sigma_values is not None:
        results = [sample_gbfry(eta, sigma, c, tau, T) for sigma in sigma_values]
    elif eta_values is not None:
        results = [sample_gbfry(eta, sigma, c, tau, T) for eta in eta_values]
    elif tau_values is not None:
        results = [sample_gbfry(eta, sigma, c, tau, T) for tau in tau_values]


    for k,result in enumerate(results):
        if n_values is not None :
            for n in n_values: 
                counts = np.random.poisson(lam=n * result / np.sum(result))
                counts = counts[counts > 0]
                plt.loglog(np.sort(counts)[::-1]/np.sum(counts), marker='o', linestyle='', markersize=3)
        else :
            counts = np.random.poisson(lam=n * result / np.sum(result))
            counts = counts[counts > 0]
            plt.loglog(np.sort(counts)[::-1]/np.sum(counts), marker='o', linestyle='', markersize=3,label =fr'$\eta$ = {eta_values[k]}')
    plt.xlabel('Size')
    plt.ylabel('Distribution')
    plt.show()