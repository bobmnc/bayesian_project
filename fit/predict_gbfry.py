import numpy as np
from sample_gbfry import sample_gbfry

def predict_gbfry(fit_results, n_slice=100):
    N = len(fit_results['eta'])
    etas = fit_results["eta"][N//2::n_slice]
    sigmas = fit_results["sigma"][N//2::n_slice]
    taus = fit_results["tau"][N//2::n_slice]

    predictions = []
    for eta, sigma, tau in zip(etas, sigmas, taus):
        W_pred = sample_gbfry(eta, sigma, 1.0, tau, T=1e-8)
        counts_pred = np.random.poisson(lam=n * W_pred / np.sum(W_pred))
        counts_pred = counts_pred[counts_pred > 0]
        predictions.append(np.sort(counts_pred)[::-1])
    return predictions