import numpy as np
from scipy.special import gammaln, gamma

def sample_ggp(eta, sigma, tau, n_jumps=20000, T=None):
    def W(t, x):
        if tau > 0:
            log_output = np.log(eta) + np.log(1. - np.exp(-tau * (x - t))) - (1. + sigma)*np.log(t) - (t*tau) - np.log(tau) - gammaln(1. - sigma)
        else:
            log_output = np.log(eta) - gammaln(1. - sigma) - np.log(sigma) + np.log(t**(-sigma) - x**(-sigma))
        return np.exp(log_output)
    
    def inv_W(t, x):
        if tau > 0:
            return t - np.log(1. - gamma(1. - sigma) * x * tau / (eta * t**(-1.-sigma) * np.exp(-t * tau))) / tau
        else:
            log_output = -np.log(t**(-sigma) - sigma * gamma(1. - sigma) / eta * x) / sigma
            return np.exp(log_output)

    if T is None:
        T = np.exp(
            (np.log(eta) - np.log(sigma) - gammaln(1. - sigma) - np.log(n_jumps)) / sigma
        )
    else:
        if sigma > 0:
            n_jumps = np.floor(eta / sigma / gamma(1. - sigma)*T**(-sigma))
        else:
            n_jumps = np.floor(-eta * np.log(T));

    samples = []
    t, count = T, 0
    while True:
        r = -np.log(np.random.rand())
        if r > W(t, np.inf):
            break
        else:
            t_new = inv_W(t, r)
        if tau == 0 or np.log(np.random.rand()) < (-(1. + sigma) * np.log(t_new / t)):
            samples.append(t_new)
        t = t_new
        count += 1
        if count > 1e8:
            print("Threshold T is too large:", T)
            T = T / 10
            samples = []
            t, count = T, 0
    return np.array(samples)