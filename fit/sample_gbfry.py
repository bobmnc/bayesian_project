import numpy as np
from sample_ggp import sample_ggp

def sample_gbfry(eta, sigma, c, tau, T):
    eta0 = eta * (c**(tau - sigma)) / tau
    W = sample_ggp(eta0, sigma, c, T=T)
    beta = np.random.beta(tau, 1., W.size)
    W_normalised = W / beta
    return W_normalised