

def gbfry_psi(t, eta, sigma, c, tau):
    return (eta / sigma) * integrate.quad(
        lambda x: ((t+x)**sigma - x**sigma) * x**(tau-sigma-1),
        0, c
    )[0]

def ll_gbfry(counts, ty, u, eta, sigma, c, tau):
    n = np.sum(counts)
    y = c / (1. + np.exp(-ty))
    ll = (n - 1) * np.log(u) - gbfry_psi(u, eta, sigma, c, tau) - gammaln(n)
    ll += counts.size * (np.log(eta) - gammaln(1. - sigma) - np.log(c))
    ll += np.sum(gammaln(counts - sigma) + (tau - sigma) * np.log(y) + np.log(c-y) - (counts - sigma) * np.log(y + u))
    return ll

def grad_ll_gbfry(ty, counts, u, sigma, c, tau):
    y = c / (1. + np.exp(-ty))
    grad = ((tau - sigma) / y - 1. / (c - y) - (counts - sigma) / (y + u)) * y * (c - y) / c
    return grad

def estimate_tau(counts):
    sorted_counts = np.sort(counts)[::-1]
    lr = LinearRegression().fit(np.log(np.arange(100, 200)).reshape(-1, 1), np.log(sorted_counts[100:200]))
    tau = -1 / lr.coef_[0]
    return tau

def init_fit_gbfry(counts):
    eta = np.random.gamma(shape=1000., scale=1.)
    sigma = np.sum(counts == 1) / counts.size
    tau = estimate_tau(counts)
    W_sum = np.sum(sample_gbfry(eta, sigma, 1.0, tau, T=1e-8))
    u = np.random.gamma(shape=np.sum(counts), scale=1./W_sum)
    ty = np.random.randn(counts.size)
    return ty, u, eta, sigma, tau

def hmc_step(x0, U, grad_U, eps, L):
    x = x0
    r = np.random.randn(x.size)
    H_0 = U(x) + 0.5 * np.sum(r*r)
    for _ in range(L):
        r -= 0.5 * eps * grad_U(x)
        x += eps * r
        r -= 0.5 * eps * grad_U(x)
    H = U(x) + 0.5 * np.sum(r*r)
    if np.random.rand() > np.exp(np.min(H_0 - H, 0)):
        x = x0
    return x

def fit_gbfry(counts, n_samples, eps=5e-2, L=30):
    print("Init parameters..")
    ty, u, eta, sigma, tau = init_fit_gbfry(counts)
    c = 1.0
    print(f"Initial parameter values - u: {u}, eta: {eta}, sigma: {sigma}, tau: {tau}")

    log_u = np.log(u)
    log_eta = np.log(eta)
    logit_sigma = np.log(sigma) - np.log(1. - sigma)
    delta = tau - sigma
    log_delta = np.log(delta)

    results = {
        "ll": [],
        "eta": [],
        "sigma": [],
        "tau": [],
        "u": [],
        "y": [],
    }

    for _ in trange(n_samples):
        # sample ty
        U = lambda x: -ll_gbfry(counts, x, u, eta, sigma, c, tau)
        grad_U = lambda x: -grad_ll_gbfry(x, counts, u, sigma, c, tau)
        ty = hmc_step(ty, U, grad_U, eps, L)

        # sample eta
        log_eta_new = log_eta + 0.01 * np.random.randn()
        eta_new = np.exp(log_eta_new)
        log_r = ll_gbfry(counts, ty, u, eta_new, sigma, c, tau) - 0.5 * log_eta_new**2 \
            - ll_gbfry(counts, ty, u, eta, sigma, c, tau) + 0.5 * log_eta**2
        if np.random.rand() < np.exp(np.min(log_r, 0)):
            log_eta = log_eta_new
            eta = eta_new

        # sample sigma
        logit_sigma_new = logit_sigma + 0.01 * np.random.randn()
        sigma_new = 1. / (1. + np.exp(-logit_sigma_new))
        tau_new = sigma_new + delta;
        log_r = ll_gbfry(counts, ty, u, eta, sigma_new, c, tau_new) - 0.5 * logit_sigma_new**2 \
            - ll_gbfry(counts, ty, u, eta, sigma, c, tau) + 0.5 * logit_sigma**2
        if np.random.rand() < np.exp(np.min(log_r, 0)):
            logit_sigma = logit_sigma_new
            sigma = sigma_new
            tau = tau_new

        # sample delta
        log_delta_new = log_delta + 0.01 * np.random.randn()
        delta_new = np.exp(log_delta_new)
        tau_new = sigma + delta_new
        log_r = ll_gbfry(counts, ty, u, eta, sigma, c, tau_new) - 0.5 * log_delta_new**2 \
            - ll_gbfry(counts, ty, u, eta, sigma, c, tau) + 0.5 * log_delta**2
        if np.random.rand() < np.exp(np.min(log_r, 0)):
            delta = delta_new
            log_delta = log_delta_new
            tau = tau_new

        # sample u 
        log_u_new = log_u + 0.01 * np.random.randn()
        u_new = np.exp(log_u_new)
        log_r = ll_gbfry(counts, ty, u_new, eta, sigma, c, sigma+delta) - 0.5 * log_u_new**2 \
            - ll_gbfry(counts, ty, u, eta, sigma, c, sigma+delta) + 0.5 * log_u**2
        if np.random.rand() < np.exp(np.min(log_r, 0)):
            u = u_new
            log_u = log_u_new 

        #compute ll
        ll = ll_gbfry(counts, ty, u, eta, sigma, c, tau)

        results["ll"].append(ll)
        results["eta"].append(eta)
        results["sigma"].append(sigma)
        results["tau"].append(tau)
        results["u"].append(u)
        results["y"].append(c / (1. + np.exp(-ty)))

    return results