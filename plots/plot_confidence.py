import numpy as np
import matplotlib.pyplot as plt

def plot_confidence_interval_gbfry(counts, fit_results):
    n = np.sum(counts)
    predictions = predict_gbfry(fit_results)
    max_count = min(map(len, predictions))
    predictions_np = np.array([arr[:max_count] for arr in predictions])
    quantiles = np.quantile(predictions_np, [0.05, 0.95], axis=0)
    fig, ax = plt.subplots()
    ax.loglog(np.sort(counts)[::-1], label="Truth")
    ax.fill_between(np.arange(quantiles.shape[1]), quantiles[0], quantiles[1], alpha=0.5, label="Confidence interval")
    ax.legend()
    plt.show()