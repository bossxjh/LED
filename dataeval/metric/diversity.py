import numpy as np
from sklearn.metrics import pairwise_distances
from dataeval.api import extract_features
import os

def compute_task_diversity_entropy(X, sigma=0.1, eps=1e-12):
    """
    Kernel-based entropy estimator:
      H_hat = - (1/N) * sum_i log( (1/N) * sum_j K_sigma(x_i, x_j) )
    """
    X = np.asarray(X)
    N = X.shape[0]

    # pairwise euclidean distances
    dists = pairwise_distances(X, X, metric="euclidean")

    # choose sigma if not provided: median of upper triangular distances
    if sigma is None:
        iu = np.triu_indices(N, k=1)
        if iu[0].size > 0:
            sigma = np.median(dists[iu])
            if sigma == 0:
                nonzero = dists[iu][dists[iu] > 0]
                sigma = np.median(nonzero) if nonzero.size > 0 else 1.0
        else:
            sigma = 1.0

    K = np.exp(-(dists**2) / (2.0 * (sigma ** 2)))
    inner = K.mean(axis=1)
    H_hat = - np.mean(np.log(inner + eps))
    return H_hat, sigma