import tensorflow as tf
from numba import njit
import numpy as np


@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ y_))
    for _ in range(11):
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / alpha_de
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / beta_de
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
                + N / 2.0 * np.log(beta) \
                - 0.5 * np.sum(np.log(alpha + beta * s)) \
                - beta / 2.0 * beta_de \
                - alpha / 2.0 * alpha_de \
                - N / 2.0 * np.log(2 * np.pi)
    return evidence / N


# D = 20, N = 50
f_tmp = np.random.randn(20, 50).astype(np.float64)
each_evidence(np.random.randint(0, 2, 50).astype(np.float64), f_tmp, f_tmp.transpose(),
                np.eye(20, dtype=np.float64), np.ones(20, dtype=np.float64), np.eye(20, dtype=np.float64), 50,
                20)


def LogME(f: tf.Tensor, y: tf.Tensor, regression=False):
    f = f.numpy().astype(np.float64)
    y = y.numpy()
    if regression:
        y = y.numpy().astype(np.float64)

    fh = f
    f = f.transpose()
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

    evidences = []
    if regression:
        K = y.shape[1]
        for i in range(K):
            y_ = y[:, i]
            evidence = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
    else:
        K = int(y.max() + 1)
        for i in range(K):
            y_ = (y == i).astype(np.float64)
            evidence = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
    return np.mean(evidences)