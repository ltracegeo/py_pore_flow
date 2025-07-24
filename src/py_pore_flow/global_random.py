import numba as nb
import numpy as np
import scipy


random = np.random.RandomState()


def set_seed(seed):
    random.seed(seed)
    numba_set_seed(seed)


@nb.njit(cache=True)
def numba_set_seed(seed):
    np.random.seed(seed)


def weibull(n_samples, delta, gamma, center, range):
    theta_min = max(center - range / 2, 0)
    theta_max = min(center + range / 2, np.pi)
    x = random.uniform(0, 1, size=n_samples)
    if delta < 0 or gamma < 0:
        term = x
    else:
        term = (-delta * np.log(1.0 - x * (1.0 - np.exp(-1.0 / delta)))) ** (1 / gamma)

    return (theta_max - theta_min) * term + theta_min
