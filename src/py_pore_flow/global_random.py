import numba as nb
import numpy as np
import scipy.stats


random = np.random.RandomState()


def set_seed(seed):
    random.seed(seed)
    numba_set_seed(seed)


@nb.njit(cache=True)
def numba_set_seed(seed):
    np.random.seed(seed)


def gaussian(n_samples, sigma, center, range_):
    theta_min = max(center - range_ / 2.0, 0)
    theta_max = min(center + range_ / 2.0, np.pi)

    if sigma < 0:
        x = np.random.uniform(0, 1, size=n_samples)
        samples = (theta_max - theta_min) * x + theta_min
    else:
        scale = sigma * np.pi / 180.0
        a = (theta_min - center) / scale
        b = (theta_max - center) / scale
        samples = scipy.stats.truncnorm.rvs(a, b, loc=center, scale=scale, size=n_samples)

    return samples


def weibull(n_samples, delta, gamma, center, range):
    theta_min = max(center - range / 2, 0)
    theta_max = min(center + range / 2, np.pi)
    x = random.uniform(0, 1, size=n_samples)
    if delta < 0 or gamma < 0:
        term = x
    else:
        term = (-delta * np.log(1.0 - x * (1.0 - np.exp(-1.0 / delta)))) ** (1 / gamma)

    return (theta_max - theta_min) * term + theta_min
