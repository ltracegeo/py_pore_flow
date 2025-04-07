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


def weibull(k, loc, scale, size):
    return scipy.stats.weibull_min.rvs(k, loc=loc, scale=scale, size=size, random_state=random)
