import numba as nb
import numpy as np
import scipy


random = np.random.RandomState()
scipy.stats.rv_continuous.random_state = random


def set_seed(seed):
    random.seed(seed)
    numba_set_seed(seed)


@nb.njit
def numba_set_seed(seed):
    np.random.seed(seed)
