import numba as nb
import numpy as np
@nb.njit
def njit_isin(array_a, array_b):
    result = np.empty(array_a.size, dtype=np.bool_)
    set_b = set(array_b)
    for i in range(array_a.size):
        result[i] = array_a[i] in set_b
    return result
a = np.array([1, 2, 3])
b = np.array([2, 4])
print(njit_isin(a, b))
