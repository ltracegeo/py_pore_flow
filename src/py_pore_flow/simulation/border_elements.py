import numpy as np
from numba import njit, types, int32

class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements
    """

    def __init__(self, n_pores, n_throats):
        self.throats_set = SparseSet(n_throats)
        self.pores_set = SparseSet(n_pores)

    def set_throats_ids(self, throat_ids):
        self.throats_set.reset()
        self.throats_set.add(throat_ids)

    def set_pores_ids(self, pore_ids):
        self.pores_set.reset()
        self.pores_set.add(pore_ids)

    def get_throats_ids(self):
        return self.throats_set.list()

    def get_pores_ids(self):
        return self.pores_set.list()

    def add_throats_ids(self, throat_ids):
        self.throats_set.add(throat_ids)

    def add_pores_ids(self, pore_ids):
        self.pores_set.add(pore_ids)

    def remove_throats_ids(self, throat_ids):
        self.throats_set.remove(throat_ids)

    def remove_pores_ids(self, pore_ids):
        self.pores_set.remove(pore_ids)


class SparseSet:
    def __init__(self, n: int):
        self.dense = np.empty(n, dtype=np.int32)
        self.sparse = np.empty(n, dtype=np.int32)  # no init needed
        self.count = np.int32(0)
        self.capacity = np.int32(n)

    def reset(self):
        self.count = np.int32(0)

    def contains(self, x: int) -> bool:
        i = self.sparse[x]
        return 0 <= i < self.count and self.dense[i] == x

    def add(self, x_array: np.ndarray) -> None:
        self.count = self._add(x_array, self.sparse, self.dense, self.count)

    @staticmethod
    @njit
    def _add(x_array, sparse, dense, count):
        for x in x_array:
            i = sparse[x]
            if 0 <= i < count and dense[i] == x:
                continue
            sparse[x] = count
            dense[count] = x
            count += 1
        return count

    def remove(self, x_array: np.ndarray) -> None:
        self.count = self._remove(x_array, self.sparse, self.dense, self.count)

    @staticmethod
    @njit
    def _remove(x_array, sparse, dense, count):
        for x in x_array:
            i = sparse[x]
            if not (0 <= i < count and dense[i] == x):
                continue
            last = dense[count - 1]
            dense[i] = last
            sparse[last] = i
            count -= 1
        return count

    def list(self):
        list_view = self.dense[: self.count]
        list_view.flags.writeable
        return list_view

    def clear(self) -> None:
        self.count = 0