from numba import njit
import numpy as np

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

    def remove_where(self, mask_a, mask_b):
        """Remove elements where mask_a[element] or mask_b[element] is True."""
        self.count = self._remove_where(self.sparse, self.dense, self.count, mask_a, mask_b)

    @staticmethod
    @njit
    def _remove_where(sparse, dense, count, mask_a, mask_b):
        i = np.int32(0)
        while i < count:
            x = dense[i]
            if mask_a[x] or mask_b[x]:
                last = dense[count - 1]
                dense[i] = last
                sparse[last] = i
                count -= 1
            else:
                i += 1
        return count

    def remove_where3(self, mask_a, mask_b, mask_c):
        """Remove elements where any of mask_a/mask_b/mask_c is True for that element."""
        self.count = self._remove_where3(self.sparse, self.dense, self.count, mask_a, mask_b, mask_c)

    @staticmethod
    @njit
    def _remove_where3(sparse, dense, count, mask_a, mask_b, mask_c):
        i = np.int32(0)
        while i < count:
            x = dense[i]
            if mask_a[x] or mask_b[x] or mask_c[x]:
                last = dense[count - 1]
                dense[i] = last
                sparse[last] = i
                count -= 1
            else:
                i += 1
        return count

    def set_from_array(self, x_array):
        """Reset and bulk-load from a numpy array (assumed unique)."""
        self.count = self._set_from_array(x_array, self.sparse, self.dense)

    @staticmethod
    @njit
    def _set_from_array(x_array, sparse, dense):
        count = np.int32(0)
        for x in x_array:
            sparse[x] = count
            dense[count] = x
            count += 1
        return count

    def list(self):
        list_view = self.dense[: self.count]
        list_view.flags.writeable = False
        return list_view

    def clear(self) -> None:
        self.count = 0
