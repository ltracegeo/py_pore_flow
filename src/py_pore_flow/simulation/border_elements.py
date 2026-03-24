import numpy as np


class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements.

    Uses the in_border boolean arrays (maintained by cycle_simulation's Numba
    functions) for O(1) membership tests, avoiding np.isin in remove operations.
    """

    def __init__(self, pore_in_border=None, throat_in_border=None):
        self.throats_ids = np.array([], np.int64)
        self.pores_ids = np.array([], np.int64)
        # References to the network's in_border boolean arrays for O(1) lookups.
        self._pore_in_border = pore_in_border
        self._throat_in_border = throat_in_border

    def set_throats_ids(self, throats_ids):
        self.throats_ids = throats_ids

    def set_pores_ids(self, pores_ids):
        self.pores_ids = pores_ids

    def get_throats_ids(self):
        return self.throats_ids

    def get_pores_ids(self):
        return self.pores_ids

    def add_throats_ids(self, throats_ids):
        self.throats_ids = np.concatenate((self.throats_ids, throats_ids))

    def add_pores_ids(self, pores_ids):
        self.pores_ids = np.concatenate((self.pores_ids, pores_ids))

    def remove_throats_ids(self, throat_ids):
        if len(throat_ids) == 0:
            return
        # Use in_border flag for O(n) filter without np.isin when available.
        if self._throat_in_border is not None:
            self.throats_ids = self.throats_ids[self._throat_in_border[self.throats_ids]]
        else:
            self.throats_ids = np.delete(self.throats_ids, np.isin(self.throats_ids, throat_ids))

    def remove_pores_ids(self, pore_ids):
        if len(pore_ids) == 0:
            return
        # Use in_border flag for O(n) filter without np.isin when available.
        if self._pore_in_border is not None:
            self.pores_ids = self.pores_ids[self._pore_in_border[self.pores_ids]]
        else:
            self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, pore_ids))
