import numpy as np


class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements.

    Uses a Python set internally for O(1) add/remove, and lazily converts to a
    numpy array only when get_*_ids() is called.
    """

    def __init__(self):
        self._throats_set = set()
        self._pores_set = set()
        self._throats_cache = None
        self._pores_cache = None

    def set_throats_ids(self, throats_ids):
        self._throats_set = set(throats_ids.tolist()) if len(throats_ids) else set()
        self._throats_cache = None

    def set_pores_ids(self, pores_ids):
        self._pores_set = set(pores_ids.tolist()) if len(pores_ids) else set()
        self._pores_cache = None

    def get_throats_ids(self):
        if self._throats_cache is None:
            self._throats_cache = np.fromiter(self._throats_set, dtype=np.int64, count=len(self._throats_set))
        return self._throats_cache

    def get_pores_ids(self):
        if self._pores_cache is None:
            self._pores_cache = np.fromiter(self._pores_set, dtype=np.int64, count=len(self._pores_set))
        return self._pores_cache

    def add_throats_ids(self, throats_ids):
        if len(throats_ids):
            self._throats_set.update(throats_ids.tolist())
            self._throats_cache = None

    def add_pores_ids(self, pores_ids):
        if len(pores_ids):
            self._pores_set.update(pores_ids.tolist())
            self._pores_cache = None

    def remove_throats_ids(self, throat_ids):
        if len(throat_ids):
            self._throats_set.difference_update(throat_ids.tolist())
            self._throats_cache = None

    def remove_pores_ids(self, pore_ids):
        if len(pore_ids):
            self._pores_set.difference_update(pore_ids.tolist())
            self._pores_cache = None
