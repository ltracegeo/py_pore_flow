import numpy as np


class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements
    """

    def __init__(self):
        self.throats_ids = np.array([], np.int64)
        self.pores_ids = np.array([], np.int64)

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
        self.throats_ids = np.delete(self.throats_ids, np.isin(self.throats_ids, throat_ids))

    def remove_pores_ids(self, pore_ids):
        self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, pore_ids))
