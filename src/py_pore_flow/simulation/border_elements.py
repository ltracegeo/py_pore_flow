from py_pore_flow.equations.sparse_set import SparseSet

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
