import numpy as np
from py_pore_flow.simulation.indexed_priority_tree import IndexedPriorityTree
from py_pore_flow.network.network import PORE, THROAT

class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements
    """

    def __init__(self, n_pores, n_throats, pore_entry_pressure, throat_entry_pressure, negate_pc=False):
        self.pore_entry_pressure = pore_entry_pressure
        self.throat_entry_pressure = throat_entry_pressure
        self.sign = -1.0 if negate_pc else 1.0

        self.pores_tree = IndexedPriorityTree(n_pores)
        self.throats_tree = IndexedPriorityTree(n_throats)

    def set_throats_ids(self, throats_ids):
        self.throats_tree.clear()
        if len(throats_ids) > 0:
            self.throats_tree.set_all(throats_ids, self.sign * self.throat_entry_pressure[throats_ids])

    def set_pores_ids(self, pores_ids):
        self.pores_tree.clear()
        if len(pores_ids) > 0:
            self.pores_tree.set_all(pores_ids, self.sign * self.pore_entry_pressure[pores_ids])

    def get_throats_ids(self):
        return self.throats_tree.get_all_ids()

    def get_pores_ids(self):
        return self.pores_tree.get_all_ids()

    def add_throats_ids(self, throats_ids):
        for id in throats_ids:
            self.throats_tree.insert(id, self.sign * self.throat_entry_pressure[id])

    def add_pores_ids(self, pores_ids):
        for id in pores_ids:
            self.pores_tree.insert(id, self.sign * self.pore_entry_pressure[id])

    def remove_throats_ids(self, throat_ids):
        for id in throat_ids:
            self.throats_tree.remove(id)

    def remove_pores_ids(self, pore_ids):
        for id in pore_ids:
            self.pores_tree.remove(id)

    def sync_throat_pcs(self, throat_ids):
        for id in throat_ids:
            self.throats_tree.update_pc(id, self.sign * self.throat_entry_pressure[id])

    def sync_pore_pcs(self, pore_ids):
        for id in pore_ids:
            self.pores_tree.update_pc(id, self.sign * self.pore_entry_pressure[id])

    def peek_best(self):
        min_p_id, min_p_pc = (-1, np.inf)
        if not self.pores_tree.is_empty:
            min_p_id, min_p_pc = self.pores_tree.peek_min()
            
        min_t_id, min_t_pc = (-1, np.inf)
        if not self.throats_tree.is_empty:
            min_t_id, min_t_pc = self.throats_tree.peek_min()
            
        if min_p_id == -1 and min_t_id == -1:
            return None
            
        if min_p_pc < min_t_pc:
            return (PORE, min_p_id, self.sign * min_p_pc)
        else:
            return (THROAT, min_t_id, self.sign * min_t_pc)
