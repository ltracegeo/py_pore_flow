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
        if len(throats_ids) > 0:
            pcs = self.sign * self.throat_entry_pressure[throats_ids]
            # Since IndexedPriorityTree doesn't have a dedicated "insert_all" that doesn't clear, we can just iterate.
            # But we can also use set_all if it's the only elements, but add_throats_ids is for additions.
            # Actually, `_dt_bulk_insert` doesn't clear the tree! `set_all` calls clear(), then `_dt_bulk_insert`.
            # Let's use loop for add since add is usually small, but I can add insert_all if needed.
            for i in range(len(throats_ids)):
                self.throats_tree.insert(throats_ids[i], pcs[i])

    def add_pores_ids(self, pores_ids):
        if len(pores_ids) > 0:
            pcs = self.sign * self.pore_entry_pressure[pores_ids]
            for i in range(len(pores_ids)):
                self.pores_tree.insert(pores_ids[i], pcs[i])

    def remove_throats_ids(self, throat_ids):
        self.throats_tree.remove_all(throat_ids)

    def remove_pores_ids(self, pore_ids):
        self.pores_tree.remove_all(pore_ids)

    def sync_throat_pcs(self, throat_ids):
        if len(throat_ids) > 0:
            pcs = self.sign * self.throat_entry_pressure[throat_ids]
            self.throats_tree.update_all(throat_ids, pcs)

    def sync_pore_pcs(self, pore_ids):
        if len(pore_ids) > 0:
            pcs = self.sign * self.pore_entry_pressure[pore_ids]
            self.pores_tree.update_all(pore_ids, pcs)

    def peek_best(self):
        min_p_id, min_p_pc = (-1, np.inf)
        if not self.pores_tree.is_empty:
            min_p_id, min_p_pc = self.pores_tree.peek_min()
            
        min_t_id, min_t_pc = (-1, np.inf)
        if not self.throats_tree.is_empty:
            min_t_id, min_t_pc = self.throats_tree.peek_min()
            
        if min_p_id == -1 and min_t_id == -1:
            return None
            
        if min_p_pc <= min_t_pc:
            return (PORE, min_p_id, self.sign * min_p_pc)
        else:
            return (THROAT, min_t_id, self.sign * min_t_pc)
