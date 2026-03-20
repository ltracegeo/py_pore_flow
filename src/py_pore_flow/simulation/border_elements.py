import numpy as np
from py_pore_flow.network.network import PORE, THROAT

class PCBuckets:
    def __init__(self, pc_array, min_pc, max_pc, num_bins=10000, negate_pc=False):
        self.pc_array = pc_array
        self.min_pc = min_pc
        self.max_pc = max_pc
        if self.max_pc <= self.min_pc:
            self.max_pc = self.min_pc + 1.0
        self.num_bins = num_bins
        self.bin_width = (self.max_pc - self.min_pc) / self.num_bins
        self.bins = [set() for _ in range(self.num_bins + 1)]
        self.element_bin = {} # id -> bin_idx
        self.negate_pc = negate_pc

    def _get_bin_idx(self, pc):
        pc_val = -pc if self.negate_pc else pc
        if pc_val <= self.min_pc: return 0
        if pc_val >= self.max_pc: return self.num_bins
        return int((pc_val - self.min_pc) / self.bin_width)

    def add(self, element_id):
        if element_id in self.element_bin:
            self.update_pc(element_id)
            return
        pc = self.pc_array[element_id]
        idx = self._get_bin_idx(pc)
        self.bins[idx].add(element_id)
        self.element_bin[element_id] = idx

    def remove(self, element_id):
        idx = self.element_bin.pop(element_id, None)
        if idx is not None:
            self.bins[idx].discard(element_id)

    def update_pc(self, element_id):
        if element_id in self.element_bin:
            old_idx = self.element_bin[element_id]
            pc = self.pc_array[element_id]
            new_idx = self._get_bin_idx(pc)
            if old_idx != new_idx:
                self.bins[old_idx].discard(element_id)
                self.bins[new_idx].add(element_id)
                self.element_bin[element_id] = new_idx

    def peek_best(self):
        for i in range(len(self.bins)):
            if self.bins[i]:
                best_id = -1
                best_pc_val = float('inf')
                best_real_pc = 0.0
                for elem_id in self.bins[i]:
                    real_pc = self.pc_array[elem_id]
                    pc_val = -real_pc if self.negate_pc else real_pc
                    if pc_val < best_pc_val:
                        best_pc_val = pc_val
                        best_real_pc = real_pc
                        best_id = elem_id
                    elif pc_val == best_pc_val and elem_id < best_id:
                        best_id = elem_id
                return best_id, best_real_pc
        return -1, np.inf

    def is_empty(self):
        return len(self.element_bin) == 0
    
    def clear(self):
        for s in self.bins:
            s.clear()
        self.element_bin.clear()

    def set_all(self, element_ids):
        self.clear()
        for eid in element_ids:
            self.add(eid)
            
    def get_all_ids(self):
        return np.array(list(self.element_bin.keys()), dtype=np.int64)


class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements using PC Bucketing.
    """

    def __init__(self, pore_entry_pressure, throat_entry_pressure, min_pc, max_pc, negate_pc=False):
        self.pore_entry_pressure = pore_entry_pressure
        self.throat_entry_pressure = throat_entry_pressure
        self.sign = -1.0 if negate_pc else 1.0

        self.pores_tree = PCBuckets(pore_entry_pressure, min_pc, max_pc, num_bins=10000, negate_pc=negate_pc)
        self.throats_tree = PCBuckets(throat_entry_pressure, min_pc, max_pc, num_bins=10000, negate_pc=negate_pc)

    def set_throats_ids(self, throats_ids):
        self.throats_tree.set_all(throats_ids)

    def set_pores_ids(self, pores_ids):
        self.pores_tree.set_all(pores_ids)

    def get_throats_ids(self):
        return self.throats_tree.get_all_ids()

    def get_pores_ids(self):
        return self.pores_tree.get_all_ids()

    def add_throats_ids(self, throats_ids):
        for id in throats_ids:
            self.throats_tree.add(id)

    def add_pores_ids(self, pores_ids):
        for id in pores_ids:
            self.pores_tree.add(id)

    def remove_throats_ids(self, throat_ids):
        for id in throat_ids:
            self.throats_tree.remove(id)

    def remove_pores_ids(self, pore_ids):
        for id in pore_ids:
            self.pores_tree.remove(id)

    def sync_throat_pcs(self, throat_ids):
        for id in throat_ids:
            self.throats_tree.update_pc(id)

    def sync_pore_pcs(self, pore_ids):
        for id in pore_ids:
            self.pores_tree.update_pc(id)

    def peek_best(self):
        min_p_id, min_p_pc = (-1, np.inf)
        if not self.pores_tree.is_empty():
            min_p_id, min_p_real = self.pores_tree.peek_best()
            min_p_pc = min_p_real * self.sign
            
        min_t_id, min_t_pc = (-1, np.inf)
        if not self.throats_tree.is_empty():
            min_t_id, min_t_real = self.throats_tree.peek_best()
            min_t_pc = min_t_real * self.sign
            
        if min_p_id == -1 and min_t_id == -1:
            return None
            
        if min_p_pc <= min_t_pc:
            return (PORE, min_p_id, self.sign * min_p_pc)
        else:
            return (THROAT, min_t_id, self.sign * min_t_pc)
