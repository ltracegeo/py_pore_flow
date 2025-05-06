import numba as nb
import numpy as np


class SortedRootTree:
    def __init__(self, throat_entry_pressure, pore_entry_pressure):
        self.throat_entry_pressure = throat_entry_pressure
        self.pore_entry_pressure = pore_entry_pressure

        self.pore_n = len(pore_entry_pressure)
        self.throat_n = len(throat_entry_pressure)
        self.tree_invasion_status, self.tree_element_label, self.tree_elements_n = initialize(
            self.pore_n, self.throat_n
        )

    def get_pressure(self, element_label):
        return _get_pressure(
            element_label,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def add_node(  # nodes should be added to the tree when they change from 0 to 1 tree_invasion_status
        self,
        new_element_pressure,
        new_element_label,  # label is equal to (index+1) for pores, and (index+pore_n+1) for throats
    ):
        _add_node(
            new_element_pressure,
            new_element_label,
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def get_root(self):
        return _get_root(self.tree_element_label)

    def pop_root(self):
        return _pop_root(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def reorder_tree(self):
        _reorder_tree(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def check_tree(self):
        return _check_tree(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )


class DecreasingSortedRootTree(SortedRootTree):
    def add_node(  # nodes should be added to the tree when they change from 0 to 1 tree_invasion_status
        self,
        new_element_pressure,
        new_element_label,  # label is equal to (index+1) for pores, and (index+pore_n+1) for throats
    ):
        _dec_add_node(
            new_element_pressure,
            new_element_label,
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def pop_root(self):
        return _dec_pop_root(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def reorder_tree(self):
        _dec_reorder_tree(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )

    def check_tree(self):
        return _dec_check_tree(
            self.tree_element_label,
            self.tree_elements_n,
            self.throat_entry_pressure,
            self.pore_entry_pressure,
        )


def initialize(pore_n, throat_n):
    tree_invasion_status = np.zeros(pore_n + throat_n + 1, dtype=np.uint8)
    # 0 : uninvaded
    # 1 : uninvaded, adjacent to invaded
    # 2 : invaded
    tree_element_label = np.zeros(pore_n + throat_n + 1, dtype=np.uint32)
    tree_elements_n = np.zeros(1, dtype=np.uint32)

    return tree_invasion_status, tree_element_label, tree_elements_n


@nb.njit(cache=True)
def _get_pressure(
    element_label,
    throat_entry_pressure,
    pore_entry_pressure,
):
    pore_n = pore_entry_pressure.size
    if element_label <= pore_n:
        pore_index = element_label - 1
        pressure = pore_entry_pressure[pore_index]
    else:
        throat_index = element_label - pore_n - 1
        pressure = throat_entry_pressure[throat_index]
    return pressure


@nb.njit(cache=True)
def _add_node(
    new_element_pressure,
    new_element_label,
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    new_n = tree_elements_n[0]
    tree_element_label[new_n] = new_element_label
    changed = True
    while changed is True:
        if new_n == 0:
            changed = False
            continue
        parent_n = (new_n + 1) // 2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = get_pressure(parent_label)
        if parent_pressure > new_element_pressure:
            tree_element_label[new_n] = parent_label
            tree_element_label[parent_n] = new_element_label
            new_n = parent_n
            changed = True
            continue
        else:  # new_element_pressure >= parent_pressure:
            changed = False
            continue
    tree_elements_n[0] += 1


@nb.njit(cache=True)
def _get_root(tree_element_label):
    root_label = tree_element_label[0]
    return root_label


@nb.njit(cache=True)
def _pop_root(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    root_label = tree_element_label[0]
    if root_label == 0:
        return 0

    center_n = 0
    tree_element_label[0] = 0
    left_n = 2 * (center_n + 1) - 1
    left_node = tree_element_label[left_n]
    right_n = 2 * (center_n + 1)
    right_node = tree_element_label[right_n]

    while left_node != 0:
        if right_node != 0:
            if get_pressure(left_node) < get_pressure(right_node):
                tree_element_label[center_n] = left_node
                tree_element_label[left_n] = 0
                center_n = left_n
            else:
                tree_element_label[center_n] = right_node
                tree_element_label[right_n] = 0
                center_n = right_n
        else:
            tree_element_label[center_n] = left_node
            tree_element_label[left_n] = 0
            center_n = left_n
        left_n = 2 * (center_n + 1) - 1
        left_node = tree_element_label[left_n]
        right_n = 2 * (center_n + 1)
        right_node = tree_element_label[right_n]

    last_slot = tree_elements_n[0] - 1
    tree_element_label[center_n] = tree_element_label[last_slot]
    tree_element_label[last_slot] = 0
    center_label = tree_element_label[center_n]
    center_pressure = get_pressure(center_label)

    if center_label != 0:
        changed = True
    else:
        changed = False
    while changed is True:
        if center_n == 0:
            changed = False
        parent_n = (center_n + 1) // 2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = get_pressure(parent_label)
        if parent_label == 0 or parent_pressure > center_pressure:
            tree_element_label[center_n] = parent_label
            tree_element_label[parent_n] = center_label
            center_n = parent_n
            changed = True
            continue
        else:  # moved_element_pressure >= parent_pressure:
            changed = False
            continue

    tree_elements_n[0] -= 1
    return root_label


@nb.njit(cache=True)
def _reorder_tree(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    for starting_n in range(1, tree_elements_n[0]):
        center_n = starting_n
        while center_n > 0:
            parent_n = (center_n + 1) // 2 - 1
            center_label = tree_element_label[center_n]
            parent_label = tree_element_label[parent_n]
            center_pressure = get_pressure(center_label)
            parent_pressure = get_pressure(parent_label)
            if parent_pressure > center_pressure:
                tree_element_label[center_n] = parent_label
                tree_element_label[parent_n] = center_label
                center_n = parent_n
            else:
                break


@nb.njit(cache=True)
def _check_tree(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    for center_n in range(tree_elements_n[0] - 1, 0, -1):
        parent_n = (center_n + 1) // 2 - 1
        center_label = tree_element_label[center_n]
        parent_label = tree_element_label[parent_n]
        center_pressure = get_pressure(center_label)
        parent_pressure = get_pressure(parent_label)
        if parent_pressure > center_pressure:
            return False
    return True


@nb.njit(cache=True)
def _dec_add_node(
    new_element_pressure,
    new_element_label,
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    new_n = tree_elements_n[0]
    tree_element_label[new_n] = new_element_label
    changed = True
    while changed is True:
        if new_n == 0:
            changed = False
            continue
        parent_n = (new_n + 1) // 2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = get_pressure(parent_label)
        if parent_pressure < new_element_pressure:
            tree_element_label[new_n] = parent_label
            tree_element_label[parent_n] = new_element_label
            new_n = parent_n
            changed = True
            continue
        else:  # new_element_pressure >= parent_pressure:
            changed = False
            continue
    tree_elements_n[0] += 1


@nb.njit(cache=True)
def _dec_pop_root(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    root_label = tree_element_label[0]
    if root_label == 0:
        return 0

    center_n = 0
    tree_element_label[0] = 0
    left_n = 2 * (center_n + 1) - 1
    left_node = tree_element_label[left_n]
    right_n = 2 * (center_n + 1)
    right_node = tree_element_label[right_n]

    while left_node != 0:
        if right_node != 0:
            if get_pressure(left_node) < get_pressure(right_node):
                tree_element_label[center_n] = left_node
                tree_element_label[left_n] = 0
                center_n = left_n
            else:
                tree_element_label[center_n] = right_node
                tree_element_label[right_n] = 0
                center_n = right_n
        else:
            tree_element_label[center_n] = left_node
            tree_element_label[left_n] = 0
            center_n = left_n
        left_n = 2 * (center_n + 1) - 1
        left_node = tree_element_label[left_n]
        right_n = 2 * (center_n + 1)
        right_node = tree_element_label[right_n]

    last_slot = tree_elements_n[0] - 1
    tree_element_label[center_n] = tree_element_label[last_slot]
    tree_element_label[last_slot] = 0
    center_label = tree_element_label[center_n]
    center_pressure = get_pressure(center_label)

    if center_label != 0:
        changed = True
    else:
        changed = False
    while changed is True:
        if center_n == 0:
            changed = False
        parent_n = (center_n + 1) // 2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = get_pressure(parent_label)
        if parent_label == 0 or parent_pressure < center_pressure:
            tree_element_label[center_n] = parent_label
            tree_element_label[parent_n] = center_label
            center_n = parent_n
            changed = True
            continue
        else:  # moved_element_pressure >= parent_pressure:
            changed = False
            continue

    tree_elements_n[0] -= 1
    return root_label


@nb.njit(cache=True)
def _dec_reorder_tree(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    for starting_n in range(1, tree_elements_n[0]):
        center_n = starting_n
        while center_n > 0:
            parent_n = (center_n + 1) // 2 - 1
            center_label = tree_element_label[center_n]
            parent_label = tree_element_label[parent_n]
            center_pressure = get_pressure(center_label)
            parent_pressure = get_pressure(parent_label)
            if parent_pressure < center_pressure:
                tree_element_label[center_n] = parent_label
                tree_element_label[parent_n] = center_label
                center_n = parent_n
            else:
                break


@nb.njit(cache=True)
def _dec_check_tree(
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    get_pressure = lambda element_label: _get_pressure(element_label, throat_entry_pressure, pore_entry_pressure)

    for center_n in range(tree_elements_n[0] - 1, 0, -1):
        parent_n = (center_n + 1) // 2 - 1
        center_label = tree_element_label[center_n]
        parent_label = tree_element_label[parent_n]
        center_pressure = get_pressure(center_label)
        parent_pressure = get_pressure(parent_label)
        if parent_pressure < center_pressure:
            return False
    return True
