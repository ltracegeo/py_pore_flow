import numpy as np
from numba import njit

#@njit
def _get_pressure(
    element_label,
    throat_entry_pressure,
    pore_entry_pressure,
):
    if element_label == 0:
        return np.inf
    pore_n = pore_entry_pressure.size
    if element_label <= pore_n:
        pore_index = element_label - 1
        pressure = pore_entry_pressure[pore_index]
    else:
        throat_index = element_label - pore_n - 1
        pressure = throat_entry_pressure[throat_index]
    return pressure

#@njit
def _add_node(
    new_element_pressure, 
    new_element_label,
    tree_element_label,
    tree_elements_n,
    throat_entry_pressure,
    pore_entry_pressure,
):
    new_n = tree_elements_n[0]
    tree_element_label[new_n] = new_element_label
    changed = True
    while changed is True:
        if new_n == 0:
            changed = False
            continue
        parent_n = (new_n+1)//2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = _get_pressure(
            parent_label,
            throat_entry_pressure,
            pore_entry_pressure,
        )

        if new_element_pressure < parent_pressure:
            tree_element_label[new_n] = parent_label
            tree_element_label[parent_n] = new_element_label
            new_n = parent_n
            changed = True
            continue
        else: # new_element_pressure >= parent_pressure:
            changed = False
            continue
    tree_elements_n[0] += 1


#@njit
def _pop_root(
tree_element_label,
tree_elements_n,
throat_entry_pressure,
pore_entry_pressure,
):
    root_label = tree_element_label[0]
    if root_label == 0:
        return 0

    center_n = 0
    tree_element_label[0] = 0
    left_n = 2 * (center_n + 1) - 1
    left_node = tree_element_label[left_n]
    right_n = 2 * (center_n + 1)
    right_node = tree_element_label[right_n]

    #print(left_node, right_node, tree_element_label[0])
    while left_node != 0:
        #print(left_node, right_node, tree_element_label[0])
        if right_node != 0:
            if _get_pressure(
                left_node,
                throat_entry_pressure,
                pore_entry_pressure,
                ) < _get_pressure(
                right_node,
                throat_entry_pressure,
                pore_entry_pressure,
            ):
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

    last_slot = tree_elements_n[0]-1
    tree_element_label[center_n] = tree_element_label[last_slot]
    tree_element_label[last_slot] = 0
    center_label = tree_element_label[center_n]
    center_pressure = _get_pressure(
        center_label,
        throat_entry_pressure,
        pore_entry_pressure,
    )

    if center_label != 0:
        changed = True
    else:
        changed = False
    while changed is True:
        parent_n = (center_n+1)//2 - 1
        parent_label = tree_element_label[parent_n]
        parent_pressure = _get_pressure(
            parent_label,
            throat_entry_pressure,
            pore_entry_pressure,
        )
        if parent_label == 0 or center_pressure < parent_pressure:
            tree_element_label[center_n] = parent_label
            tree_element_label[parent_n] = center_label
            center_n = parent_n
            changed = True
            if center_n == 0:
                changed = False
        else: # moved_element_pressure >= parent_pressure:
            changed = False
        

    tree_elements_n[0] -= 1
    #print(left_node, right_node, tree_element_label[0])
    return root_label


#@njit
def _reorder_tree(
        tree_element_label,
        tree_elements_n,
        throat_entry_pressure,
        pore_entry_pressure,
    ):
    for starting_n in range(1, tree_elements_n[0]):
        center_n = starting_n
        while center_n > 0:
            parent_n = (center_n+1)//2 - 1
            center_label = tree_element_label[center_n]
            parent_label = tree_element_label[parent_n]
            center_pressure = _get_pressure(
                center_label,
                throat_entry_pressure,
                pore_entry_pressure,
            )
            parent_pressure = _get_pressure(
                parent_label,
                throat_entry_pressure,
                pore_entry_pressure,
            )
            if parent_pressure > center_pressure:
                tree_element_label[center_n] = parent_label
                tree_element_label[parent_n] = center_label
                center_n = parent_n
            else:
                break

#@njit
def _check_tree(
        tree_element_label,
        tree_elements_n,
        throat_entry_pressure,
        pore_entry_pressure,
    ):
    for center_n in range(tree_elements_n[0]-1, 0, -1):
        parent_n = (center_n+1)//2 - 1
        center_label = tree_element_label[center_n]
        parent_label = tree_element_label[parent_n]
        center_pressure = _get_pressure(
            center_label,
            throat_entry_pressure,
            pore_entry_pressure,
        )
        parent_pressure = _get_pressure(
            parent_label,
            throat_entry_pressure,
            pore_entry_pressure,
        )
        if parent_pressure > center_pressure:
            #print(parent_pressure, center_pressure, center_label, parent_label, center_n, parent_n)
            return False
    return True

