import unittest
from numpy.testing import assert_almost_equal, assert_equal

import numpy as np

from py_pore_flow.simulation.algorithms.sorted_root_tree import DecreasingSortedRootTree, SortedRootTree


class SortRootTreeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pore_n = 20
        throat_n = 26

        np.random.seed(42)
        cls.pore_entry_pressure = np.random.random(pore_n)
        cls.pore_trapped = np.full(pore_n, False, np.bool_)

        cls.throat_conns = np.array(
            [
                [1, 4],
                [2, 4],
                [5, 3],
                [7, 4],
                [8, 4],
                [9, 4],
                [9, 10],
                [4, 10],
                [10, 3],
                [12, 13],
                [14, 13],
                [13, 18],
                [13, 16],
                [13, 9],
                [16, 20],
                [9, 15],
                [20, 15],
                [15, 10],
                [10, 11],
                [15, 11],
                [15, 19],
                [19, 17],
                [11, 17],
                [6, 11],
                [7, 8],
                [8, 9],
            ]
        )
        cls.throat_entry_pressure = np.random.random(throat_n)
        cls.throat_trapped = np.full(pore_n, False, np.bool_)

        cls.pore_conns_n = np.zeros(pore_n, dtype=np.int16)
        for i in cls.throat_conns.flatten():
            cls.pore_conns_n[i - 1] += 1
        cls.pore_conns = np.zeros((pore_n, cls.pore_conns_n.max()), dtype=np.int16)
        cls.pore_conns_n = np.zeros(pore_n, dtype=np.int16)
        for i_throat, (i_pore, j_pore) in enumerate(cls.throat_conns):
            throat_element_index = i_throat + pore_n + 1
            i_index = cls.pore_conns_n[i_pore - 1]
            j_index = cls.pore_conns_n[j_pore - 1]
            cls.pore_conns[i_pore - 1, i_index] = throat_element_index
            cls.pore_conns[j_pore - 1, j_index] = throat_element_index
            cls.pore_conns_n[i_pore - 1] += 1
            cls.pore_conns_n[j_pore - 1] += 1

    def test_sort(self):
        sorted_root_tree = SortedRootTree(
            self.throat_entry_pressure,
            self.pore_entry_pressure,
            self.throat_trapped,
            self.pore_trapped,
        )
        pore_id = 5
        pore_pressure = sorted_root_tree.get_pressure(pore_id + 1)
        assert_almost_equal(pore_pressure, 0.15599452033620265)
        sorted_root_tree.add_node(pore_pressure, pore_id + 1)

        throat_id = 3
        throat_pressure = sorted_root_tree.get_pressure(throat_id + sorted_root_tree.pore_n + 1)
        assert_almost_equal(throat_pressure, 0.3663618432936917)
        sorted_root_tree.add_node(throat_pressure, throat_id + sorted_root_tree.pore_n + 1)

        element_type, element_id, pressure = sorted_root_tree.get_root_element(
            0, self.pore_trapped, self.throat_trapped
        )
        assert_equal(element_type, 0)
        assert_equal(element_id, 5)
        assert_almost_equal(pressure, 0.15599452033620265)

    def test_decreasing_sort(self):
        sorted_root_tree = DecreasingSortedRootTree(
            self.throat_entry_pressure,
            self.pore_entry_pressure,
            self.throat_trapped,
            self.pore_trapped,
        )

        # Add elements
        pore_id = 5
        pore_pressure = sorted_root_tree.get_pressure(pore_id + 1)
        assert_almost_equal(pore_pressure, 0.15599452033620265)
        sorted_root_tree.add_node(pore_pressure, pore_id + 1)

        throat_id = 3
        throat_pressure = sorted_root_tree.get_pressure(throat_id + sorted_root_tree.pore_n + 1)
        assert_almost_equal(throat_pressure, 0.3663618432936917)
        sorted_root_tree.add_node(throat_pressure, throat_id + sorted_root_tree.pore_n + 1)

        throat_id = 7
        throat_pressure = sorted_root_tree.get_pressure(throat_id + sorted_root_tree.pore_n + 1)
        assert_almost_equal(throat_pressure, 0.5142344384136116)
        sorted_root_tree.add_node(throat_pressure, throat_id + sorted_root_tree.pore_n + 1)

        element_type, element_id, pressure = sorted_root_tree.get_root_element(
            1000, self.pore_trapped, self.throat_trapped
        )
        assert_equal(element_type, 1)
        assert_equal(element_id, 7)
        assert_almost_equal(pressure, 0.5142344384136116)

        # Change pressure
        self.throat_entry_pressure[7] = 0.14
        self.throat_entry_pressure[3] = 0.13
        sorted_root_tree.reorder_tree()
        assert(sorted_root_tree.check_tree())

        element_type, element_id, pressure = sorted_root_tree.get_root_element(
            1000, self.pore_trapped, self.throat_trapped
        )
        assert_equal(element_type, 0)
        assert_equal(element_id, 5)
        assert_almost_equal(pressure, 0.15599452033620265)
        sorted_root_tree.pop_root()
        assert(sorted_root_tree.check_tree())

        element_type, element_id, pressure = sorted_root_tree.get_root_element(
            1000, self.pore_trapped, self.throat_trapped
        )
        assert_equal(element_type, 1)
        assert_equal(element_id, 7)
        assert_almost_equal(pressure, 0.14)
