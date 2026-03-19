"""Tests for BorderElements backed by IndexedPriorityTree."""

import unittest

import numpy as np

from py_pore_flow.network.constants import PORE, THROAT
from py_pore_flow.simulation.border_elements import BorderElements


class TestBorderElementsBasic(unittest.TestCase):
    """Construction, add, remove, and ID retrieval."""

    def _make(self, n_pores: int = 10, n_throats: int = 10):
        """Create a BorderElements with dummy entry pressure arrays."""
        pore_pc = np.arange(n_pores, dtype=np.float64) * 100.0
        throat_pc = np.arange(n_throats, dtype=np.float64) * 200.0
        return BorderElements(pore_pc, throat_pc), pore_pc, throat_pc

    def test_empty_after_construction(self):
        be, _, _ = self._make()
        self.assertTrue(be.is_empty)
        self.assertEqual(be.pore_count, 0)
        self.assertEqual(be.throat_count, 0)
        np.testing.assert_array_equal(be.get_pores_ids(), [])
        np.testing.assert_array_equal(be.get_throats_ids(), [])

    def test_set_pores_ids(self):
        be, _, _ = self._make()
        be.set_pores_ids(np.array([3, 5, 7], np.int64))
        np.testing.assert_array_equal(be.get_pores_ids(), [3, 5, 7])
        self.assertEqual(be.pore_count, 3)

    def test_set_throats_ids(self):
        be, _, _ = self._make()
        be.set_throats_ids(np.array([1, 4], np.int64))
        np.testing.assert_array_equal(be.get_throats_ids(), [1, 4])
        self.assertEqual(be.throat_count, 2)

    def test_add_pores_ids(self):
        be, _, _ = self._make()
        be.set_pores_ids(np.array([1], np.int64))
        be.add_pores_ids(np.array([3, 5], np.int64))
        np.testing.assert_array_equal(be.get_pores_ids(), [1, 3, 5])

    def test_add_throats_ids(self):
        be, _, _ = self._make()
        be.add_throats_ids(np.array([2, 4], np.int64))
        np.testing.assert_array_equal(be.get_throats_ids(), [2, 4])

    def test_remove_pores_ids(self):
        be, _, _ = self._make()
        be.set_pores_ids(np.array([1, 3, 5, 7], np.int64))
        be.remove_pores_ids(np.array([3, 7], np.int64))
        np.testing.assert_array_equal(be.get_pores_ids(), [1, 5])

    def test_remove_throats_ids(self):
        be, _, _ = self._make()
        be.set_throats_ids(np.array([0, 2, 4, 6], np.int64))
        be.remove_throats_ids(np.array([0, 4], np.int64))
        np.testing.assert_array_equal(be.get_throats_ids(), [2, 6])

    def test_set_replaces_previous(self):
        be, _, _ = self._make()
        be.set_pores_ids(np.array([1, 2, 3], np.int64))
        be.set_pores_ids(np.array([7, 8], np.int64))
        np.testing.assert_array_equal(be.get_pores_ids(), [7, 8])
        self.assertEqual(be.pore_count, 2)

    def test_contains(self):
        be, _, _ = self._make()
        be.set_pores_ids(np.array([3], np.int64))
        be.set_throats_ids(np.array([5], np.int64))
        self.assertTrue(be.contains_pore(3))
        self.assertFalse(be.contains_pore(5))
        self.assertTrue(be.contains_throat(5))
        self.assertFalse(be.contains_throat(3))


class TestBorderElementsMinMax(unittest.TestCase):
    """Min/max element extraction."""

    def test_get_min_element_pores_only(self):
        pore_pc = np.array([500.0, 100.0, 300.0])
        throat_pc = np.array([999.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1, 2], np.int64))
        result = be.get_min_element()
        self.assertEqual(result, (PORE, 1, 100.0))

    def test_get_min_element_throats_only(self):
        pore_pc = np.array([999.0])
        throat_pc = np.array([400.0, 200.0, 600.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_throats_ids(np.array([0, 1, 2], np.int64))
        result = be.get_min_element()
        self.assertEqual(result, (THROAT, 1, 200.0))

    def test_get_min_element_mixed(self):
        pore_pc = np.array([300.0, 500.0])
        throat_pc = np.array([100.0, 400.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1], np.int64))
        be.set_throats_ids(np.array([0, 1], np.int64))
        result = be.get_min_element()
        self.assertEqual(result, (THROAT, 0, 100.0))

    def test_get_min_element_empty(self):
        be = BorderElements(np.array([1.0]), np.array([1.0]))
        self.assertIsNone(be.get_min_element())

    def test_get_max_element_pores_only(self):
        pore_pc = np.array([500.0, 100.0, 300.0])
        throat_pc = np.array([0.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1, 2], np.int64))
        result = be.get_max_element()
        self.assertEqual(result, (PORE, 0, 500.0))

    def test_get_max_element_mixed(self):
        pore_pc = np.array([300.0, 500.0])
        throat_pc = np.array([100.0, 700.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1], np.int64))
        be.set_throats_ids(np.array([0, 1], np.int64))
        result = be.get_max_element()
        self.assertEqual(result, (THROAT, 1, 700.0))

    def test_get_max_element_empty(self):
        be = BorderElements(np.array([1.0]), np.array([1.0]))
        self.assertIsNone(be.get_max_element())


class TestBorderElementsPcSync(unittest.TestCase):
    """Sync tree PCs after external entry_pressure array modifications."""

    def test_sync_pore_pcs(self):
        pore_pc = np.array([500.0, 100.0, 300.0])
        throat_pc = np.array([999.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1, 2], np.int64))

        # Externally modify entry_pressure array
        pore_pc[1] = 600.0

        # Before sync, tree still has old value
        result_before = be.get_min_element()
        self.assertEqual(result_before, (PORE, 1, 100.0))

        # After sync, tree reflects new value
        be.sync_pore_pcs()
        result_after = be.get_min_element()
        self.assertEqual(result_after, (PORE, 2, 300.0))

    def test_sync_throat_pcs_specific_ids(self):
        pore_pc = np.array([999.0])
        throat_pc = np.array([200.0, 400.0, 100.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_throats_ids(np.array([0, 1, 2], np.int64))

        throat_pc[2] = 500.0
        be.sync_throat_pcs(np.array([2], np.int64))

        result = be.get_max_element()
        self.assertEqual(result, (THROAT, 2, 500.0))

    def test_sync_after_add(self):
        """Sync works correctly for elements added after initial set."""
        pore_pc = np.array([100.0, 200.0, 300.0, 50.0])
        throat_pc = np.array([999.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1], np.int64))
        be.add_pores_ids(np.array([3], np.int64))

        self.assertEqual(be.get_min_element(), (PORE, 3, 50.0))

        pore_pc[3] = 999.0
        be.sync_pore_pcs(np.array([3], np.int64))
        self.assertEqual(be.get_min_element(), (PORE, 0, 100.0))


if __name__ == "__main__":
    unittest.main()
