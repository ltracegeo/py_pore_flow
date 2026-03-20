"""Tests for BorderElements plain array storage."""

import unittest

import numpy as np

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

    def test_sync_is_noop(self):
        """sync_*_pcs is a no-op for plain arrays."""
        be, _, _ = self._make()
        be.set_pores_ids(np.array([1, 2], np.int64))
        be.sync_pore_pcs()
        be.sync_throat_pcs(np.array([0], np.int64))
        # Should not raise, IDs unchanged
        np.testing.assert_array_equal(be.get_pores_ids(), [1, 2])


if __name__ == "__main__":
    unittest.main()
