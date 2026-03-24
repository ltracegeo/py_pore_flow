"""Unit tests for the BorderElements container."""

import unittest
import numpy as np
from py_pore_flow.simulation.border_elements import BorderElements


class TestBorderElements(unittest.TestCase):

    def test_empty_initial(self):
        be = BorderElements()
        self.assertEqual(len(be.get_pores_ids()), 0)
        self.assertEqual(len(be.get_throats_ids()), 0)

    def test_set_and_get(self):
        be = BorderElements()
        be.set_pores_ids(np.array([1, 3, 5], dtype=np.int64))
        np.testing.assert_array_equal(sorted(be.get_pores_ids()), [1, 3, 5])

    def test_add_and_get(self):
        be = BorderElements()
        be.add_pores_ids(np.array([2, 4], dtype=np.int64))
        be.add_pores_ids(np.array([6], dtype=np.int64))
        self.assertEqual(len(be.get_pores_ids()), 3)
        self.assertTrue(set(be.get_pores_ids().tolist()) == {2, 4, 6})

    def test_remove_with_in_border(self):
        """Test remove using in_border boolean array (optimized path)."""
        in_border = np.array([True, False, True, False, True], dtype=bool)
        be = BorderElements(pore_in_border=in_border)
        be.set_pores_ids(np.array([0, 1, 2, 3, 4], dtype=np.int64))
        # in_border marks 0, 2, 4 as True — removing anything filters by this
        be.remove_pores_ids(np.array([1, 3], dtype=np.int64))
        result = set(be.get_pores_ids().tolist())
        self.assertEqual(result, {0, 2, 4})

    def test_remove_fallback(self):
        """Test remove via np.isin fallback (no in_border arrays)."""
        be = BorderElements()
        be.set_pores_ids(np.array([0, 1, 2, 3, 4], dtype=np.int64))
        be.remove_pores_ids(np.array([1, 3], dtype=np.int64))
        result = set(be.get_pores_ids().tolist())
        self.assertEqual(result, {0, 2, 4})

    def test_remove_nonexistent(self):
        be = BorderElements()
        be.set_pores_ids(np.array([1, 2], dtype=np.int64))
        be.remove_pores_ids(np.array([99], dtype=np.int64))
        self.assertEqual(len(be.get_pores_ids()), 2)

    def test_throats_add_remove(self):
        be = BorderElements()
        be.add_throats_ids(np.array([0, 5, 9], dtype=np.int64))
        be.remove_throats_ids(np.array([5], dtype=np.int64))
        result = set(be.get_throats_ids().tolist())
        self.assertEqual(result, {0, 9})

    def test_set_replaces(self):
        be = BorderElements()
        be.set_pores_ids(np.array([1, 2, 3], dtype=np.int64))
        be.set_pores_ids(np.array([7, 8], dtype=np.int64))
        result = set(be.get_pores_ids().tolist())
        self.assertEqual(result, {7, 8})

    def test_remove_empty(self):
        """Removing empty array should be a no-op."""
        be = BorderElements()
        be.set_pores_ids(np.array([1, 2], dtype=np.int64))
        be.remove_pores_ids(np.array([], dtype=np.int64))
        self.assertEqual(len(be.get_pores_ids()), 2)


if __name__ == "__main__":
    unittest.main()
