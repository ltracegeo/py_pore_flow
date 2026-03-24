"""Unit tests for the InvasionHeap indexed binary heap."""

import unittest
import numpy as np
from py_pore_flow.simulation.invasion_heap import InvasionHeap
from py_pore_flow.network.constants import PORE, THROAT


class TestInvasionHeapMinHeap(unittest.TestCase):
    """Tests for min-heap behaviour (drainage / 2nd drainage)."""

    def setUp(self):
        self.heap = InvasionHeap(n_pores=10, n_throats=10, is_min_heap=True)

    def test_empty_heap(self):
        self.assertEqual(len(self.heap), 0)
        self.assertIsNone(self.heap.pop())
        self.assertIsNone(self.heap.peek())

    def test_insert_and_pop_single(self):
        self.heap.insert(PORE, 3, 100.0)
        self.assertEqual(len(self.heap), 1)
        result = self.heap.pop()
        self.assertEqual(result, (PORE, 3, 100.0))
        self.assertEqual(len(self.heap), 0)

    def test_pop_min_order(self):
        """Elements should be popped in ascending order of key."""
        self.heap.insert(PORE, 0, 500.0)
        self.heap.insert(THROAT, 2, 100.0)
        self.heap.insert(PORE, 5, 300.0)
        self.heap.insert(THROAT, 7, 200.0)

        keys = []
        while len(self.heap) > 0:
            _, _, key = self.heap.pop()
            keys.append(key)
        self.assertEqual(keys, [100.0, 200.0, 300.0, 500.0])

    def test_peek_does_not_remove(self):
        self.heap.insert(PORE, 1, 42.0)
        result = self.heap.peek()
        self.assertEqual(result, (PORE, 1, 42.0))
        self.assertEqual(len(self.heap), 1)

    def test_update_key(self):
        self.heap.insert(PORE, 0, 500.0)
        self.heap.insert(PORE, 1, 300.0)
        # Update pore 0 to be the new minimum
        self.heap.update_key(PORE, 0, 100.0)
        result = self.heap.peek()
        self.assertEqual(result, (PORE, 0, 100.0))

    def test_remove(self):
        self.heap.insert(PORE, 0, 100.0)
        self.heap.insert(PORE, 1, 200.0)
        self.heap.remove(PORE, 0)
        self.assertEqual(len(self.heap), 1)
        result = self.heap.peek()
        self.assertEqual(result, (PORE, 1, 200.0))

    def test_contains(self):
        self.heap.insert(THROAT, 5, 99.0)
        self.assertTrue(self.heap.contains(THROAT, 5))
        self.assertFalse(self.heap.contains(PORE, 5))
        self.assertFalse(self.heap.contains(THROAT, 0))

    def test_insert_duplicate_updates_key(self):
        """Inserting the same element again should update its key."""
        self.heap.insert(PORE, 3, 500.0)
        self.heap.insert(PORE, 3, 100.0)
        self.assertEqual(len(self.heap), 1)
        result = self.heap.pop()
        self.assertEqual(result, (PORE, 3, 100.0))

    def test_clear(self):
        self.heap.insert(PORE, 0, 1.0)
        self.heap.insert(THROAT, 0, 2.0)
        self.heap.clear()
        self.assertEqual(len(self.heap), 0)
        self.assertFalse(self.heap.contains(PORE, 0))

    def test_many_elements(self):
        """Insert many elements and verify they come out in sorted order."""
        rng = np.random.default_rng(42)
        keys = rng.uniform(0, 10000, size=20)
        for i, key in enumerate(keys):
            if i < 10:
                self.heap.insert(PORE, i, key)
            else:
                self.heap.insert(THROAT, i - 10, key)

        popped_keys = []
        while len(self.heap) > 0:
            _, _, key = self.heap.pop()
            popped_keys.append(key)
        # Verify sorted ascending
        for i in range(len(popped_keys) - 1):
            self.assertLessEqual(popped_keys[i], popped_keys[i + 1])


class TestInvasionHeapMaxHeap(unittest.TestCase):
    """Tests for max-heap behaviour (imbibition)."""

    def setUp(self):
        self.heap = InvasionHeap(n_pores=10, n_throats=10, is_min_heap=False)

    def test_pop_max_order(self):
        """Elements should be popped in descending order of key."""
        self.heap.insert(PORE, 0, 500.0)
        self.heap.insert(THROAT, 2, 100.0)
        self.heap.insert(PORE, 5, 300.0)
        self.heap.insert(THROAT, 7, 200.0)

        keys = []
        while len(self.heap) > 0:
            _, _, key = self.heap.pop()
            keys.append(key)
        self.assertEqual(keys, [500.0, 300.0, 200.0, 100.0])

    def test_update_key_max(self):
        self.heap.insert(PORE, 0, 100.0)
        self.heap.insert(PORE, 1, 300.0)
        # Update pore 0 to be the new maximum
        self.heap.update_key(PORE, 0, 999.0)
        result = self.heap.peek()
        self.assertEqual(result, (PORE, 0, 999.0))


if __name__ == "__main__":
    unittest.main()
