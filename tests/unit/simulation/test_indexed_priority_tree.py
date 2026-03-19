"""Tests for IndexedPriorityTree — dual Red-Black tree indexed priority queue."""
import unittest

import numpy as np

from py_pore_flow.simulation.indexed_priority_tree import IndexedPriorityTree


class TestBasicOperations(unittest.TestCase):
    """Insert, contains, get_pc, remove."""

    def test_insert_and_contains(self):
        tree = IndexedPriorityTree(100)
        self.assertFalse(tree.contains(5))
        tree.insert(5, 100.0)
        self.assertTrue(tree.contains(5))
        self.assertEqual(tree.count, 1)

    def test_insert_returns_false_on_duplicate(self):
        tree = IndexedPriorityTree(100)
        self.assertTrue(tree.insert(5, 100.0))
        self.assertFalse(tree.insert(5, 200.0))
        self.assertEqual(tree.count, 1)
        self.assertEqual(tree.get_pc(5), 100.0)

    def test_get_pc(self):
        tree = IndexedPriorityTree(100)
        tree.insert(10, 500.0)
        tree.insert(20, 300.0)
        self.assertEqual(tree.get_pc(10), 500.0)
        self.assertEqual(tree.get_pc(20), 300.0)

    def test_get_pc_raises_on_missing(self):
        tree = IndexedPriorityTree(100)
        with self.assertRaises(KeyError):
            tree.get_pc(42)

    def test_remove(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        tree.insert(10, 200.0)
        self.assertTrue(tree.remove(5))
        self.assertFalse(tree.contains(5))
        self.assertTrue(tree.contains(10))
        self.assertEqual(tree.count, 1)

    def test_remove_returns_false_on_missing(self):
        tree = IndexedPriorityTree(100)
        self.assertFalse(tree.remove(42))

    def test_len_and_in(self):
        tree = IndexedPriorityTree(100)
        self.assertEqual(len(tree), 0)
        tree.insert(5, 100.0)
        self.assertEqual(len(tree), 1)
        self.assertIn(5, tree)
        self.assertNotIn(99, tree)


class TestUpdatePC(unittest.TestCase):
    """Capillary-pressure update."""

    def test_update_pc_changes_value(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        tree.update_pc(5, 50.0)
        self.assertEqual(tree.get_pc(5), 50.0)

    def test_update_pc_changes_min(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        tree.insert(10, 200.0)
        self.assertEqual(tree.peek_min(), (5, 100.0))
        tree.update_pc(10, 50.0)
        self.assertEqual(tree.peek_min(), (10, 50.0))

    def test_update_pc_noop_when_same(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        tree.update_pc(5, 100.0)  # same value, should be no-op
        self.assertEqual(tree.get_pc(5), 100.0)
        self.assertTrue(tree.validate())

    def test_update_pc_on_missing_is_noop(self):
        tree = IndexedPriorityTree(100)
        tree.update_pc(42, 999.0)  # should not crash
        self.assertEqual(tree.count, 0)


class TestPopAndPeekMin(unittest.TestCase):
    """Min-extraction operations."""

    def test_pop_min_single(self):
        tree = IndexedPriorityTree(100)
        tree.insert(7, 42.0)
        idx, pc = tree.pop_min()
        self.assertEqual(idx, 7)
        self.assertEqual(pc, 42.0)
        self.assertTrue(tree.is_empty)

    def test_pop_min_returns_lowest(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 300.0)
        tree.insert(10, 100.0)
        tree.insert(15, 200.0)
        idx, pc = tree.pop_min()
        self.assertEqual(idx, 10)
        self.assertEqual(pc, 100.0)
        self.assertEqual(tree.count, 2)
        self.assertFalse(tree.contains(10))

    def test_pop_min_full_order(self):
        tree = IndexedPriorityTree(100)
        pcs = [50.0, 10.0, 80.0, 30.0, 60.0]
        for i, pc in enumerate(pcs):
            tree.insert(i, pc)
        result = []
        while not tree.is_empty:
            _, pc = tree.pop_min()
            result.append(pc)
        self.assertEqual(result, sorted(pcs))

    def test_pop_min_empty_raises(self):
        tree = IndexedPriorityTree(100)
        with self.assertRaises(IndexError):
            tree.pop_min()

    def test_peek_min_does_not_remove(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        idx, pc = tree.peek_min()
        self.assertEqual((idx, pc), (5, 100.0))
        self.assertEqual(tree.count, 1)

    def test_peek_min_empty_raises(self):
        tree = IndexedPriorityTree(100)
        with self.assertRaises(IndexError):
            tree.peek_min()

    def test_same_pc_pop_order(self):
        """Elements with equal PC are ordered by ascending element index."""
        tree = IndexedPriorityTree(100)
        tree.insert(3, 100.0)
        tree.insert(1, 100.0)
        tree.insert(2, 100.0)
        self.assertEqual(tree.pop_min(), (1, 100.0))
        self.assertEqual(tree.pop_min(), (2, 100.0))
        self.assertEqual(tree.pop_min(), (3, 100.0))


class TestPopAndPeekMax(unittest.TestCase):
    """Max-extraction operations."""

    def test_pop_max_single(self):
        tree = IndexedPriorityTree(100)
        tree.insert(7, 42.0)
        idx, pc = tree.pop_max()
        self.assertEqual(idx, 7)
        self.assertEqual(pc, 42.0)
        self.assertTrue(tree.is_empty)

    def test_pop_max_returns_highest(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 300.0)
        tree.insert(10, 100.0)
        tree.insert(15, 200.0)
        idx, pc = tree.pop_max()
        self.assertEqual(idx, 5)
        self.assertEqual(pc, 300.0)
        self.assertEqual(tree.count, 2)
        self.assertFalse(tree.contains(5))

    def test_pop_max_full_order(self):
        tree = IndexedPriorityTree(100)
        pcs = [50.0, 10.0, 80.0, 30.0, 60.0]
        for i, pc in enumerate(pcs):
            tree.insert(i, pc)
        result = []
        while not tree.is_empty:
            _, pc = tree.pop_max()
            result.append(pc)
        self.assertEqual(result, sorted(pcs, reverse=True))

    def test_pop_max_empty_raises(self):
        tree = IndexedPriorityTree(100)
        with self.assertRaises(IndexError):
            tree.pop_max()

    def test_peek_max_does_not_remove(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        idx, pc = tree.peek_max()
        self.assertEqual((idx, pc), (5, 100.0))
        self.assertEqual(tree.count, 1)

    def test_peek_max_empty_raises(self):
        tree = IndexedPriorityTree(100)
        with self.assertRaises(IndexError):
            tree.peek_max()

    def test_same_pc_pop_max_order(self):
        """Elements with equal PC are popped in descending element index order."""
        tree = IndexedPriorityTree(100)
        tree.insert(3, 100.0)
        tree.insert(1, 100.0)
        tree.insert(2, 100.0)
        self.assertEqual(tree.pop_max(), (3, 100.0))
        self.assertEqual(tree.pop_max(), (2, 100.0))
        self.assertEqual(tree.pop_max(), (1, 100.0))

    def test_peek_max_after_update(self):
        tree = IndexedPriorityTree(100)
        tree.insert(5, 100.0)
        tree.insert(10, 200.0)
        self.assertEqual(tree.peek_max(), (10, 200.0))
        tree.update_pc(5, 500.0)
        self.assertEqual(tree.peek_max(), (5, 500.0))

    def test_rb_valid_after_pop_max(self):
        tree = IndexedPriorityTree(200)
        for i in range(100):
            tree.insert(i, float(i * 10))
        for _ in range(50):
            tree.pop_max()
        self.assertTrue(tree.validate())


class TestMerge(unittest.TestCase):
    """Merge two trees."""

    def test_merge_disjoint(self):
        a = IndexedPriorityTree(100)
        b = IndexedPriorityTree(100)
        a.insert(1, 100.0)
        a.insert(2, 200.0)
        b.insert(3, 50.0)
        b.insert(4, 150.0)
        inserted = a.merge(b)
        self.assertEqual(inserted, 2)
        self.assertEqual(a.count, 4)
        self.assertTrue(a.contains(3))
        self.assertEqual(a.peek_min(), (3, 50.0))

    def test_merge_with_duplicates(self):
        a = IndexedPriorityTree(100)
        b = IndexedPriorityTree(100)
        a.insert(1, 100.0)
        a.insert(2, 200.0)
        b.insert(2, 50.0)
        b.insert(3, 150.0)
        inserted = a.merge(b)
        self.assertEqual(inserted, 1)
        self.assertEqual(a.count, 3)
        self.assertEqual(a.get_pc(2), 200.0)  # original unchanged

    def test_merge_empty_source(self):
        a = IndexedPriorityTree(100)
        b = IndexedPriorityTree(100)
        a.insert(1, 100.0)
        self.assertEqual(a.merge(b), 0)
        self.assertEqual(a.count, 1)

    def test_merge_into_empty_target(self):
        a = IndexedPriorityTree(100)
        b = IndexedPriorityTree(100)
        b.insert(1, 100.0)
        b.insert(2, 200.0)
        self.assertEqual(a.merge(b), 2)
        self.assertEqual(a.count, 2)
        self.assertTrue(a.validate())


class TestBulkAndQuery(unittest.TestCase):
    """set_all, get_all, get_all_ids, get_all_sorted_by_pc, clear."""

    def test_get_all_sorted_by_index(self):
        tree = IndexedPriorityTree(100)
        tree.insert(20, 300.0)
        tree.insert(5, 100.0)
        tree.insert(10, 200.0)
        idx, pcs = tree.get_all()
        np.testing.assert_array_equal(idx, [5, 10, 20])
        np.testing.assert_array_equal(pcs, [100.0, 200.0, 300.0])

    def test_get_all_ids(self):
        tree = IndexedPriorityTree(100)
        tree.insert(3, 100.0)
        tree.insert(1, 200.0)
        tree.insert(2, 300.0)
        np.testing.assert_array_equal(tree.get_all_ids(), [1, 2, 3])

    def test_get_all_sorted_by_pc(self):
        tree = IndexedPriorityTree(100)
        tree.insert(1, 300.0)
        tree.insert(2, 100.0)
        tree.insert(3, 200.0)
        idx, pcs = tree.get_all_sorted_by_pc()
        np.testing.assert_array_equal(pcs, [100.0, 200.0, 300.0])
        np.testing.assert_array_equal(idx, [2, 3, 1])

    def test_get_all_empty(self):
        tree = IndexedPriorityTree(100)
        idx, pcs = tree.get_all()
        self.assertEqual(len(idx), 0)
        self.assertEqual(len(pcs), 0)

    def test_set_all(self):
        tree = IndexedPriorityTree(100)
        tree.insert(99, 1.0)  # will be cleared
        tree.set_all(np.array([5, 10, 15]), np.array([500.0, 100.0, 300.0]))
        self.assertEqual(tree.count, 3)
        self.assertFalse(tree.contains(99))
        self.assertEqual(tree.peek_min(), (10, 100.0))

    def test_clear(self):
        tree = IndexedPriorityTree(100)
        tree.insert(1, 100.0)
        tree.insert(2, 200.0)
        tree.clear()
        self.assertEqual(tree.count, 0)
        self.assertTrue(tree.is_empty)
        self.assertFalse(tree.contains(1))

    def test_clear_and_reinsert(self):
        tree = IndexedPriorityTree(100)
        tree.insert(1, 100.0)
        tree.clear()
        tree.insert(1, 200.0)
        self.assertEqual(tree.get_pc(1), 200.0)
        self.assertTrue(tree.validate())


class TestRBProperties(unittest.TestCase):
    """Verify Red-Black tree invariants are maintained."""

    def test_after_sequential_inserts(self):
        tree = IndexedPriorityTree(200)
        for i in range(150):
            tree.insert(i, float(i * 10))
        self.assertTrue(tree.validate())

    def test_after_reverse_inserts(self):
        tree = IndexedPriorityTree(200)
        for i in range(149, -1, -1):
            tree.insert(i, float(i))
        self.assertTrue(tree.validate())

    def test_after_removes(self):
        tree = IndexedPriorityTree(200)
        for i in range(150):
            tree.insert(i, float(i * 10))
        for i in range(0, 150, 3):
            tree.remove(i)
        self.assertTrue(tree.validate())

    def test_after_updates(self):
        tree = IndexedPriorityTree(200)
        for i in range(100):
            tree.insert(i, float(i * 10))
        for i in range(50):
            tree.update_pc(i, float(1000 - i))
        self.assertTrue(tree.validate())

    def test_after_pop_min(self):
        tree = IndexedPriorityTree(200)
        for i in range(100):
            tree.insert(i, float(i * 10))
        for _ in range(50):
            tree.pop_min()
        self.assertTrue(tree.validate())

    def test_single_element(self):
        tree = IndexedPriorityTree(10)
        tree.insert(0, 1.0)
        self.assertTrue(tree.validate())
        tree.remove(0)
        self.assertTrue(tree.validate())

    def test_two_elements(self):
        tree = IndexedPriorityTree(10)
        tree.insert(0, 1.0)
        tree.insert(1, 2.0)
        self.assertTrue(tree.validate())
        tree.remove(0)
        self.assertTrue(tree.validate())
        tree.remove(1)
        self.assertTrue(tree.validate())


class TestStressMixedOperations(unittest.TestCase):
    """Randomly interleave insert/remove/update/pop_min and verify consistency."""

    def test_random_ops(self):
        rng = np.random.RandomState(42)
        tree = IndexedPriorityTree(500)
        ref = {}  # elem_index → pc (ground-truth mirror)

        for _ in range(3000):
            op = rng.randint(0, 6)

            if op == 0 and len(ref) < 400:
                idx = rng.randint(0, 500)
                pc = rng.uniform(0, 10000)
                if idx not in ref:
                    tree.insert(idx, pc)
                    ref[idx] = pc

            elif op == 1 and ref:
                idx = rng.choice(list(ref.keys()))
                tree.remove(idx)
                del ref[idx]

            elif op == 2 and ref:
                idx = rng.choice(list(ref.keys()))
                new_pc = rng.uniform(0, 10000)
                tree.update_pc(idx, new_pc)
                ref[idx] = new_pc

            elif op == 3 and ref:
                # Pop min — must agree with reference.
                min_key = min(ref, key=lambda k: (ref[k], k))
                min_pc = ref[min_key]
                got_idx, got_pc = tree.pop_min()
                self.assertEqual(got_idx, min_key,
                                 f"pop_min index: expected {min_key}, got {got_idx}")
                self.assertAlmostEqual(got_pc, min_pc, places=10)
                del ref[min_key]

            elif op == 4 and ref:
                # Pop max — must agree with reference.
                max_key = max(ref, key=lambda k: (ref[k], k))
                max_pc = ref[max_key]
                got_idx, got_pc = tree.pop_max()
                self.assertEqual(got_idx, max_key,
                                 f"pop_max index: expected {max_key}, got {got_idx}")
                self.assertAlmostEqual(got_pc, max_pc, places=10)
                del ref[max_key]

            elif ref:
                idx = rng.choice(list(ref.keys()))
                self.assertAlmostEqual(tree.get_pc(idx), ref[idx], places=10)

        # Final consistency check.
        self.assertEqual(tree.count, len(ref))
        self.assertTrue(tree.validate(), "RB invariants violated after stress test")
        for idx, pc in ref.items():
            self.assertTrue(tree.contains(idx))
            self.assertAlmostEqual(tree.get_pc(idx), pc, places=10)


if __name__ == "__main__":
    unittest.main()
