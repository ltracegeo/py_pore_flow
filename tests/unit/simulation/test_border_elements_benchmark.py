"""Benchmark and correctness tests comparing old (numpy-array) vs new (tree-backed)
BorderElements implementations, plus the operations they replaced.

Runs each operation at realistic network sizes (100, 1000, 10000 elements) and
reports timings so we can identify which operations are faster or slower.
"""

import time
import unittest
import numpy as np

from py_pore_flow.network.constants import PORE, THROAT
from py_pore_flow.simulation.border_elements import BorderElements
from py_pore_flow.simulation.indexed_priority_tree import IndexedPriorityTree


# ============================================================================
# Old (numpy-array) BorderElements implementation — recovered for comparison
# ============================================================================

class OldBorderElements:
    """Original implementation: plain numpy arrays with concatenate/delete."""

    def __init__(self):
        self.throats_ids = np.array([], np.int64)
        self.pores_ids = np.array([], np.int64)

    def set_throats_ids(self, throats_ids):
        self.throats_ids = throats_ids

    def set_pores_ids(self, pores_ids):
        self.pores_ids = pores_ids

    def get_throats_ids(self):
        return self.throats_ids

    def get_pores_ids(self):
        return self.pores_ids

    def add_throats_ids(self, throats_ids):
        self.throats_ids = np.concatenate((self.throats_ids, throats_ids))

    def add_pores_ids(self, pores_ids):
        self.pores_ids = np.concatenate((self.pores_ids, pores_ids))

    def remove_throats_ids(self, throat_ids):
        self.throats_ids = np.delete(self.throats_ids, np.isin(self.throats_ids, throat_ids))

    def remove_pores_ids(self, pore_ids):
        self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, pore_ids))


# ============================================================================
# Old _get_element_to_fill: linear scan (the njit versions from drainage/imbibition)
# ============================================================================

def old_get_min_element(pore_ids, pore_entry_pressure, throat_ids, throat_entry_pressure):
    """Old drainage: linear scan for minimum entry pressure."""
    min_pressure = np.inf
    min_id = -1
    min_type = -1
    for pid in pore_ids:
        p = pore_entry_pressure[pid]
        if p < min_pressure:
            min_pressure = p
            min_id = pid
            min_type = PORE
    for tid in throat_ids:
        p = throat_entry_pressure[tid]
        if p < min_pressure:
            min_pressure = p
            min_id = tid
            min_type = THROAT
    if min_id == -1:
        return None
    return (min_type, min_id, min_pressure)


def old_get_max_element(pore_ids, pore_entry_pressure, throat_ids, throat_entry_pressure):
    """Old imbibition: linear scan for maximum entry pressure."""
    max_pressure = -np.inf
    max_id = -1
    max_type = -1
    for pid in pore_ids:
        p = pore_entry_pressure[pid]
        if p > max_pressure:
            max_pressure = p
            max_id = pid
            max_type = PORE
    for tid in throat_ids:
        p = throat_entry_pressure[tid]
        if p > max_pressure:
            max_pressure = p
            max_id = tid
            max_type = THROAT
    if max_id == -1:
        return None
    return (max_type, max_id, max_pressure)


# ============================================================================
# Helpers
# ============================================================================

def _time_fn(fn, repeat=50):
    """Time a function call, return median time in microseconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e6  # microseconds


# ============================================================================
# Correctness tests: tree-backed vs old implementation
# ============================================================================

class TestCorrectnessVsOld(unittest.TestCase):
    """Verify that the tree-backed BorderElements gives the same results as the
    old numpy-array implementation for all operations."""

    def test_set_get_pores(self):
        """set_pores_ids + get_pores_ids returns same sorted IDs."""
        rng = np.random.RandomState(42)
        pore_pc = rng.uniform(0, 10000, 500).astype(np.float64)
        throat_pc = rng.uniform(0, 10000, 300).astype(np.float64)

        ids = np.array([10, 50, 200, 3, 499], np.int64)

        old = OldBorderElements()
        old.set_pores_ids(ids.copy())

        new = BorderElements(pore_pc, throat_pc)
        new.set_pores_ids(ids.copy())

        np.testing.assert_array_equal(
            np.sort(old.get_pores_ids()),
            np.sort(new.get_pores_ids()),
        )

    def test_add_remove_sequence(self):
        """After identical add/remove sequences, both have the same IDs."""
        rng = np.random.RandomState(123)
        pore_pc = rng.uniform(0, 10000, 1000).astype(np.float64)
        throat_pc = rng.uniform(0, 10000, 500).astype(np.float64)

        old = OldBorderElements()
        new = BorderElements(pore_pc, throat_pc)

        # Add
        add_ids = np.array([5, 10, 20, 50, 100, 200], np.int64)
        old.add_pores_ids(add_ids)
        new.add_pores_ids(add_ids)

        # Remove some
        rm_ids = np.array([10, 100], np.int64)
        old.remove_pores_ids(rm_ids)
        new.remove_pores_ids(rm_ids)

        np.testing.assert_array_equal(
            np.sort(old.get_pores_ids()),
            np.sort(new.get_pores_ids()),
        )

    def test_get_min_element_matches_old_scan(self):
        """Tree peek_min matches old linear-scan minimum."""
        rng = np.random.RandomState(99)
        n_pores, n_throats = 500, 300
        pore_pc = rng.uniform(100, 10000, n_pores).astype(np.float64)
        throat_pc = rng.uniform(100, 10000, n_throats).astype(np.float64)

        pore_ids = rng.choice(n_pores, 100, replace=False).astype(np.int64)
        throat_ids = rng.choice(n_throats, 80, replace=False).astype(np.int64)

        new = BorderElements(pore_pc, throat_pc)
        new.set_pores_ids(pore_ids)
        new.set_throats_ids(throat_ids)

        tree_result = new.get_min_element()
        old_result = old_get_min_element(pore_ids, pore_pc, throat_ids, throat_pc)

        self.assertIsNotNone(tree_result)
        self.assertIsNotNone(old_result)
        # Both should find the same minimum PC value
        self.assertAlmostEqual(tree_result[2], old_result[2], places=10)

    def test_get_max_element_matches_old_scan(self):
        """Tree peek_max matches old linear-scan maximum."""
        rng = np.random.RandomState(77)
        n_pores, n_throats = 500, 300
        pore_pc = rng.uniform(100, 10000, n_pores).astype(np.float64)
        throat_pc = rng.uniform(100, 10000, n_throats).astype(np.float64)

        pore_ids = rng.choice(n_pores, 100, replace=False).astype(np.int64)
        throat_ids = rng.choice(n_throats, 80, replace=False).astype(np.int64)

        new = BorderElements(pore_pc, throat_pc)
        new.set_pores_ids(pore_ids)
        new.set_throats_ids(throat_ids)

        tree_result = new.get_max_element()
        old_result = old_get_max_element(pore_ids, pore_pc, throat_ids, throat_pc)

        self.assertIsNotNone(tree_result)
        self.assertIsNotNone(old_result)
        self.assertAlmostEqual(tree_result[2], old_result[2], places=10)

    def test_sync_pcs_updates_min(self):
        """After external PC modification + sync, tree reports correct min."""
        pore_pc = np.array([500.0, 100.0, 300.0, 700.0, 200.0])
        throat_pc = np.array([999.0])
        be = BorderElements(pore_pc, throat_pc)
        be.set_pores_ids(np.array([0, 1, 2, 3, 4], np.int64))

        # Externally change the minimum
        pore_pc[1] = 900.0
        be.sync_pore_pcs()

        result = be.get_min_element()
        # New min should be pore 4 (200.0)
        self.assertEqual(result, (PORE, 4, 200.0))


# ============================================================================
# Benchmark tests
# ============================================================================

class TestBenchmarkOperations(unittest.TestCase):
    """Compare timing of each operation between old and new implementations.

    These tests always pass — they just print timing information.
    """

    def _run_benchmark(self, label, sizes=(100, 500, 2000, 10000)):
        """Run a benchmark across multiple sizes and print results."""
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {label}")
        print(f"{'='*70}")
        print(f"  {'Size':>8} {'Old (µs)':>12} {'New (µs)':>12} {'Ratio':>8} {'Winner':>8}")
        print(f"  {'-'*56}")
        return sizes

    def test_bench_get_all_ids(self):
        """get_pores_ids: old returns stored array, new does tree traversal."""
        sizes = self._run_benchmark("get_pores_ids (returning ID array)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            ids = rng.choice(n, n // 2, replace=False).astype(np.int64)

            old = OldBorderElements()
            old.set_pores_ids(ids.copy())

            new = BorderElements(pore_pc, throat_pc)
            new.set_pores_ids(ids.copy())

            t_old = _time_fn(lambda: old.get_pores_ids())
            t_new = _time_fn(lambda: new.get_pores_ids())
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_set_ids(self):
        """set_pores_ids: old stores array, new does bulk tree insert."""
        sizes = self._run_benchmark("set_pores_ids (bulk replace)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            ids = rng.choice(n, n // 2, replace=False).astype(np.int64)

            old = OldBorderElements()
            new = BorderElements(pore_pc, throat_pc)

            t_old = _time_fn(lambda: old.set_pores_ids(ids.copy()))
            t_new = _time_fn(lambda: new.set_pores_ids(ids.copy()))
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_add_ids(self):
        """add_pores_ids: old uses np.concatenate, new loops tree.insert."""
        sizes = self._run_benchmark("add_pores_ids (incremental add)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            # Add 5 elements (typical per invasion step)
            add_ids = rng.choice(n, min(5, n), replace=False).astype(np.int64)

            def bench_old():
                o = OldBorderElements()
                o.set_pores_ids(np.arange(n // 4, dtype=np.int64))
                o.add_pores_ids(add_ids)

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(np.arange(n // 4, dtype=np.int64))
                be.add_pores_ids(add_ids)

            # Warm up
            bench_old(); bench_new()

            t_old = _time_fn(bench_old, repeat=30)
            t_new = _time_fn(bench_new, repeat=30)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_remove_ids(self):
        """remove_pores_ids: old uses np.delete+isin, new loops tree.remove."""
        sizes = self._run_benchmark("remove_pores_ids (incremental remove)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            base_ids = np.arange(n // 2, dtype=np.int64)
            rm_ids = rng.choice(base_ids, min(5, len(base_ids)), replace=False).astype(np.int64)

            def bench_old():
                o = OldBorderElements()
                o.set_pores_ids(base_ids.copy())
                o.remove_pores_ids(rm_ids)

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(base_ids.copy())
                be.remove_pores_ids(rm_ids)

            bench_old(); bench_new()

            t_old = _time_fn(bench_old, repeat=30)
            t_new = _time_fn(bench_new, repeat=30)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_get_min_element(self):
        """get_min_element: old linear scan vs new tree peek."""
        sizes = self._run_benchmark("get_min_element (find best candidate)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)

            new = BorderElements(pore_pc, throat_pc)
            new.set_pores_ids(pore_ids)
            new.set_throats_ids(throat_ids)

            t_old = _time_fn(lambda: old_get_min_element(pore_ids, pore_pc, throat_ids, throat_pc))
            t_new = _time_fn(lambda: new.get_min_element())
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_get_max_element(self):
        """get_max_element: old linear scan vs new tree peek."""
        sizes = self._run_benchmark("get_max_element (find best candidate)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)

            new = BorderElements(pore_pc, throat_pc)
            new.set_pores_ids(pore_ids)
            new.set_throats_ids(throat_ids)

            t_old = _time_fn(lambda: old_get_max_element(pore_ids, pore_pc, throat_ids, throat_pc))
            t_new = _time_fn(lambda: new.get_max_element())
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_sync_pcs(self):
        """sync_pore_pcs: Python loop calling tree.update_pc per element."""
        sizes = self._run_benchmark("sync_pore_pcs (update all border PCs)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            ids = rng.choice(n, n // 3, replace=False).astype(np.int64)

            be = BorderElements(pore_pc, throat_pc)
            be.set_pores_ids(ids)

            # Modify some PCs
            pore_pc[ids] = rng.uniform(0, 10000, len(ids))

            # Old approach: no sync needed (it just re-reads the array on access)
            # So old cost is 0 for sync. The cost shows up in get_min_element instead.
            # But we need to measure the sync itself since it's new overhead.
            t_new = _time_fn(lambda: be.sync_pore_pcs())
            print(f"  {n:>8} {'N/A':>12} {t_new:>12.1f} {'(new overhead)':>20}")

    def test_bench_full_invasion_step(self):
        """Simulate one invasion step: get_element + remove_filled + add_new + sync.

        This measures the total overhead per invasion step, which is the metric
        that matters for end-to-end performance.
        """
        sizes = self._run_benchmark("Full invasion step (get + remove + add + sync)")
        for n in sizes:
            rng = np.random.RandomState(42)
            n_border = n // 3
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n_border, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n_border, replace=False).astype(np.int64)

            # New border elements to add (typical: 1-5 per step)
            add_pores = rng.choice(
                np.setdiff1d(np.arange(n), pore_ids), min(3, n - n_border), replace=False
            ).astype(np.int64)
            # Elements to remove (filled element + trapped)
            rm_pores = pore_ids[:min(2, n_border)].copy()

            def bench_old():
                old = OldBorderElements()
                old.set_pores_ids(pore_ids.copy())
                old.set_throats_ids(throat_ids.copy())
                # Find best element
                old_get_min_element(
                    old.get_pores_ids(), pore_pc,
                    old.get_throats_ids(), throat_pc
                )
                # Remove filled + add new
                old.remove_pores_ids(rm_pores)
                old.add_pores_ids(add_pores)
                # No sync needed in old implementation

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(pore_ids.copy())
                be.set_throats_ids(throat_ids.copy())
                # Find best element
                be.get_min_element()
                # Remove filled + add new
                be.remove_pores_ids(rm_pores)
                be.add_pores_ids(add_pores)
                # Sync (simulating entry pressure update for new elements)
                be.sync_pore_pcs(add_pores)

            bench_old(); bench_new()

            t_old = _time_fn(bench_old, repeat=30)
            t_new = _time_fn(bench_new, repeat=30)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_repeated_invasion_loop(self):
        """Simulate a realistic invasion loop: repeated get_min + remove + add cycles.

        This is the most realistic benchmark — it shows cumulative performance
        over many invasion steps without rebuilding the tree from scratch.
        """
        sizes = self._run_benchmark("Invasion loop (100 steps, no rebuild)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            n_border = n // 3
            initial_pore_ids = rng.choice(n, n_border, replace=False).astype(np.int64)
            initial_throat_ids = rng.choice(n, n_border, replace=False).astype(np.int64)

            n_steps = 100

            def bench_old():
                old = OldBorderElements()
                old.set_pores_ids(initial_pore_ids.copy())
                old.set_throats_ids(initial_throat_ids.copy())
                for _ in range(n_steps):
                    pids = old.get_pores_ids()
                    tids = old.get_throats_ids()
                    if len(pids) == 0 and len(tids) == 0:
                        break
                    old_get_min_element(pids, pore_pc, tids, throat_pc)
                    # Remove 1 pore
                    if len(pids) > 0:
                        old.remove_pores_ids(pids[:1])
                    # Add 1 new pore
                    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), old.get_pores_ids())
                    if len(remaining) > 0:
                        old.add_pores_ids(remaining[:1])

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(initial_pore_ids.copy())
                be.set_throats_ids(initial_throat_ids.copy())
                for _ in range(n_steps):
                    if be.is_empty:
                        break
                    be.get_min_element()
                    pids = be.get_pores_ids()
                    if len(pids) > 0:
                        be.remove_pores_ids(pids[:1])
                    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), be.get_pores_ids())
                    if len(remaining) > 0:
                        be.add_pores_ids(remaining[:1])

            bench_old(); bench_new()

            t_old = _time_fn(bench_old, repeat=10)
            t_new = _time_fn(bench_new, repeat=10)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")


    def test_bench_imbibition_pc_step(self):
        """Simulate a realistic imbibition Pc step: sync all PCs, then 20 invasions.

        This is the most critical benchmark — it simulates the exact hot path
        that was 5x slower: entry pressure update (full sync) + invasion loop.
        """
        sizes = self._run_benchmark("Imbibition Pc step (full sync + 20 invasions)")
        for n in sizes:
            rng = np.random.RandomState(42)
            n_border = n // 3
            pore_pc = rng.uniform(-10000, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(-10000, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n_border, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n_border, replace=False).astype(np.int64)

            n_invasions = 20

            def bench_old():
                old = OldBorderElements()
                old.set_pores_ids(pore_ids.copy())
                old.set_throats_ids(throat_ids.copy())
                # Simulate _update_entry_pressures: modify PCs, then scan all border elements
                pore_pc[:] = rng.uniform(-10000, 10000, n)
                throat_pc[:] = rng.uniform(-10000, 10000, n)
                # Old: no sync needed, PCs read directly from arrays during scan
                for _ in range(n_invasions):
                    pids = old.get_pores_ids()
                    tids = old.get_throats_ids()
                    if len(pids) == 0 and len(tids) == 0:
                        break
                    # Imbibition: find max
                    old_get_max_element(pids, pore_pc, tids, throat_pc)
                    if len(pids) > 0:
                        old.remove_pores_ids(pids[:1])
                    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), old.get_pores_ids())
                    if len(remaining) > 0:
                        old.add_pores_ids(remaining[:1])

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(pore_ids.copy())
                be.set_throats_ids(throat_ids.copy())
                # Simulate _update_entry_pressures: modify PCs, then full sync
                pore_pc[:] = rng.uniform(-10000, 10000, n)
                throat_pc[:] = rng.uniform(-10000, 10000, n)
                be.sync_pore_pcs()  # Full rebuild
                be.sync_throat_pcs()  # Full rebuild
                for _ in range(n_invasions):
                    if be.is_empty:
                        break
                    be.get_max_element()
                    pids = be.get_pores_ids()
                    if len(pids) > 0:
                        be.remove_pores_ids(pids[:1])
                    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), be.get_pores_ids())
                    if len(remaining) > 0:
                        be.add_pores_ids(remaining[:1])

            bench_old(); bench_new()

            t_old = _time_fn(bench_old, repeat=10)
            t_new = _time_fn(bench_new, repeat=10)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
