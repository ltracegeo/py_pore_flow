"""Benchmark and correctness tests comparing old (numpy-array) vs current
BorderElements implementation.

Runs each operation at realistic network sizes and reports timings.
"""

import time
import unittest
import numpy as np

from py_pore_flow.network.constants import PORE, THROAT
from py_pore_flow.simulation.border_elements import BorderElements


# ============================================================================
# Old (numpy-array) BorderElements implementation — for comparison
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
# Scan functions (same as used in cycle classes)
# ============================================================================

def scan_min_element(pore_ids, pore_entry_pressure, throat_ids, throat_entry_pressure):
    """Linear scan for minimum entry pressure (drainage)."""
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


def scan_max_element(pore_ids, pore_entry_pressure, throat_ids, throat_entry_pressure):
    """Linear scan for maximum entry pressure (imbibition)."""
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
# Correctness tests
# ============================================================================

class TestCorrectnessVsOld(unittest.TestCase):
    """Verify that BorderElements gives the same results as OldBorderElements."""

    def test_set_get_pores(self):
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
        rng = np.random.RandomState(123)
        pore_pc = rng.uniform(0, 10000, 1000).astype(np.float64)
        throat_pc = rng.uniform(0, 10000, 500).astype(np.float64)

        old = OldBorderElements()
        new = BorderElements(pore_pc, throat_pc)

        add_ids = np.array([5, 10, 20, 50, 100, 200], np.int64)
        old.add_pores_ids(add_ids)
        new.add_pores_ids(add_ids)

        rm_ids = np.array([10, 100], np.int64)
        old.remove_pores_ids(rm_ids)
        new.remove_pores_ids(rm_ids)

        np.testing.assert_array_equal(
            np.sort(old.get_pores_ids()),
            np.sort(new.get_pores_ids()),
        )

    def test_scan_min_matches(self):
        """Scan on new BorderElements IDs matches scan on old IDs."""
        rng = np.random.RandomState(99)
        n_pores, n_throats = 500, 300
        pore_pc = rng.uniform(100, 10000, n_pores).astype(np.float64)
        throat_pc = rng.uniform(100, 10000, n_throats).astype(np.float64)
        pore_ids = rng.choice(n_pores, 100, replace=False).astype(np.int64)
        throat_ids = rng.choice(n_throats, 80, replace=False).astype(np.int64)

        new = BorderElements(pore_pc, throat_pc)
        new.set_pores_ids(pore_ids.copy())
        new.set_throats_ids(throat_ids.copy())

        result_new = scan_min_element(new.get_pores_ids(), pore_pc, new.get_throats_ids(), throat_pc)
        result_old = scan_min_element(pore_ids, pore_pc, throat_ids, throat_pc)

        self.assertIsNotNone(result_new)
        self.assertIsNotNone(result_old)
        self.assertAlmostEqual(result_new[2], result_old[2], places=10)

    def test_scan_max_matches(self):
        """Scan on new BorderElements IDs matches scan on old IDs."""
        rng = np.random.RandomState(77)
        n_pores, n_throats = 500, 300
        pore_pc = rng.uniform(100, 10000, n_pores).astype(np.float64)
        throat_pc = rng.uniform(100, 10000, n_throats).astype(np.float64)
        pore_ids = rng.choice(n_pores, 100, replace=False).astype(np.int64)
        throat_ids = rng.choice(n_throats, 80, replace=False).astype(np.int64)

        new = BorderElements(pore_pc, throat_pc)
        new.set_pores_ids(pore_ids.copy())
        new.set_throats_ids(throat_ids.copy())

        result_new = scan_max_element(new.get_pores_ids(), pore_pc, new.get_throats_ids(), throat_pc)
        result_old = scan_max_element(pore_ids, pore_pc, throat_ids, throat_pc)

        self.assertIsNotNone(result_new)
        self.assertIsNotNone(result_old)
        self.assertAlmostEqual(result_new[2], result_old[2], places=10)


# ============================================================================
# Benchmark tests
# ============================================================================

class TestBenchmarkOperations(unittest.TestCase):
    """Compare timing of operations. These always pass — they print timings."""

    def _run_benchmark(self, label, sizes=(100, 500, 2000, 10000)):
        print(f"\n{'='*70}")
        print(f"  BENCHMARK: {label}")
        print(f"{'='*70}")
        print(f"  {'Size':>8} {'Old (µs)':>12} {'New (µs)':>12} {'Ratio':>8} {'Winner':>8}")
        print(f"  {'-'*56}")
        return sizes

    def test_bench_get_all_ids(self):
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
        sizes = self._run_benchmark("add_pores_ids (incremental add)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            add_ids = rng.choice(n, min(5, n), replace=False).astype(np.int64)

            def bench_old():
                o = OldBorderElements()
                o.set_pores_ids(np.arange(n // 4, dtype=np.int64))
                o.add_pores_ids(add_ids)

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(np.arange(n // 4, dtype=np.int64))
                be.add_pores_ids(add_ids)

            bench_old(); bench_new()
            t_old = _time_fn(bench_old, repeat=30)
            t_new = _time_fn(bench_new, repeat=30)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_remove_ids(self):
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

    def test_bench_scan_min(self):
        """Scan for min element using IDs from BorderElements."""
        sizes = self._run_benchmark("scan_min (find best candidate)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n // 3, replace=False).astype(np.int64)

            be = BorderElements(pore_pc, throat_pc)
            be.set_pores_ids(pore_ids)
            be.set_throats_ids(throat_ids)

            t_old = _time_fn(lambda: scan_min_element(pore_ids, pore_pc, throat_ids, throat_pc))
            t_new = _time_fn(lambda: scan_min_element(be.get_pores_ids(), pore_pc, be.get_throats_ids(), throat_pc))
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_full_invasion_step(self):
        """Full invasion step: scan + remove + add."""
        sizes = self._run_benchmark("Full invasion step (scan + remove + add)")
        for n in sizes:
            rng = np.random.RandomState(42)
            n_border = n // 3
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = rng.uniform(0, 10000, n).astype(np.float64)
            pore_ids = rng.choice(n, n_border, replace=False).astype(np.int64)
            throat_ids = rng.choice(n, n_border, replace=False).astype(np.int64)
            add_pores = rng.choice(
                np.setdiff1d(np.arange(n), pore_ids), min(3, n - n_border), replace=False
            ).astype(np.int64)
            rm_pores = pore_ids[:min(2, n_border)].copy()

            def bench_old():
                old = OldBorderElements()
                old.set_pores_ids(pore_ids.copy())
                old.set_throats_ids(throat_ids.copy())
                scan_min_element(old.get_pores_ids(), pore_pc, old.get_throats_ids(), throat_pc)
                old.remove_pores_ids(rm_pores)
                old.add_pores_ids(add_pores)

            def bench_new():
                be = BorderElements(pore_pc, throat_pc)
                be.set_pores_ids(pore_ids.copy())
                be.set_throats_ids(throat_ids.copy())
                scan_min_element(be.get_pores_ids(), pore_pc, be.get_throats_ids(), throat_pc)
                be.remove_pores_ids(rm_pores)
                be.add_pores_ids(add_pores)

            bench_old(); bench_new()
            t_old = _time_fn(bench_old, repeat=30)
            t_new = _time_fn(bench_new, repeat=30)
            ratio = t_new / t_old if t_old > 0 else float('inf')
            winner = "OLD" if ratio > 1.0 else "NEW"
            print(f"  {n:>8} {t_old:>12.1f} {t_new:>12.1f} {ratio:>7.1f}x {winner:>8}")

    def test_bench_sync_pcs(self):
        """sync_pore_pcs is a no-op."""
        sizes = self._run_benchmark("sync_pore_pcs (no-op)")
        for n in sizes:
            rng = np.random.RandomState(42)
            pore_pc = rng.uniform(0, 10000, n).astype(np.float64)
            throat_pc = np.array([0.0])
            ids = rng.choice(n, n // 3, replace=False).astype(np.int64)

            be = BorderElements(pore_pc, throat_pc)
            be.set_pores_ids(ids)

            t_new = _time_fn(lambda: be.sync_pore_pcs())
            print(f"  {n:>8} {'N/A':>12} {t_new:>12.1f} {'(no-op)':>20}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
