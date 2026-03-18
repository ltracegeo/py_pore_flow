"""
Micro-benchmark comparing BorderElements implementations.

Simulates the add/remove/get pattern that occurs on every invasion step:
  - get  border IDs  (passed to the element-selection jit function)
  - add  1-3 new neighbours
  - remove  the just-filled element (and maybe trapped neighbours)
  - get  border IDs again (for entry-pressure update)

Run with:  python tests/benchmarks/border_elements_benchmark.py
"""
import timeit
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Implementations under test
# ──────────────────────────────────────────────────────────────────────────────

class BorderElementsOriginal:
    """Baseline: numpy array rebuilt on every add/remove via concatenate/delete."""

    def __init__(self):
        self.pores_ids = np.empty(0, np.int64)

    def set_pores_ids(self, ids: np.ndarray) -> None:
        self.pores_ids = ids.copy()

    def get_pores_ids(self) -> np.ndarray:
        return self.pores_ids

    def add_pores_ids(self, ids: np.ndarray) -> None:
        self.pores_ids = np.concatenate((self.pores_ids, ids))

    def remove_pores_ids(self, ids: np.ndarray) -> None:
        self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, ids))


class BorderElementsSet:
    """Set-based with lazy numpy rebuild — the broken attempt."""

    def __init__(self):
        self._set: set = set()
        self._arr = np.empty(0, np.int64)
        self._dirty = False

    def set_pores_ids(self, ids: np.ndarray) -> None:
        self._set = set(ids.tolist())
        self._dirty = True

    def get_pores_ids(self) -> np.ndarray:
        if self._dirty:
            self._arr = (
                np.array(list(self._set), np.int64) if self._set
                else np.empty(0, np.int64)
            )
            self._dirty = False
        return self._arr

    def add_pores_ids(self, ids: np.ndarray) -> None:
        if len(ids):
            self._set.update(ids.tolist())
            self._dirty = True

    def remove_pores_ids(self, ids: np.ndarray) -> None:
        if len(ids):
            self._set.difference_update(ids.tolist())
            self._dirty = True


class BorderElementsBuffer:
    """Pre-allocated buffer: O(1) get (view), O(K) add, O(border) remove."""

    def __init__(self, max_size: int = 256):
        self._buf = np.empty(max(max_size, 256), np.int64)
        self._count = 0

    def set_pores_ids(self, ids: np.ndarray) -> None:
        n = len(ids)
        self._buf[:n] = ids
        self._count = n

    def get_pores_ids(self) -> np.ndarray:
        return self._buf[:self._count]          # O(1) view, zero allocation

    def add_pores_ids(self, ids: np.ndarray) -> None:
        n = len(ids)
        if n == 0:
            return
        self._buf[self._count:self._count + n] = ids
        self._count += n

    def remove_pores_ids(self, ids: np.ndarray) -> None:
        if len(ids) == 0 or self._count == 0:
            return
        n = self._count
        mask = ~np.isin(self._buf[:n], ids)
        new_count = int(mask.sum())
        if new_count < n:
            self._buf[:new_count] = self._buf[:n][mask]
            self._count = new_count


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def make_invasion_sequence(n_border: int, n_invasions: int):
    """
    Build a sequence of (add_ids, remove_ids) pairs that keep the border size
    roughly stable so the buffer never overflows.

    Strategy: a rotating pool of size n_border*2.  Each invasion:
      - adds   the next 2 IDs from the pool (wrapping around)
      - removes 2 IDs that are currently in the border (always present)
    """
    pool_size = n_border + n_invasions * 3 + 10   # enough unique IDs

    # Pre-compute per-invasion pairs
    adds    = []
    removes = []

    # Track which IDs are currently in the border so removes are always valid.
    in_border = set(range(n_border))
    border_list = list(range(n_border))   # ordered so we pop from front
    next_id = n_border

    for _ in range(n_invasions):
        # Remove the "oldest" 2 ids (simulate filling + trapping)
        rem = border_list[:2]
        border_list = border_list[2:]
        for r in rem:
            in_border.discard(r)

        # Add 2 fresh ids
        add = [next_id, next_id + 1]
        next_id += 2
        for a in add:
            in_border.add(a)
            border_list.append(a)

        adds.append(np.array(add, np.int64))
        removes.append(np.array(rem, np.int64))

    return adds, removes


def bench_cycle(cls, n_border: int, adds, removes, number: int = 3) -> float:
    """
    Time the full invasion inner loop:  get → add → remove → get.

    Re-initialises the object before each timed run.
    """
    base_ids = np.arange(n_border, dtype=np.int64)
    max_buf  = n_border + len(adds) * 2 + 100

    def _make():
        b = cls(max_buf) if cls is BorderElementsBuffer else cls()
        b.set_pores_ids(base_ids)
        if cls is BorderElementsSet:
            b.get_pores_ids()           # flush initial dirty
        return b

    def _run():
        b = _make()
        for add_ids, rem_ids in zip(adds, removes):
            _ = b.get_pores_ids()
            b.add_pores_ids(add_ids)
            b.remove_pores_ids(rem_ids)
            _ = b.get_pores_ids()

    return timeit.timeit(_run, number=number) / number


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    scenarios = [
        (500,    1_000, "small  border  (500 elems,  1 000 invasions)"),
        (5_000,  2_000, "medium border (5 000 elems, 2 000 invasions)"),
        (20_000, 2_000, "large  border (20 000 elems, 2 000 invasions)"),
    ]

    print("=" * 72)
    print(f"{'Scenario':<46}  {'Orig':>6}  {'Set':>6}  {'Buf':>6}  {'Buf/Orig':>8}")
    print("-" * 72)

    for n_border, n_inv, label in scenarios:
        adds, removes = make_invasion_sequence(n_border, n_inv)

        t_orig = bench_cycle(BorderElementsOriginal, n_border, adds, removes)
        t_set  = bench_cycle(BorderElementsSet,      n_border, adds, removes)
        t_buf  = bench_cycle(BorderElementsBuffer,   n_border, adds, removes)

        print(
            f"{label:<46}  {t_orig:>5.3f}s  {t_set:>5.3f}s  {t_buf:>5.3f}s"
            f"  {t_buf/t_orig:>7.2f}x"
        )

    print("=" * 72)
    print("(Buf/Orig < 1.00 = faster than original)")


if __name__ == "__main__":
    main()
