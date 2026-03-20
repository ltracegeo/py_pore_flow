"""Border element tracking with lazy numba-native binary heaps.

Border elements are unfilled pores/throats adjacent to filled elements and thus
candidates for invasion.  This module uses:

- **Numpy arrays** for O(1) ID retrieval and fast iteration (passed to njit functions).
- **Numba-native binary heaps** for O(log N) min/max entry-pressure extraction.
- **Lazy rebuild** (Strategy 1): heaps are only rebuilt when queried after
  bulk entry-pressure changes, collapsing redundant sync operations into a single
  O(N) heap construction at peek time.
- **Incremental updates** (Strategy 4): after a lazy rebuild, subsequent add/remove
  and selective PC sync operations maintain heap validity via batch ``@njit`` calls,
  giving O(K log N) per-step cost where K is the number of changed elements.

Compared to the previous dual Red-Black tree approach:
  - Eliminates redundant full-tree rebuilds (lazy flag collapses multiple syncs).
  - Uses O(N) Floyd's heapify instead of O(N log N) sequential inserts for rebuilds.
  - Batch ``@njit`` calls for add/remove avoid per-element Python-to-Numba boundary crossing.
  - Simpler data structure with better cache locality than a pointer-based tree.
"""

from typing import Optional, Tuple

import numpy as np
from numba import njit

from py_pore_flow.network.constants import PORE, THROAT


# ══════════════════════════════════════════════════════════════════════════════
# Numba-native binary heap operations
# ══════════════════════════════════════════════════════════════════════════════
#
# The heap stores (key, value) pairs:
#   - key:  entry pressure (negated for max-heap mode)
#   - value: element index (pore or throat ID)
#
# A direct-address array ``pos[elem_id]`` gives O(1) lookup of an element's
# position in the heap, enabling O(log N) remove and key-update by ID.
#
# Tie-breaking: when keys are equal, the smaller element ID wins.  This makes
# the ordering fully deterministic and consistent with the previous RB-tree.

_HEAP_NIL = np.int64(-1)


@njit(cache=True)
def _sift_up(keys, vals, pos, i):
    """Sift element at position *i* upward to restore min-heap property."""
    while i > 0:
        parent = (i - 1) >> 1
        k_i, k_p = keys[i], keys[parent]
        if k_i < k_p or (k_i == k_p and vals[i] < vals[parent]):
            keys[i], keys[parent] = k_p, k_i
            vals[i], vals[parent] = vals[parent], vals[i]
            pos[vals[i]] = i
            pos[vals[parent]] = parent
            i = parent
        else:
            break


@njit(cache=True)
def _sift_down(keys, vals, pos, i, size):
    """Sift element at position *i* downward to restore min-heap property."""
    while True:
        sm = i
        left = (i << 1) + 1
        right = left + 1
        if left < size:
            if keys[left] < keys[sm] or (keys[left] == keys[sm] and vals[left] < vals[sm]):
                sm = left
        if right < size:
            if keys[right] < keys[sm] or (keys[right] == keys[sm] and vals[right] < vals[sm]):
                sm = right
        if sm != i:
            keys[i], keys[sm] = keys[sm], keys[i]
            vals[i], vals[sm] = vals[sm], vals[i]
            pos[vals[i]] = i
            pos[vals[sm]] = sm
            i = sm
        else:
            break


@njit(cache=True)
def _heap_build(keys, vals, pos, ids, entry_pressure, n, negate):
    """Build a min-heap from *n* border element IDs and their entry pressures.

    If *negate* is True, keys are stored as ``-entry_pressure[id]``, which turns
    the min-heap into a max-heap (peek returns the most-negative key, whose
    original pressure is the highest).

    Uses Floyd's algorithm for O(N) construction.  Clears and rebuilds the
    position array.

    Parameters
    ----------
    keys, vals, pos : np.ndarray
        Pre-allocated heap storage arrays (capacity >= *n*).
    ids : np.ndarray[int64]
        Border element IDs to insert.
    entry_pressure : np.ndarray[float64]
        Full entry-pressure array (indexed by element ID).
    n : int64
        Number of elements to insert (``== len(ids)``).
    negate : bool
        If True, negate keys for max-heap mode.

    Returns
    -------
    int64
        The heap size (== *n*).
    """
    pos[:] = _HEAP_NIL
    for i in range(n):
        idx = ids[i]
        if negate:
            keys[i] = -entry_pressure[idx]
        else:
            keys[i] = entry_pressure[idx]
        vals[i] = idx
        pos[idx] = np.int64(i)
    # Floyd's heapify — O(N)
    start = (n - 2) >> 1
    for i in range(start, -1, -1):
        _sift_down(keys, vals, pos, i, n)
    return np.int64(n)


@njit(cache=True)
def _heap_remove_one(keys, vals, pos, size, elem_id):
    """Remove one element by ID.  Returns new size.  O(log N).

    No-op if *elem_id* is not present.
    """
    i = pos[elem_id]
    if i == _HEAP_NIL:
        return size
    pos[elem_id] = _HEAP_NIL
    size -= np.int64(1)
    if i == size:
        return size  # Was the last slot — nothing to fix
    # Replace with the last element and restore heap property
    moved_id = vals[size]
    keys[i] = keys[size]
    vals[i] = moved_id
    pos[moved_id] = np.int64(i)
    _sift_up(keys, vals, pos, i)
    # After sift_up the element may have moved; sift_down from its current pos
    _sift_down(keys, vals, pos, pos[moved_id], size)
    return size


@njit(cache=True)
def _heap_batch_insert(keys, vals, pos, size, new_ids, entry_pressure, n, negate):
    """Insert *n* elements into the heap.  Skips duplicates.  O(K log(N+K))."""
    for j in range(n):
        eid = new_ids[j]
        if pos[eid] != _HEAP_NIL:
            continue
        if negate:
            key = -entry_pressure[eid]
        else:
            key = entry_pressure[eid]
        i = size
        keys[i] = key
        vals[i] = eid
        pos[eid] = np.int64(i)
        _sift_up(keys, vals, pos, i)
        size += np.int64(1)
    return size


@njit(cache=True)
def _heap_batch_remove(keys, vals, pos, size, rem_ids, n):
    """Remove *n* elements from the heap.  O(K log N)."""
    for j in range(n):
        size = _heap_remove_one(keys, vals, pos, size, rem_ids[j])
    return size


@njit(cache=True)
def _heap_batch_update_keys(keys, vals, pos, size, upd_ids, entry_pressure, n, negate):
    """Update keys for *n* elements already in the heap.  O(K log N).

    Elements not present in the heap are silently skipped.
    """
    for j in range(n):
        eid = upd_ids[j]
        i = pos[eid]
        if i == _HEAP_NIL:
            continue
        if negate:
            new_key = -entry_pressure[eid]
        else:
            new_key = entry_pressure[eid]
        if keys[i] == new_key:
            continue  # No change — skip sift
        keys[i] = new_key
        _sift_up(keys, vals, pos, i)
        _sift_down(keys, vals, pos, pos[eid], size)


# ══════════════════════════════════════════════════════════════════════════════
# _HeapState — per-heap storage container
# ══════════════════════════════════════════════════════════════════════════════


class _HeapState:
    """Pre-allocated arrays for a single binary heap (min or max).

    Attributes
    ----------
    keys : np.ndarray[float64]
        Heap keys (entry pressures, possibly negated for max-heap mode).
    vals : np.ndarray[int64]
        Element IDs stored in the heap.
    pos : np.ndarray[int64]
        Direct-address position lookup: ``pos[elem_id]`` gives the heap index
        of that element, or -1 if absent.
    size : np.int64
        Current number of elements in the heap.
    valid : bool
        If False, the heap must be rebuilt from the ID array before peeking.
    negate : bool
        If True, keys are negated (min-heap used as max-heap).
    """

    __slots__ = ("keys", "vals", "pos", "size", "valid", "negate")

    def __init__(self, capacity: int, negate: bool) -> None:
        self.keys = np.empty(capacity, np.float64)
        self.vals = np.empty(capacity, np.int64)
        self.pos = np.full(capacity, -1, np.int64)
        self.size = np.int64(0)
        self.valid = False
        self.negate = negate


# ══════════════════════════════════════════════════════════════════════════════
# BorderElements
# ══════════════════════════════════════════════════════════════════════════════


class BorderElements:
    """Tracks border elements using numpy arrays + lazy numba-native heaps.

    The numpy arrays provide O(1) ``get_*_ids()`` and fast bulk operations
    (same interface as the original implementation).  The heaps provide
    O(log N) ``get_min_element()`` / ``get_max_element()`` extraction, with
    lazy rebuild to avoid redundant synchronization overhead.

    **Lazy strategy**: bulk PC changes (``sync_*_pcs(None)``) only set an
    invalidation flag.  The actual O(N) heap rebuild is deferred until the
    next ``get_min_element()`` / ``get_max_element()`` call.  Incremental
    changes (add/remove/selective sync) maintain the heap in O(K log N) if
    valid, or are skipped (no-op) if already invalidated.

    Parameters
    ----------
    pore_entry_pressure : np.ndarray
        Reference to ``pn_core.pore.entry_pressure``.
    throat_entry_pressure : np.ndarray
        Reference to ``pn_core.throat.entry_pressure``.
    """

    def __init__(
        self,
        pore_entry_pressure: np.ndarray,
        throat_entry_pressure: np.ndarray,
    ) -> None:
        self._pore_pc: np.ndarray = pore_entry_pressure
        self._throat_pc: np.ndarray = throat_entry_pressure

        # Fast numpy arrays for iteration (same as original implementation)
        self._pore_ids: np.ndarray = np.array([], np.int64)
        self._throat_ids: np.ndarray = np.array([], np.int64)

        n_pores = max(len(pore_entry_pressure), 1)
        n_throats = max(len(throat_entry_pressure), 1)

        # Heaps: min and max for each element type (lazily built on first peek)
        self._pore_min = _HeapState(n_pores, negate=False)
        self._pore_max = _HeapState(n_pores, negate=True)
        self._throat_min = _HeapState(n_throats, negate=False)
        self._throat_max = _HeapState(n_throats, negate=True)

        # O(1) membership tracking (independent of heap validity)
        self._pore_member: np.ndarray = np.zeros(n_pores, np.bool_)
        self._throat_member: np.ndarray = np.zeros(n_throats, np.bool_)

    # ------------------------------------------------------------------ #
    # ID accessors — O(1) array return
    # ------------------------------------------------------------------ #

    def get_pores_ids(self) -> np.ndarray:
        """Return the stored border pore indices array (O(1))."""
        return self._pore_ids

    def get_throats_ids(self) -> np.ndarray:
        """Return the stored border throat indices array (O(1))."""
        return self._throat_ids

    # ------------------------------------------------------------------ #
    # Bulk set (replaces all stored elements)
    # ------------------------------------------------------------------ #

    def set_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Replace all border pores and invalidate pore heaps."""
        if len(self._pore_ids) > 0:
            self._pore_member[self._pore_ids] = False
        self._pore_ids = pore_ids
        if len(pore_ids) > 0:
            self._pore_member[pore_ids] = True
        self._pore_min.valid = False
        self._pore_max.valid = False

    def set_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Replace all border throats and invalidate throat heaps."""
        if len(self._throat_ids) > 0:
            self._throat_member[self._throat_ids] = False
        self._throat_ids = throat_ids
        if len(throat_ids) > 0:
            self._throat_member[throat_ids] = True
        self._throat_min.valid = False
        self._throat_max.valid = False

    # ------------------------------------------------------------------ #
    # Incremental add / remove
    # ------------------------------------------------------------------ #

    def add_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Append border pores and update valid heaps incrementally."""
        self._pore_ids = np.concatenate((self._pore_ids, pore_ids))
        n = len(pore_ids)
        if n > 0:
            self._pore_member[pore_ids] = True
            self._incr_insert(self._pore_min, pore_ids, self._pore_pc, n)
            self._incr_insert(self._pore_max, pore_ids, self._pore_pc, n)

    def add_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Append border throats and update valid heaps incrementally."""
        self._throat_ids = np.concatenate((self._throat_ids, throat_ids))
        n = len(throat_ids)
        if n > 0:
            self._throat_member[throat_ids] = True
            self._incr_insert(self._throat_min, throat_ids, self._throat_pc, n)
            self._incr_insert(self._throat_max, throat_ids, self._throat_pc, n)

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove border pores from both the array and valid heaps."""
        self._pore_ids = np.delete(self._pore_ids, np.isin(self._pore_ids, pore_ids))
        n = len(pore_ids)
        if n > 0:
            self._pore_member[pore_ids] = False
            self._incr_remove(self._pore_min, pore_ids, n)
            self._incr_remove(self._pore_max, pore_ids, n)

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove border throats from both the array and valid heaps."""
        self._throat_ids = np.delete(self._throat_ids, np.isin(self._throat_ids, throat_ids))
        n = len(throat_ids)
        if n > 0:
            self._throat_member[throat_ids] = False
            self._incr_remove(self._throat_min, throat_ids, n)
            self._incr_remove(self._throat_max, throat_ids, n)

    # ------------------------------------------------------------------ #
    # Entry-pressure sync (lazy — Strategy 1)
    # ------------------------------------------------------------------ #

    def sync_pore_pcs(self, pore_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures for pore border elements.

        If *pore_ids* is ``None`` (bulk change after ``_update_entry_pressures``),
        both pore heaps are lazily invalidated — the actual O(N) rebuild is
        deferred until the next peek.  If specific IDs are given, only those
        keys are updated in valid heaps (O(K log N)).
        """
        if pore_ids is None:
            self._pore_min.valid = False
            self._pore_max.valid = False
        else:
            n = len(pore_ids)
            if n > 0:
                self._incr_update(self._pore_min, pore_ids, self._pore_pc, n)
                self._incr_update(self._pore_max, pore_ids, self._pore_pc, n)

    def sync_throat_pcs(self, throat_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures for throat border elements.

        Lazy invalidation when *throat_ids* is ``None``; incremental key
        update otherwise.
        """
        if throat_ids is None:
            self._throat_min.valid = False
            self._throat_max.valid = False
        else:
            n = len(throat_ids)
            if n > 0:
                self._incr_update(self._throat_min, throat_ids, self._throat_pc, n)
                self._incr_update(self._throat_max, throat_ids, self._throat_pc, n)

    # ------------------------------------------------------------------ #
    # O(log N) min/max element extraction
    # ------------------------------------------------------------------ #

    def get_min_element(self) -> Optional[Tuple[int, int, float]]:
        """Return ``(element_type, element_id, entry_pc)`` with the lowest PC.

        Lazily rebuilds invalidated heaps before peeking.  Returns ``None``
        if no border elements exist.
        """
        self._ensure_valid(self._pore_min, self._pore_ids, self._pore_pc)
        self._ensure_valid(self._throat_min, self._throat_ids, self._throat_pc)

        pore_min = self._peek_min(self._pore_min)
        throat_min = self._peek_min(self._throat_min)

        if pore_min is None and throat_min is None:
            return None
        if pore_min is None:
            return (THROAT, throat_min[0], throat_min[1])
        if throat_min is None:
            return (PORE, pore_min[0], pore_min[1])
        if pore_min[1] <= throat_min[1]:
            return (PORE, pore_min[0], pore_min[1])
        return (THROAT, throat_min[0], throat_min[1])

    def get_max_element(self) -> Optional[Tuple[int, int, float]]:
        """Return ``(element_type, element_id, entry_pc)`` with the highest PC.

        Lazily rebuilds invalidated heaps before peeking.  Returns ``None``
        if no border elements exist.
        """
        self._ensure_valid(self._pore_max, self._pore_ids, self._pore_pc)
        self._ensure_valid(self._throat_max, self._throat_ids, self._throat_pc)

        pore_max = self._peek_negated(self._pore_max)
        throat_max = self._peek_negated(self._throat_max)

        if pore_max is None and throat_max is None:
            return None
        if pore_max is None:
            return (THROAT, throat_max[0], throat_max[1])
        if throat_max is None:
            return (PORE, pore_max[0], pore_max[1])
        if pore_max[1] >= throat_max[1]:
            return (PORE, pore_max[0], pore_max[1])
        return (THROAT, throat_max[0], throat_max[1])

    # ------------------------------------------------------------------ #
    # Membership queries
    # ------------------------------------------------------------------ #

    def contains_pore(self, pore_id: int) -> bool:
        """O(1) check whether *pore_id* is a border pore."""
        return bool(self._pore_member[pore_id])

    def contains_throat(self, throat_id: int) -> bool:
        """O(1) check whether *throat_id* is a border throat."""
        return bool(self._throat_member[throat_id])

    @property
    def pore_count(self) -> int:
        """Number of border pores."""
        return len(self._pore_ids)

    @property
    def throat_count(self) -> int:
        """Number of border throats."""
        return len(self._throat_ids)

    @property
    def is_empty(self) -> bool:
        """True if no border elements are stored."""
        return len(self._pore_ids) == 0 and len(self._throat_ids) == 0

    # ------------------------------------------------------------------ #
    # Internal: heap management helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_valid(h: _HeapState, ids: np.ndarray, entry_pc: np.ndarray) -> None:
        """Rebuild heap *h* from current IDs and PCs if invalidated.

        Uses Floyd's O(N) heapify.  If *ids* is empty, resets the position
        array and sets size to 0.
        """
        if h.valid:
            return
        n = len(ids)
        if n > 0:
            h.size = _heap_build(h.keys, h.vals, h.pos, ids, entry_pc, np.int64(n), h.negate)
        else:
            h.size = np.int64(0)
            h.pos[:] = _HEAP_NIL
        h.valid = True

    @staticmethod
    def _peek_min(h: _HeapState) -> Optional[Tuple[int, float]]:
        """Peek at the root of a non-negated min-heap.

        Returns ``(elem_id, entry_pc)`` or ``None`` if empty.
        """
        if h.size == 0:
            return None
        return (int(h.vals[0]), float(h.keys[0]))

    @staticmethod
    def _peek_negated(h: _HeapState) -> Optional[Tuple[int, float]]:
        """Peek at the root of a negated-key min-heap (= max element).

        The actual entry pressure is the negation of the stored key.
        Returns ``(elem_id, actual_pc)`` or ``None`` if empty.
        """
        if h.size == 0:
            return None
        return (int(h.vals[0]), float(-h.keys[0]))

    @staticmethod
    def _incr_insert(h: _HeapState, ids: np.ndarray, entry_pc: np.ndarray, n: int) -> None:
        """Insert elements into heap *h* if valid; skip if invalidated."""
        if not h.valid:
            return
        h.size = _heap_batch_insert(h.keys, h.vals, h.pos, h.size, ids, entry_pc, np.int64(n), h.negate)

    @staticmethod
    def _incr_remove(h: _HeapState, ids: np.ndarray, n: int) -> None:
        """Remove elements from heap *h* if valid; skip if invalidated."""
        if not h.valid:
            return
        h.size = _heap_batch_remove(h.keys, h.vals, h.pos, h.size, ids, np.int64(n))

    @staticmethod
    def _incr_update(h: _HeapState, ids: np.ndarray, entry_pc: np.ndarray, n: int) -> None:
        """Update keys in heap *h* if valid; skip if invalidated."""
        if not h.valid:
            return
        _heap_batch_update_keys(h.keys, h.vals, h.pos, h.size, ids, entry_pc, np.int64(n), h.negate)
