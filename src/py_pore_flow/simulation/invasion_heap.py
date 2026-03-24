"""
Indexed binary heap for efficient invasion element selection.

Provides O(log n) insert, pop_min/pop_max, and update_key operations,
replacing the O(n) linear scan used in _get_element_to_fill().

Elements are identified by a composite key: (element_type, element_id),
mapped to a flat index for heap storage. Pore IDs map to [0, n_pores),
throat IDs map to [n_pores, n_pores + n_throats).
"""

import numba as nb
import numpy as np

from py_pore_flow.network.constants import PORE, THROAT


# ── Numba‑compiled heap primitives ────────────────────────────────────────────

@nb.njit(cache=True)
def _swap(heap_keys, heap_idx, idx_to_pos, pos_a, pos_b):
    """Swap two entries in the heap and update the reverse index."""
    heap_keys[pos_a], heap_keys[pos_b] = heap_keys[pos_b], heap_keys[pos_a]
    heap_idx[pos_a], heap_idx[pos_b] = heap_idx[pos_b], heap_idx[pos_a]
    idx_to_pos[heap_idx[pos_a]] = pos_a
    idx_to_pos[heap_idx[pos_b]] = pos_b


@nb.njit(cache=True)
def _sift_up_min(heap_keys, heap_idx, idx_to_pos, pos):
    """Sift an element up in a min-heap."""
    while pos > 0:
        parent = (pos - 1) >> 1
        if heap_keys[pos] < heap_keys[parent]:
            _swap(heap_keys, heap_idx, idx_to_pos, pos, parent)
            pos = parent
        else:
            break


@nb.njit(cache=True)
def _sift_down_min(heap_keys, heap_idx, idx_to_pos, pos, size):
    """Sift an element down in a min-heap."""
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        smallest = pos
        if left < size and heap_keys[left] < heap_keys[smallest]:
            smallest = left
        if right < size and heap_keys[right] < heap_keys[smallest]:
            smallest = right
        if smallest != pos:
            _swap(heap_keys, heap_idx, idx_to_pos, pos, smallest)
            pos = smallest
        else:
            break


@nb.njit(cache=True)
def _sift_up_max(heap_keys, heap_idx, idx_to_pos, pos):
    """Sift an element up in a max-heap."""
    while pos > 0:
        parent = (pos - 1) >> 1
        if heap_keys[pos] > heap_keys[parent]:
            _swap(heap_keys, heap_idx, idx_to_pos, pos, parent)
            pos = parent
        else:
            break


@nb.njit(cache=True)
def _sift_down_max(heap_keys, heap_idx, idx_to_pos, pos, size):
    """Sift an element down in a max-heap."""
    while True:
        left = 2 * pos + 1
        right = 2 * pos + 2
        largest = pos
        if left < size and heap_keys[left] > heap_keys[largest]:
            largest = left
        if right < size and heap_keys[right] > heap_keys[largest]:
            largest = right
        if largest != pos:
            _swap(heap_keys, heap_idx, idx_to_pos, pos, largest)
            pos = largest
        else:
            break


# ── Public API ────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def heap_insert(heap_keys, heap_idx, idx_to_pos, size, flat_idx, key, is_min_heap):
    """Insert an element into the heap. Returns new size."""
    pos = size
    heap_keys[pos] = key
    heap_idx[pos] = flat_idx
    idx_to_pos[flat_idx] = pos
    size += 1
    if is_min_heap:
        _sift_up_min(heap_keys, heap_idx, idx_to_pos, pos)
    else:
        _sift_up_max(heap_keys, heap_idx, idx_to_pos, pos)
    return size


@nb.njit(cache=True)
def heap_pop(heap_keys, heap_idx, idx_to_pos, size, is_min_heap):
    """Pop the root element. Returns (flat_idx, key, new_size)."""
    root_idx = heap_idx[0]
    root_key = heap_keys[0]
    size -= 1
    if size > 0:
        _swap(heap_keys, heap_idx, idx_to_pos, 0, size)
        idx_to_pos[heap_idx[size]] = -1  # mark removed position
        if is_min_heap:
            _sift_down_min(heap_keys, heap_idx, idx_to_pos, 0, size)
        else:
            _sift_down_max(heap_keys, heap_idx, idx_to_pos, 0, size)
    else:
        idx_to_pos[root_idx] = -1
    return root_idx, root_key, size


@nb.njit(cache=True)
def heap_remove(heap_keys, heap_idx, idx_to_pos, size, flat_idx, is_min_heap):
    """Remove an element by its flat index. Returns new size."""
    pos = idx_to_pos[flat_idx]
    if pos < 0 or pos >= size:
        return size  # not in heap
    size -= 1
    if pos == size:
        idx_to_pos[flat_idx] = -1
        return size
    _swap(heap_keys, heap_idx, idx_to_pos, pos, size)
    idx_to_pos[heap_idx[size]] = -1  # mark old last position
    # Re-heapify at the swapped position
    if is_min_heap:
        _sift_up_min(heap_keys, heap_idx, idx_to_pos, pos)
        _sift_down_min(heap_keys, heap_idx, idx_to_pos, pos, size)
    else:
        _sift_up_max(heap_keys, heap_idx, idx_to_pos, pos)
        _sift_down_max(heap_keys, heap_idx, idx_to_pos, pos, size)
    return size


@nb.njit(cache=True)
def heap_update_key(heap_keys, heap_idx, idx_to_pos, size, flat_idx, new_key, is_min_heap):
    """Update the key of an element already in the heap."""
    pos = idx_to_pos[flat_idx]
    if pos < 0 or pos >= size:
        return  # not in heap
    old_key = heap_keys[pos]
    heap_keys[pos] = new_key
    if is_min_heap:
        if new_key < old_key:
            _sift_up_min(heap_keys, heap_idx, idx_to_pos, pos)
        else:
            _sift_down_min(heap_keys, heap_idx, idx_to_pos, pos, size)
    else:
        if new_key > old_key:
            _sift_up_max(heap_keys, heap_idx, idx_to_pos, pos)
        else:
            _sift_down_max(heap_keys, heap_idx, idx_to_pos, pos, size)


@nb.njit(cache=True)
def heap_contains(idx_to_pos, flat_idx, size):
    """Check if a flat index is currently in the heap."""
    pos = idx_to_pos[flat_idx]
    return 0 <= pos < size


@nb.njit(cache=True)
def heap_peek_key(heap_keys, size):
    """Return the key at the root without removing it. Returns inf if empty."""
    if size == 0:
        return np.inf
    return heap_keys[0]


# ── Flat index helpers ────────────────────────────────────────────────────────

@nb.njit(cache=True)
def to_flat_idx(element_type, element_id, n_pores):
    """Convert (element_type, element_id) to flat index."""
    if element_type == PORE:
        return element_id
    else:
        return n_pores + element_id


@nb.njit(cache=True)
def from_flat_idx(flat_idx, n_pores):
    """Convert flat index back to (element_type, element_id)."""
    if flat_idx < n_pores:
        return PORE, flat_idx
    else:
        return THROAT, flat_idx - n_pores


# ── High-level wrapper class ─────────────────────────────────────────────────

class InvasionHeap:
    """
    Indexed binary heap for tracking border element entry pressures.

    For drainage / 2nd drainage: min-heap (lowest entry pressure invaded first).
    For imbibition: max-heap (highest entry pressure invaded first).

    Supports O(log n) insert, pop, remove, and update_key operations.

    Args:
        n_pores: Number of pores in the network.
        n_throats: Number of throats in the network.
        is_min_heap: True for drainage/2nd drainage, False for imbibition.
    """

    def __init__(self, n_pores: int, n_throats: int, is_min_heap: bool = True):
        self.n_pores = n_pores
        self.n_throats = n_throats
        self.is_min_heap = is_min_heap
        capacity = n_pores + n_throats
        self.heap_keys = np.empty(capacity, dtype=np.float64)
        self.heap_idx = np.empty(capacity, dtype=np.int64)
        self.idx_to_pos = np.full(capacity, -1, dtype=np.int64)
        self.size = 0

    def insert(self, element_type: int, element_id: int, key: float) -> None:
        """Insert an element with given entry pressure key."""
        flat = to_flat_idx(element_type, element_id, self.n_pores)
        if heap_contains(self.idx_to_pos, flat, self.size):
            heap_update_key(
                self.heap_keys, self.heap_idx, self.idx_to_pos,
                self.size, flat, key, self.is_min_heap,
            )
        else:
            self.size = heap_insert(
                self.heap_keys, self.heap_idx, self.idx_to_pos,
                self.size, flat, key, self.is_min_heap,
            )

    def pop(self):
        """
        Pop the element with the best (min or max) entry pressure.

        Returns:
            Tuple of (element_type, element_id, key) or None if empty.
        """
        if self.size == 0:
            return None
        flat, key, self.size = heap_pop(
            self.heap_keys, self.heap_idx, self.idx_to_pos,
            self.size, self.is_min_heap,
        )
        etype, eid = from_flat_idx(flat, self.n_pores)
        return etype, eid, key

    def peek(self):
        """
        Peek at the best element without removing it.

        Returns:
            Tuple of (element_type, element_id, key) or None if empty.
        """
        if self.size == 0:
            return None
        flat = self.heap_idx[0]
        key = self.heap_keys[0]
        etype, eid = from_flat_idx(flat, self.n_pores)
        return etype, eid, key

    def remove(self, element_type: int, element_id: int) -> None:
        """Remove an element from the heap (e.g., when it becomes trapped)."""
        flat = to_flat_idx(element_type, element_id, self.n_pores)
        self.size = heap_remove(
            self.heap_keys, self.heap_idx, self.idx_to_pos,
            self.size, flat, self.is_min_heap,
        )

    def update_key(self, element_type: int, element_id: int, new_key: float) -> None:
        """Update the entry pressure of an element already in the heap."""
        flat = to_flat_idx(element_type, element_id, self.n_pores)
        if heap_contains(self.idx_to_pos, flat, self.size):
            heap_update_key(
                self.heap_keys, self.heap_idx, self.idx_to_pos,
                self.size, flat, new_key, self.is_min_heap,
            )

    def contains(self, element_type: int, element_id: int) -> bool:
        """Check if element is in the heap."""
        flat = to_flat_idx(element_type, element_id, self.n_pores)
        return heap_contains(self.idx_to_pos, flat, self.size)

    def clear(self) -> None:
        """Remove all elements from the heap."""
        self.idx_to_pos[:] = -1
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return self.size > 0
