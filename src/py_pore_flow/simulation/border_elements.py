"""Border element tracking with lazy numpy argmin/argmax extraction.

Border elements are unfilled pores/throats adjacent to filled elements and thus
candidates for invasion.  This module uses:

- **Numpy arrays** for O(1) ID retrieval and fast iteration (passed to njit functions).
- **Lazy numpy argmin/argmax** for min/max entry-pressure extraction: the result
  is cached and invalidated on any modification (add/remove/sync).  The next peek
  recomputes via a single vectorized numpy call.

This approach has **zero maintenance overhead** — no data structure to insert into,
remove from, or keep synchronized.  The numpy argmin/argmax on a contiguous slice
is SIMD-optimized and cache-friendly, making it faster than both the old ``@njit``
linear scan (random-access indexed lookups) and a heap rebuild (3-array Floyd's
heapify) for realistic network sizes.
"""

from typing import Optional, Tuple

import numpy as np

from py_pore_flow.network.constants import PORE, THROAT


# ══════════════════════════════════════════════════════════════════════════════
# Cached extremum — stores the result of the last argmin/argmax
# ══════════════════════════════════════════════════════════════════════════════


class _CachedExtremum:
    """Cached min or max result for one element type (pores or throats).

    Attributes
    ----------
    best_id : int
        Element ID of the cached min/max, or -1 if empty.
    best_pc : float
        Entry pressure of the cached element.
    valid : bool
        If False, the cache must be recomputed before peeking.
    use_max : bool
        If True, compute argmax instead of argmin.
    """

    __slots__ = ("best_id", "best_pc", "valid", "use_max")

    def __init__(self, use_max: bool) -> None:
        self.best_id: int = -1
        self.best_pc: float = 0.0
        self.valid: bool = False
        self.use_max: bool = use_max


# ══════════════════════════════════════════════════════════════════════════════
# BorderElements
# ══════════════════════════════════════════════════════════════════════════════


class BorderElements:
    """Tracks border elements using numpy arrays + lazy numpy argmin/argmax.

    The numpy arrays provide O(1) ``get_*_ids()`` and fast bulk operations
    (same interface as the original implementation).  Min/max extraction uses
    ``np.argmin`` / ``np.argmax`` on ``entry_pressure[border_ids]``, with lazy
    caching to avoid redundant recomputation.

    **Lazy strategy**: any modification (add, remove, bulk or selective PC sync)
    invalidates the cached min/max.  The next ``get_min_element()`` /
    ``get_max_element()`` call recomputes via a single numpy call (~25µs for
    20K elements) and caches the result.

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

        # Cached min/max results (lazily computed on peek)
        self._pore_min = _CachedExtremum(use_max=False)
        self._pore_max = _CachedExtremum(use_max=True)
        self._throat_min = _CachedExtremum(use_max=False)
        self._throat_max = _CachedExtremum(use_max=True)

        # O(1) membership tracking
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
        """Replace all border pores and invalidate cached results."""
        if len(self._pore_ids) > 0:
            self._pore_member[self._pore_ids] = False
        self._pore_ids = pore_ids
        if len(pore_ids) > 0:
            self._pore_member[pore_ids] = True
        self._pore_min.valid = False
        self._pore_max.valid = False

    def set_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Replace all border throats and invalidate cached results."""
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
        """Append border pores and invalidate pore caches."""
        self._pore_ids = np.concatenate((self._pore_ids, pore_ids))
        if len(pore_ids) > 0:
            self._pore_member[pore_ids] = True
            self._pore_min.valid = False
            self._pore_max.valid = False

    def add_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Append border throats and invalidate throat caches."""
        self._throat_ids = np.concatenate((self._throat_ids, throat_ids))
        if len(throat_ids) > 0:
            self._throat_member[throat_ids] = True
            self._throat_min.valid = False
            self._throat_max.valid = False

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove border pores and invalidate pore caches."""
        self._pore_ids = np.delete(self._pore_ids, np.isin(self._pore_ids, pore_ids))
        if len(pore_ids) > 0:
            self._pore_member[pore_ids] = False
            self._pore_min.valid = False
            self._pore_max.valid = False

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove border throats and invalidate throat caches."""
        self._throat_ids = np.delete(self._throat_ids, np.isin(self._throat_ids, throat_ids))
        if len(throat_ids) > 0:
            self._throat_member[throat_ids] = False
            self._throat_min.valid = False
            self._throat_max.valid = False

    # ------------------------------------------------------------------ #
    # Entry-pressure sync (lazy invalidation)
    # ------------------------------------------------------------------ #

    def sync_pore_pcs(self, pore_ids: Optional[np.ndarray] = None) -> None:
        """Invalidate pore caches after external entry-pressure changes.

        Both bulk (``pore_ids=None``) and selective calls invalidate — the
        next peek recomputes via numpy argmin/argmax.
        """
        self._pore_min.valid = False
        self._pore_max.valid = False

    def sync_throat_pcs(self, throat_ids: Optional[np.ndarray] = None) -> None:
        """Invalidate throat caches after external entry-pressure changes."""
        self._throat_min.valid = False
        self._throat_max.valid = False

    # ------------------------------------------------------------------ #
    # Min/max element extraction
    # ------------------------------------------------------------------ #

    def get_min_element(self) -> Optional[Tuple[int, int, float]]:
        """Return ``(element_type, element_id, entry_pc)`` with the lowest PC.

        Recomputes via numpy argmin if the cache is stale.  Returns ``None``
        if no border elements exist.
        """
        self._ensure_valid(self._pore_min, self._pore_ids, self._pore_pc)
        self._ensure_valid(self._throat_min, self._throat_ids, self._throat_pc)

        p = self._pore_min
        t = self._throat_min
        p_empty = p.best_id == -1
        t_empty = t.best_id == -1

        if p_empty and t_empty:
            return None
        if p_empty:
            return (THROAT, t.best_id, t.best_pc)
        if t_empty:
            return (PORE, p.best_id, p.best_pc)
        if p.best_pc <= t.best_pc:
            return (PORE, p.best_id, p.best_pc)
        return (THROAT, t.best_id, t.best_pc)

    def get_max_element(self) -> Optional[Tuple[int, int, float]]:
        """Return ``(element_type, element_id, entry_pc)`` with the highest PC.

        Recomputes via numpy argmax if the cache is stale.  Returns ``None``
        if no border elements exist.
        """
        self._ensure_valid(self._pore_max, self._pore_ids, self._pore_pc)
        self._ensure_valid(self._throat_max, self._throat_ids, self._throat_pc)

        p = self._pore_max
        t = self._throat_max
        p_empty = p.best_id == -1
        t_empty = t.best_id == -1

        if p_empty and t_empty:
            return None
        if p_empty:
            return (THROAT, t.best_id, t.best_pc)
        if t_empty:
            return (PORE, p.best_id, p.best_pc)
        if p.best_pc >= t.best_pc:
            return (PORE, p.best_id, p.best_pc)
        return (THROAT, t.best_id, t.best_pc)

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
    # Internal: lazy cache recomputation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_valid(c: _CachedExtremum, ids: np.ndarray, entry_pc: np.ndarray) -> None:
        """Recompute the cached extremum if invalidated.

        Uses ``np.argmin`` or ``np.argmax`` on ``entry_pc[ids]`` — a single
        vectorized numpy call on a contiguous temporary array.
        """
        if c.valid:
            return
        n = len(ids)
        if n > 0:
            pcs = entry_pc[ids]
            if c.use_max:
                best_local = int(np.argmax(pcs))
            else:
                best_local = int(np.argmin(pcs))
            c.best_id = int(ids[best_local])
            c.best_pc = float(pcs[best_local])
        else:
            c.best_id = -1
            c.best_pc = 0.0
        c.valid = True
