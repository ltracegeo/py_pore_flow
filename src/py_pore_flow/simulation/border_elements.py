"""Border element tracking with hybrid numpy-array + IndexedPriorityTree storage.

Border elements are unfilled pores/throats adjacent to filled elements and thus
candidates for invasion.  This module uses a hybrid approach:

- **Numpy arrays** for O(1) ID retrieval and fast iteration (passed to njit functions).
- **IndexedPriorityTree** for O(log N) min/max entry-pressure extraction.

The trees hold references to the network's entry_pressure arrays so PCs can
be synced efficiently via bulk rebuild after external updates.
"""

from typing import Optional, Tuple

import numpy as np

from py_pore_flow.network.constants import PORE, THROAT
from py_pore_flow.simulation.indexed_priority_tree import IndexedPriorityTree


class BorderElements:
    """Tracks border elements using numpy arrays + IndexedPriorityTree.

    The numpy arrays provide O(1) ``get_*_ids()`` and fast bulk operations
    (same performance as the original implementation).  The trees provide
    O(log N) ``get_min_element()`` / ``get_max_element()`` extraction.

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
        self._pore_pc = pore_entry_pressure
        self._throat_pc = throat_entry_pressure

        # Fast numpy arrays for iteration (same as old implementation)
        self._pore_ids = np.array([], np.int64)
        self._throat_ids = np.array([], np.int64)

        # Trees for O(log N) min/max extraction
        self._pores_tree = IndexedPriorityTree(max(len(pore_entry_pressure), 1))
        self._throats_tree = IndexedPriorityTree(max(len(throat_entry_pressure), 1))

    # ------------------------------------------------------------------ #
    # ID accessors — O(1) array return (same as old implementation)
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
        """Replace all border pores and rebuild the pore tree."""
        self._pore_ids = pore_ids
        self._rebuild_pore_tree()

    def set_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Replace all border throats and rebuild the throat tree."""
        self._throat_ids = throat_ids
        self._rebuild_throat_tree()

    # ------------------------------------------------------------------ #
    # Incremental add / remove
    # ------------------------------------------------------------------ #

    def add_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Append border pores and insert them into the tree."""
        self._pore_ids = np.concatenate((self._pore_ids, pore_ids))
        for i in range(len(pore_ids)):
            pid = int(pore_ids[i])
            self._pores_tree.insert(pid, float(self._pore_pc[pid]))

    def add_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Append border throats and insert them into the tree."""
        self._throat_ids = np.concatenate((self._throat_ids, throat_ids))
        for i in range(len(throat_ids)):
            tid = int(throat_ids[i])
            self._throats_tree.insert(tid, float(self._throat_pc[tid]))

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove border pores from both the array and the tree."""
        self._pore_ids = np.delete(
            self._pore_ids, np.isin(self._pore_ids, pore_ids)
        )
        for i in range(len(pore_ids)):
            self._pores_tree.remove(int(pore_ids[i]))

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove border throats from both the array and the tree."""
        self._throat_ids = np.delete(
            self._throat_ids, np.isin(self._throat_ids, throat_ids)
        )
        for i in range(len(throat_ids)):
            self._throats_tree.remove(int(throat_ids[i]))

    # ------------------------------------------------------------------ #
    # Entry-pressure sync (after external array updates)
    # ------------------------------------------------------------------ #

    def sync_pore_pcs(self, pore_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures from the referenced array into the pore tree.

        If *pore_ids* is ``None``, rebuilds the entire pore tree (bulk njit path,
        much faster than per-element Python-loop update).  If specific IDs are
        given, updates only those elements individually.
        """
        if pore_ids is None:
            self._rebuild_pore_tree()
        else:
            for pid_val in pore_ids:
                pid = int(pid_val)
                if self._pores_tree.contains(pid):
                    self._pores_tree.update_pc(pid, float(self._pore_pc[pid]))

    def sync_throat_pcs(self, throat_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures from the referenced array into the throat tree.

        If *throat_ids* is ``None``, rebuilds the entire throat tree (bulk njit
        path).  If specific IDs are given, updates only those elements.
        """
        if throat_ids is None:
            self._rebuild_throat_tree()
        else:
            for tid_val in throat_ids:
                tid = int(tid_val)
                if self._throats_tree.contains(tid):
                    self._throats_tree.update_pc(tid, float(self._throat_pc[tid]))

    # ------------------------------------------------------------------ #
    # O(log N) min/max element extraction
    # ------------------------------------------------------------------ #

    def get_min_element(self) -> Optional[Tuple[int, int, float]]:
        """Return ``(element_type, element_id, entry_pc)`` with the lowest PC.

        Peeks across both pore and throat trees.  Returns ``None`` if empty.
        """
        pore_min = None
        throat_min = None
        if not self._pores_tree.is_empty:
            pore_min = self._pores_tree.peek_min()
        if not self._throats_tree.is_empty:
            throat_min = self._throats_tree.peek_min()

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

        Peeks across both pore and throat trees.  Returns ``None`` if empty.
        """
        pore_max = None
        throat_max = None
        if not self._pores_tree.is_empty:
            pore_max = self._pores_tree.peek_max()
        if not self._throats_tree.is_empty:
            throat_max = self._throats_tree.peek_max()

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
        return self._pores_tree.contains(pore_id)

    def contains_throat(self, throat_id: int) -> bool:
        """O(1) check whether *throat_id* is a border throat."""
        return self._throats_tree.contains(throat_id)

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
    # Internal: tree rebuild (single bulk njit call)
    # ------------------------------------------------------------------ #

    def _rebuild_pore_tree(self) -> None:
        """Rebuild the pore tree from the current ID array and entry pressures."""
        self._pores_tree.clear()
        ids = self._pore_ids
        if len(ids) > 0:
            self._pores_tree.set_all(
                ids.astype(np.int64),
                self._pore_pc[ids].astype(np.float64),
            )

    def _rebuild_throat_tree(self) -> None:
        """Rebuild the throat tree from the current ID array and entry pressures."""
        self._throats_tree.clear()
        ids = self._throat_ids
        if len(ids) > 0:
            self._throats_tree.set_all(
                ids.astype(np.int64),
                self._throat_pc[ids].astype(np.float64),
            )
