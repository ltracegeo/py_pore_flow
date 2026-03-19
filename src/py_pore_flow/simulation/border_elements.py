"""Border element tracking backed by IndexedPriorityTree.

Border elements are unfilled pores/throats that are adjacent to filled elements
and thus candidates for invasion. This module provides O(log N) insert, remove,
and min/max entry-pressure extraction using dual Red-Black trees (one for pores,
one for throats).

The trees store (element_index, entry_pressure) pairs and hold references to
the network's entry_pressure arrays so PCs can be synced after external updates.
"""

from typing import Optional, Tuple

import numpy as np

from py_pore_flow.network.constants import PORE, THROAT
from py_pore_flow.simulation.indexed_priority_tree import IndexedPriorityTree


class BorderElements:
    """Tracks border elements using IndexedPriorityTree for O(log N) operations.

    Stores references to the network's ``entry_pressure`` arrays.  When entry
    pressures are modified externally (e.g. during imbibition PC recalculation),
    call :meth:`sync_pore_pcs` / :meth:`sync_throat_pcs` to update the trees.

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
        self._pores_tree = IndexedPriorityTree(max(len(pore_entry_pressure), 1))
        self._throats_tree = IndexedPriorityTree(max(len(throat_entry_pressure), 1))

    # ------------------------------------------------------------------ #
    # Backward-compatible ID accessors
    # ------------------------------------------------------------------ #

    def get_pores_ids(self) -> np.ndarray:
        """Return sorted array of border pore indices."""
        return self._pores_tree.get_all_ids()

    def get_throats_ids(self) -> np.ndarray:
        """Return sorted array of border throat indices."""
        return self._throats_tree.get_all_ids()

    # ------------------------------------------------------------------ #
    # Bulk set (replaces all stored elements)
    # ------------------------------------------------------------------ #

    def set_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Replace all border pores. Entry pressures are read from the referenced array."""
        self._pores_tree.clear()
        if len(pore_ids) > 0:
            pcs = self._pore_pc[pore_ids]
            self._pores_tree.set_all(
                pore_ids.astype(np.int64), pcs.astype(np.float64)
            )

    def set_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Replace all border throats. Entry pressures are read from the referenced array."""
        self._throats_tree.clear()
        if len(throat_ids) > 0:
            pcs = self._throat_pc[throat_ids]
            self._throats_tree.set_all(
                throat_ids.astype(np.int64), pcs.astype(np.float64)
            )

    # ------------------------------------------------------------------ #
    # Incremental add / remove
    # ------------------------------------------------------------------ #

    def add_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Insert new border pores. Entry pressures are read from the referenced array."""
        for i in range(len(pore_ids)):
            pid = int(pore_ids[i])
            self._pores_tree.insert(pid, float(self._pore_pc[pid]))

    def add_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Insert new border throats. Entry pressures are read from the referenced array."""
        for i in range(len(throat_ids)):
            tid = int(throat_ids[i])
            self._throats_tree.insert(tid, float(self._throat_pc[tid]))

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove border pores by index."""
        for i in range(len(pore_ids)):
            self._pores_tree.remove(int(pore_ids[i]))

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove border throats by index."""
        for i in range(len(throat_ids)):
            self._throats_tree.remove(int(throat_ids[i]))

    # ------------------------------------------------------------------ #
    # Entry-pressure sync (after external array updates)
    # ------------------------------------------------------------------ #

    def sync_pore_pcs(self, pore_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures from the referenced array into the pore tree.

        If *pore_ids* is ``None``, syncs all border pores.
        """
        if pore_ids is None:
            pore_ids = self._pores_tree.get_all_ids()
        for pid_val in pore_ids:
            pid = int(pid_val)
            if self._pores_tree.contains(pid):
                self._pores_tree.update_pc(pid, float(self._pore_pc[pid]))

    def sync_throat_pcs(self, throat_ids: Optional[np.ndarray] = None) -> None:
        """Re-read entry pressures from the referenced array into the throat tree.

        If *throat_ids* is ``None``, syncs all border throats.
        """
        if throat_ids is None:
            throat_ids = self._throats_tree.get_all_ids()
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
        # Tie-break: prefer lower PC; if equal, prefer pore (consistent with
        # the original linear-scan order which checks pores first).
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
        return self._pores_tree.count

    @property
    def throat_count(self) -> int:
        """Number of border throats."""
        return self._throats_tree.count

    @property
    def is_empty(self) -> bool:
        """True if no border elements are stored."""
        return self._pores_tree.is_empty and self._throats_tree.is_empty
