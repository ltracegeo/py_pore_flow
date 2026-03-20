"""Border element tracking using plain numpy arrays.

Border elements are unfilled pores/throats adjacent to filled elements and thus
candidates for invasion.  This module stores border element IDs in numpy arrays
with O(1) retrieval and standard numpy operations for add/remove.

Min/max entry-pressure extraction is handled externally by ``@njit`` scan
functions in each cycle class, which read the entry-pressure arrays directly —
a single-pass scan with zero maintenance overhead.
"""

import numpy as np


class BorderElements:
    """Tracks border pore and throat IDs using numpy arrays.

    Parameters
    ----------
    pore_entry_pressure : np.ndarray
        Reference to ``pn_core.pore.entry_pressure`` (unused, kept for API compatibility).
    throat_entry_pressure : np.ndarray
        Reference to ``pn_core.throat.entry_pressure`` (unused, kept for API compatibility).
    """

    def __init__(
        self,
        pore_entry_pressure: np.ndarray,
        throat_entry_pressure: np.ndarray,
    ) -> None:
        self._pore_ids: np.ndarray = np.array([], np.int64)
        self._throat_ids: np.ndarray = np.array([], np.int64)

    # ------------------------------------------------------------------ #
    # ID accessors
    # ------------------------------------------------------------------ #

    def get_pores_ids(self) -> np.ndarray:
        """Return the stored border pore indices array."""
        return self._pore_ids

    def get_throats_ids(self) -> np.ndarray:
        """Return the stored border throat indices array."""
        return self._throat_ids

    # ------------------------------------------------------------------ #
    # Bulk set
    # ------------------------------------------------------------------ #

    def set_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Replace all border pore IDs."""
        self._pore_ids = pore_ids

    def set_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Replace all border throat IDs."""
        self._throat_ids = throat_ids

    # ------------------------------------------------------------------ #
    # Incremental add / remove
    # ------------------------------------------------------------------ #

    def add_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Append pore IDs to the border set."""
        self._pore_ids = np.concatenate((self._pore_ids, pore_ids))

    def add_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Append throat IDs to the border set."""
        self._throat_ids = np.concatenate((self._throat_ids, throat_ids))

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove pore IDs from the border set."""
        self._pore_ids = np.delete(self._pore_ids, np.isin(self._pore_ids, pore_ids))

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove throat IDs from the border set."""
        self._throat_ids = np.delete(self._throat_ids, np.isin(self._throat_ids, throat_ids))

    # ------------------------------------------------------------------ #
    # Entry-pressure sync (no-op — kept for API compatibility)
    # ------------------------------------------------------------------ #

    def sync_pore_pcs(self, pore_ids=None) -> None:
        """No-op.  Plain arrays read entry pressures directly during scans."""
        pass

    def sync_throat_pcs(self, throat_ids=None) -> None:
        """No-op.  Plain arrays read entry pressures directly during scans."""
        pass

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

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
