import numpy as np


class BorderElements:
    """
    Tracks unfilled elements adjacent to the invasion front.

    Internally uses Python sets for O(1) insertion and O(1) removal.
    The numpy arrays returned by :meth:`get_pores_ids` /
    :meth:`get_throats_ids` are rebuilt lazily whenever the underlying set
    has been modified, avoiding the repeated O(border_size) allocations
    that ``np.concatenate`` / ``np.delete`` would incur on every invasion.

    Performance note
    ----------------
    Each invasion typically adds 1–5 border neighbours and removes 1–3
    trapped / just-filled elements.  With the old numpy-array backend
    those operations were O(border_size) due to array copies; with the
    set backend they are O(added) and O(removed) respectively.  The
    :meth:`get_*_ids` rebuild is O(border_size), but it is deferred until
    the array is actually needed (lazy evaluation) and the cost is
    comparable to a single ``np.concatenate`` that it replaces.
    """

    def __init__(self) -> None:
        self._pores_set: set[int] = set()
        self._throats_set: set[int] = set()

        # Cached numpy arrays; rebuilt on the next get_* call after any mutation.
        self._pores_array: np.ndarray = np.empty(0, np.int64)
        self._throats_array: np.ndarray = np.empty(0, np.int64)
        self._pores_dirty: bool = False
        self._throats_dirty: bool = False

    # ------------------------------------------------------------------ #
    # Pore helpers
    # ------------------------------------------------------------------ #

    def set_pores_ids(self, pores_ids: np.ndarray) -> None:
        """Replace the current pore set with *pores_ids*."""
        self._pores_set = set(pores_ids.tolist())
        self._pores_dirty = True

    def get_pores_ids(self) -> np.ndarray:
        """Return the border-pore IDs as a numpy int64 array.

        The array is rebuilt from the internal set only when the set has
        been modified since the last call (lazy rebuild).
        """
        if self._pores_dirty:
            self._pores_array = (
                np.array(list(self._pores_set), np.int64)
                if self._pores_set
                else np.empty(0, np.int64)
            )
            self._pores_dirty = False
        return self._pores_array

    def add_pores_ids(self, pores_ids: np.ndarray) -> None:
        """Add *pores_ids* to the border-pore set. O(|pores_ids|)."""
        if len(pores_ids):
            self._pores_set.update(pores_ids.tolist())
            self._pores_dirty = True

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove *pore_ids* from the border-pore set. O(|pore_ids|)."""
        if len(pore_ids):
            self._pores_set.difference_update(pore_ids.tolist())
            self._pores_dirty = True

    # ------------------------------------------------------------------ #
    # Throat helpers
    # ------------------------------------------------------------------ #

    def set_throats_ids(self, throats_ids: np.ndarray) -> None:
        """Replace the current throat set with *throats_ids*."""
        self._throats_set = set(throats_ids.tolist())
        self._throats_dirty = True

    def get_throats_ids(self) -> np.ndarray:
        """Return the border-throat IDs as a numpy int64 array.

        Rebuilt lazily when the set has been modified since the last call.
        """
        if self._throats_dirty:
            self._throats_array = (
                np.array(list(self._throats_set), np.int64)
                if self._throats_set
                else np.empty(0, np.int64)
            )
            self._throats_dirty = False
        return self._throats_array

    def add_throats_ids(self, throats_ids: np.ndarray) -> None:
        """Add *throats_ids* to the border-throat set. O(|throats_ids|)."""
        if len(throats_ids):
            self._throats_set.update(throats_ids.tolist())
            self._throats_dirty = True

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove *throat_ids* from the border-throat set. O(|throat_ids|)."""
        if len(throat_ids):
            self._throats_set.difference_update(throat_ids.tolist())
            self._throats_dirty = True
