import numpy as np


class BorderElements:
    """
    Tracks unfilled elements adjacent to the invasion front.

    Internally uses numpy arrays. Add and remove operations create new arrays
    via ``np.concatenate`` / ``np.delete``; these are O(border_size) but rely
    on highly-optimised C code and are faster in practice than Python-set or
    pre-allocated-buffer alternatives for the border sizes seen in typical
    pore-network simulations.
    """

    def __init__(self) -> None:
        self.pores_ids: np.ndarray = np.empty(0, np.int64)
        self.throats_ids: np.ndarray = np.empty(0, np.int64)

    # ------------------------------------------------------------------ #
    # Pore helpers
    # ------------------------------------------------------------------ #

    def set_pores_ids(self, pores_ids: np.ndarray) -> None:
        """Replace the current pore border with *pores_ids*."""
        self.pores_ids = pores_ids.copy()

    def get_pores_ids(self) -> np.ndarray:
        """Return the border-pore IDs as a numpy int64 array."""
        return self.pores_ids

    def add_pores_ids(self, pores_ids: np.ndarray) -> None:
        """Append *pores_ids* to the border-pore array."""
        self.pores_ids = np.concatenate((self.pores_ids, pores_ids))

    def remove_pores_ids(self, pore_ids: np.ndarray) -> None:
        """Remove *pore_ids* from the border-pore array."""
        self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, pore_ids))

    # ------------------------------------------------------------------ #
    # Throat helpers
    # ------------------------------------------------------------------ #

    def set_throats_ids(self, throats_ids: np.ndarray) -> None:
        """Replace the current throat border with *throats_ids*."""
        self.throats_ids = throats_ids.copy()

    def get_throats_ids(self) -> np.ndarray:
        """Return the border-throat IDs as a numpy int64 array."""
        return self.throats_ids

    def add_throats_ids(self, throats_ids: np.ndarray) -> None:
        """Append *throats_ids* to the border-throat array."""
        self.throats_ids = np.concatenate((self.throats_ids, throats_ids))

    def remove_throats_ids(self, throat_ids: np.ndarray) -> None:
        """Remove *throat_ids* from the border-throat array."""
        self.throats_ids = np.delete(self.throats_ids, np.isin(self.throats_ids, throat_ids))
