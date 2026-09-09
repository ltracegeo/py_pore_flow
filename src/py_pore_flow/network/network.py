from numba import njit
import numpy as np
import openpnm

from py_pore_flow.equations.geometry import (
    MAX_NUMBER_OF_CORNERS,
)
from .constants import PORE, THROAT
from .pn_element import PnElement
from .pores_manager import PoresManager
from .throats_manager import ThroatsManager
from .element_network import ElementNetwork
from .pores_network import PoresNetwork
from .throats_network import ThroatsNetwork
from .network_core import NetworkCore
from .network_manager import NetworkManager


class PoreNetwork(openpnm.network.Network):
    """Local wrapper around OpenPNM's Network class.

    Stores pore connections in a sparse (CSR-like) format instead of a dense
    2-D array, since max_n_conns can be extremely large due to outlier pores.

    The sparse representation uses three 1-D arrays stored directly on this
    instance:
        _conns_values  – concatenated connection (throat) ids for every pore
        _conns_offsets – start index in _conns_values for each pore
        _conns_counts  – number of connections for each pore

    Access patterns:
        pn["pore.conns"][pore_id]  →  1-D array of throat ids for that pore
        pn["pore.conns"]           →  raises RuntimeError (use per-pore access)
    """

    _conns_values = None
    _conns_offsets = None

    # ------------------------------------------------------------------
    # Thin proxy returned by  pn["pore.conns"]  so that  [...][pore_id]
    # can resolve to the sparse slice without exposing the dense array.
    # ------------------------------------------------------------------
    class _ConnsAccessor:
        __slots__ = ("_pn",)

        def __init__(self, pn: "PoreNetwork"):
            self._pn = pn

        @property
        def values(self):
            return self._pn._conns_values

        @property
        def offsets(self):
            return self._pn._conns_offsets

        def __getitem__(self, pore_id):
            start = self._pn._conns_offsets[pore_id]
            end = self._pn._conns_offsets[pore_id + 1]
            return self._pn._conns_values[start:end]

        # Guard-rails: blow up on any attempt to treat this as a dense array
        def __array__(self, *a, **kw):
            raise RuntimeError(
                "pore.conns is stored as a sparse structure.  "
                "Index with pn['pore.conns'][pore_id] instead of converting to a dense array."
            )

        def __iter__(self):
            raise RuntimeError(
                "pore.conns is stored as a sparse structure.  "
                "Index with pn['pore.conns'][pore_id] instead of iterating."
            )

        def __len__(self):
            return len(self._pn._conns_offsets)

    def init_sparse_conns(self, values: np.ndarray, offsets: np.ndarray):
        self._conns_values = values
        self._conns_offsets = offsets

    def __getitem__(self, key):
        if key == "pore.conns":
            if self._conns_values is None:
                raise RuntimeError("Sparse pore.conns has not been initialised yet.")
            return self._ConnsAccessor(self)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key == "pore.conns":
            raise RuntimeError(
                "Direct assignment to pn['pore.conns'] is not supported.  "
                "Use pn.init_sparse_conns(values, offsets, counts) instead."
            )
        super().__setitem__(key, value)

    @classmethod
    def from_openpnm(cls, network: openpnm.network.Network) -> "PoreNetwork":
        """Create a PoreNetwork from a plain OpenPNM Network instance."""
        pore_network = cls()
        pore_network.update(network)
        return pore_network


def initialize_pn(pn: PoreNetwork) -> None:
    def set_default(pn, key, value):
        if key not in pn:
            pn[key] = value

    for element_type in ["pore", "throat"]:
        number_of_elements = len(pn[f"{element_type}.all"])

        set_default(pn, f"{element_type}.filled", False)
        set_default(pn, f"{element_type}.oil_filled", False)
        set_default(pn, f"{element_type}.water_filled", True)
        set_default(pn, f"{element_type}.oil_saturation", 0.0)
        set_default(pn, f"{element_type}.water_saturation", 1.0)
        pn[f"{element_type}.corner_half_angle"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.sides"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.hinging_contact_angle"] = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
        pn[f"{element_type}.hinging_contact_angle_o"] = np.empty(
            (number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64
        )
        pn[f"{element_type}.b"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.bo"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.b_pinned"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), 0.0, np.float64)
        pn[f"{element_type}.water_filled_multiphase"] = False
        pn[f"{element_type}.oil_filled_multiphase"] = False
        pn[f"{element_type}.max_b"] = 0.0
        pn[f"{element_type}.corner_water_layer"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
        pn[f"{element_type}.corner_oil_layer"] = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
        pn[f"{element_type}.C_r"] = 0.0
        pn[f"{element_type}.C_a"] = 0.0
        pn[f"{element_type}.C_h"] = 0.0
        pn[f"{element_type}.advancing_contact_angle"] = 0.0
        pn[f"{element_type}.receding_contact_angle"] = 0.0
        pn[f"{element_type}.oil_singlephase_conductivity"] = 0.0
        pn[f"{element_type}.water_singlephase_conductivity"] = 0.0
        pn[f"{element_type}.oil_multiphase_conductivity"] = 0.0
        pn[f"{element_type}.water_multiphase_conductivity"] = 0.0
        pn[f"{element_type}.shape"] = 0
        pn[f"{element_type}.trapped"] = False
        pn[f"{element_type}.trapped_evacuation"] = False
        pn[f"{element_type}.n_corners"] = int(0)
        pn[f"{element_type}.snap_off_viability"] = True
        pn[f"{element_type}.entry_pressure"] = 0.0
        pn[f"{element_type}.invasion_method"] = int(0)
        pn[f"{element_type}.snap_off_pressure"] = 0.0
        pn[f"{element_type}.piston_like_pressure"] = 0.0
        pn[f"{element_type}.oil_layer_collapse_pressure"] = np.empty(
            (number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64
        )
        pn[f"{element_type}.water_corner_collapse_pressure"] = np.empty(
            (number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64
        )
        if f"{element_type}.initial_advancing_ca" not in pn:
            pn[f"{element_type}.initial_advancing_ca"] = 0.0
        if f"{element_type}.initial_receding_ca" not in pn:
            pn[f"{element_type}.initial_receding_ca"] = 0.0
        if f"{element_type}.equilibrium_advancing_ca" not in pn:
            pn[f"{element_type}.equilibrium_advancing_ca"] = 0.0
        if f"{element_type}.equilibrium_receding_ca" not in pn:
            pn[f"{element_type}.equilibrium_receding_ca"] = 0.0
        if f"{element_type}.second_advancing_ca" not in pn:
            pn[f"{element_type}.second_advancing_ca"] = 0.0
        if f"{element_type}.second_receding_ca" not in pn:
            pn[f"{element_type}.second_receding_ca"] = 0.0
        pn[f"{element_type}.oil_connected_to_inlet"] = False
        pn[f"{element_type}.oil_connected_to_outlet"] = False
        pn[f"{element_type}.water_connected_to_inlet"] = True
        pn[f"{element_type}.water_connected_to_outlet"] = True

        pn[f"{element_type}.is_multiphase"] = (pn[f"{element_type}.water_saturation"] > 0) & (
            pn[f"{element_type}.water_saturation"] < 1.0
        )

        pn[f"{element_type}.id"] = list(range(len(pn[f"{element_type}.all"])))
        if f"{element_type}.number_of_capilaries" not in pn:
            pn[f"{element_type}.number_of_capilaries"] = 1
        pn[f"{element_type}.changed_wettability"] = False
        pn[f"{element_type}.imminent_snap_off"] = False
        pn[f"{element_type}.in_border"] = False

    pn["throat.oil_singlephase_conductance"] = 0.0
    pn["throat.water_singlephase_conductance"] = 0.0
    pn["throat.oil_multiphase_conductance"] = 0.0
    pn["throat.water_multiphase_conductance"] = 0.0
    pn["throat.hydraulic_conductance"] = 0.0

    pn["pore.flow_rate"] = -1.0

    # Initialize connection properties
    pn["throat.n_conns"] = np.int64(2)
    pn["pore.n_conns"] = int(0)

    number_of_pores = len(pn["pore.all"])
    offsets = np.zeros(number_of_pores + 1, dtype=np.int64)
    _update_counts(offsets, pn["throat.conns"])
    pn["pore.n_conns"] = offsets[1:].copy()
    total_conns = int(offsets.sum())
    _make_cummulative(offsets)
    values = np.empty(total_conns, dtype=np.int64)
    _fill_values(values, offsets, pn["throat.conns"])

    pn.init_sparse_conns(values, offsets)

    # Build sparse representation for throat.conns (uniform 2 conns per throat)
    number_of_throats = len(pn["throat.all"])
    pn._throat_conns_values = pn["throat.conns"].ravel().astype(np.int64)
    pn._throat_conns_offsets = np.arange(0, 2 * number_of_throats, 2, dtype=np.int64)

@njit(cache=True)
def _update_counts(counts, throat_conns):
    for throat_id in range(len(throat_conns)):
        counts[throat_conns[throat_id][0] + 1] += 1
        counts[throat_conns[throat_id][1] + 1] += 1

@njit(cache=True)
def _make_cummulative(counts):
    for i in range(1, len(counts)):
        counts[i] += counts[i-1]

@njit(cache=True)
def _fill_values(values, offsets, throat_conns):
    filled_values = np.zeros(len(offsets)-1, dtype=np.int64)
    for throat_id in range(len(throat_conns)):
        pore1, pore2 = throat_conns[throat_id]
        index1 = offsets[pore1] + filled_values[pore1]
        index2 = offsets[pore2] + filled_values[pore2]
        values[index1] = throat_id
        values[index2] = throat_id
        filled_values[pore1] += 1
        filled_values[pore2] += 1
