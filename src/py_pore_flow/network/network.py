import numpy as np

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


def initialize_pn(pn):
    def set_default(pn, key, value):
        if key not in pn:
            pn[key] = value

    for element_type in ["pore", "throat"]:
        number_of_elements = len(pn[f"{element_type}.all"])

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
        if f"{element_type}.N" not in pn:
            pn[f"{element_type}.N"] = 1
        pn[f"{element_type}.changed_wettability"] = False
        pn[f"{element_type}.imminent_snap_off"] = False
        pn[f"{element_type}.in_border"] = False

    pn["throat.oil_singlephase_conductance"] = 0.0
    pn["throat.water_singlephase_conductance"] = 0.0
    pn["throat.oil_multiphase_conductance"] = 0.0
    pn["throat.water_multiphase_conductance"] = 0.0
    pn["throat.hydraulic_conductance"] = 0.0

    # Initialize connection properties
    pn["throat.n_conns"] = int(2)
    pn["pore.n_conns"] = int(0)

    number_of_pores = len(pn["pore.all"])

    pore_n_conns_list = [0] * number_of_pores
    pore_conns_list = [None] * number_of_pores
    for pore_id in pn.pores():
        neighbor_throats_list = pn.find_neighbor_throats(pore_id)
        pore_n_conns_list[pore_id] = len(neighbor_throats_list)
        pore_conns_list[pore_id] = neighbor_throats_list

    max_n_conns = max(pore_n_conns_list)
    pn["pore.conns"] = np.empty((number_of_pores, max_n_conns), np.int64)
    for pore_id in pn.pores():
        pn["pore.n_conns"][pore_id] = pore_n_conns_list[pore_id]
        pn["pore.conns"][pore_id] = np.resize(pore_conns_list[pore_id], max_n_conns)

    # Required to maintain compatibility between windows and linux:
    pn["throat.conns"] = pn["throat.conns"].astype(np.int64)
