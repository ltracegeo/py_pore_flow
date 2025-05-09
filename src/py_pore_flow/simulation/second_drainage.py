import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    OIL,
    check_water_layer,
    calculate_2nd_drainage_element_b,
    calculate_imbibition_network_b,
    calculate_2nd_drainage_network_b,
    calculate_2nd_drainage_network_bo,
    calculate_imbibition_network_bo,
    calculate_network_hinging_contact_angle,
    calculate_network_outter_hinging_contact_angle,
    calculate_oil_layer,
    calculate_r,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import calculate_imbibition_pc, PORE_BODY_FILLING_STRATEGY
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log
from py_pore_flow.simulation.algorithms.sorted_root_tree import SortedRootTree


class SecondDrainageSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, pc_max, OIL, calculate_full_conductance)

        self.adversary_fluid_filled = network.pn_core.throat.water_filled
        self.limit_contact_angle = network.pn_core.pore.receding_contact_angle

        if c.pore_fill_alg == "oren2":
            self.calculate_pore_body_filling_pc = PORE_BODY_FILLING_STRATEGY["oren1"]

        self.initialize()

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_throats_snap_offs_pcs(self, pc):
        throat_snap_off_viability = np.empty(len(self.network.pn.throats()), np.bool_)
        self.__set_snap_off_viability(
            throat_snap_off_viability,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.n_conns,
        )
        return self._snap_off(
            self.network.pn.throats(),
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.conns,
            self.network.pn_core.throat.n_conns,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.radius,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            throat_snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.pore.oil_filled,
            self.Rmin,
            pc,
            is_throats=True,
        )

    def _update_pores_snap_offs_pcs(self, pc):
        pore_snap_off_viability = np.empty(len(self.network.pn.pores()), np.bool_)
        self.__set_snap_off_viability(
            pore_snap_off_viability,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_conns,
        )
        return self._snap_off(
            self.network.pn.pores(),
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            pore_snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.throat.oil_filled,
            self.Rmin,
            pc,
            is_throats=False,
        )

    @staticmethod
    @nb.njit(cache=True)
    def __set_snap_off_viability(snap_off_viability, corner_oil_layer, n_conns):
        for i in range(len(snap_off_viability)):
            snap_off_viability[i] = corner_oil_layer[i][: n_conns[i]].any()

    @staticmethod
    # @nb.njit(cache=True)
    def _jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        throat_id_list,
        throat_entry_pressure,
    ):
        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        sorted_root_tree = SortedRootTree(throat_entry_pressure, pore_entry_pressure)
        for pore_id in pore_id_list:
            sorted_root_tree.add_node(pore_entry_pressure[pore_id], pore_id + 1)
        for throat_id in throat_id_list:
            sorted_root_tree.add_node(throat_entry_pressure[throat_id], throat_id + sorted_root_tree.pore_n + 1)
        root_label = sorted_root_tree.get_root()
        if root_label != 0:
            if root_label > sorted_root_tree.pore_n:
                min_id = root_label - sorted_root_tree.pore_n - 1
                min_pressure = throat_entry_pressure[min_id]
                min_type = THROAT
            else:
                min_id = root_label - 1
                min_pressure = pore_entry_pressure[min_id]
                min_type = PORE
        else:
            min_pressure = -np.inf
            min_id = -1
            min_type = -1

        return (min_type, min_id, max(min_pressure, pc))

    def _update_element_filled_geometry(self, pc, element):
        element.corner_oil_layer = False
        element.bo = 0.0
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
        self._update_element_b(element.type_name, element.id, pc)
        return super()._update_element_filled_geometry(pc, element)

    def _update_b(self, pc):
        calculate_2nd_drainage_network_b(
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_water_layer,
            self.network.pn_core.pore.n_corners,
            pc,
            self.Rmin,
        )
        calculate_2nd_drainage_network_b(
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_water_layer,
            self.network.pn_core.throat.n_corners,
            pc,
            self.Rmin,
        )

        calculate_imbibition_network_bo(
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_corners,
            pc,
        )
        calculate_imbibition_network_bo(
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.n_corners,
            pc,
        )

    def _update_element_b(self, type_name, id, pc):
        if type_name == "pore":
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.b[id] = calculate_2nd_drainage_element_b(
            element.b[id],
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            element.hinging_contact_angle[id],
            element.corner_water_layer[id],
            element.n_corners[id],
            self.Rmin,
            pc,
        )

    def _update_b(self, pc):
        calculate_imbibition_network_b(
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_water_layer,
            self.network.pn_core.pore.n_corners,
            pc,
            self.Rmin,
        )
        calculate_imbibition_network_b(
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_water_layer,
            self.network.pn_core.throat.n_corners,
            pc,
            self.Rmin,
        )

        calculate_2nd_drainage_network_bo(
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle_o,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_corners,
            pc,
        )
        calculate_2nd_drainage_network_bo(
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle_o,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.n_corners,
            pc,
        )

    def _update_hinging_contact_angles(self, pc):
        calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.trapped_evacuation,
        )
        calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.trapped_evacuation,
        )

        calculate_network_outter_hinging_contact_angle(
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle_o,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.max_b,
        )
        calculate_network_outter_hinging_contact_angle(
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle_o,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.max_b,
        )
