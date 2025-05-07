import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    WATER,
    calculate_imbibition_element_b,
    calculate_imbibition_element_bo,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_network_hinging_contact_angle,
    calculate_oil_layer,
    calculate_r,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import calculate_imbibition_pc, PORE_BODY_FILLING_STRATEGY
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log
from py_pore_flow.simulation.algorithms.sorted_root_tree import DecreasingSortedRootTree


class ImbibitionSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, pc_max, WATER, calculate_full_conductance)
        self.adversary_fluid_filled = network.pn_core.throat.oil_filled
        self.limit_contact_angle = network.pn_core.pore.advancing_contact_angle

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc <= invasion_pc

    def _update_snap_offs_pcs(self, pc):
        self.snap_off_throat_list = self._snap_off(
            self.network.pn.throats(),
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.conns,
            self.network.pn_core.throat.n_conns,
            self.network.pn_core.throat.water_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.radius,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.pore.water_filled,
            self.Rmin,
            pc,
            is_throats=True,
        )

        self.snap_off_pore_list = self._snap_off(
            self.network.pn.pores(),
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            self.network.pn_core.pore.water_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.throat.water_filled,
            self.Rmin,
            pc,
            is_throats=False,
        )

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

        sorted_root_tree = DecreasingSortedRootTree(throat_entry_pressure, pore_entry_pressure)
        for pore_id in pore_id_list:
            sorted_root_tree.add_node(pore_entry_pressure[pore_id], pore_id + 1)
        for throat_id in throat_id_list:
            sorted_root_tree.add_node(throat_entry_pressure[throat_id], throat_id + sorted_root_tree.pore_n + 1)
        root_label = sorted_root_tree.get_root()
        if root_label != 0:
            if root_label > sorted_root_tree.pore_n:
                max_id = root_label - sorted_root_tree.pore_n - 1
                max_pressure = throat_entry_pressure[max_id]
                max_type = THROAT
            else:
                max_id = root_label - 1
                max_pressure = pore_entry_pressure[max_id]
                max_type = PORE
        else:
            max_pressure = -np.inf
            max_id = -1
            max_type = -1

        return (max_type, max_id, min(max_pressure, pc))

    def _update_element_filled_geometry(self, pc, element):
        element.corner_oil_layer = calculate_oil_layer(
            element.advancing_contact_angle, element.beta, element.n_corners, pc
        )
        self._update_element_b(element.type_name, element.id, pc)

        return super()._update_element_filled_geometry(pc, element)
