import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    WATER,
    calculate_hinging_contact_angle,
    calculate_imbibition_element_b,
    calculate_imbibition_element_bo,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_network_hinging_contact_angle,
    calculate_oil_layer,
    calculate_oil_layer_element,
    calculate_r,
    check_water_layer,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import PORE_BODY_FILLING_STRATEGY
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log


class ImbibitionSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, pc_max, WATER, calculate_full_conductance)
        self.adversary_fluid_filled = network.pn_core.throat.oil_filled
        self.limit_contact_angle = network.pn_core.pore.advancing_contact_angle

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc <= invasion_pc

    def _update_snap_offs_pcs(self, pc):
        self.snap_off_throat_list = self._snap_off(
            self.network.pn_core.throat.imminent_snap_off,
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
            self.network.pn_core.pore.imminent_snap_off,
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
    @nb.njit(cache=True)
    def _jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        throat_id_list,
        throat_entry_pressure,
    ):
        max_pressure = -np.inf
        max_id = -1
        max_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            pressure = pore_entry_pressure[id]
            if pressure > max_pressure:
                max_pressure = pressure
                max_id = id
                max_type = PORE

        for id in throat_id_list:
            pressure = throat_entry_pressure[id]
            if pressure > max_pressure:
                max_pressure = pressure
                max_id = id
                max_type = THROAT

        return (max_type, max_id, min(max_pressure, pc))

    def _update_element_filled_geometry(self, pc, element):
        self.__update_element_pinned_hinging_contact_angles(element.type, element.id, pc)
        self.__update_element_pinned_b(element.type, element.id, pc)
        element.corner_oil_layer = calculate_oil_layer(
            element.advancing_contact_angle, element.beta, element.n_corners, pc
        )
        self.__update_oil_layer_element(element.type, element.id, pc)

        self.__update_element_bo(element.type, element.id, pc)
        self.__check_water_layer(element.type, element.id)
        self._update_element_hinging_contact_angles(element.type, element.id, pc)
        self.__update_element_b(element.type, element.id, pc)

        return super()._update_element_filled_geometry(pc, element)

    def __update_element_pinned_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        n_corners = element.n_corners[id]
        element.b[id] = calculate_imbibition_element_b(
            element.b[id],
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            element.hinging_contact_angle[id],
            np.full(n_corners, True, np.bool_),
            n_corners,
            self.Rmin,
            pc,
        )

    def __update_element_bo(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
        )

    def __update_oil_layer_element(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        calculate_oil_layer_element(
            element.corner_oil_layer[id],
            element.oil_layer_collapse_pressure[id],
            element.b[id],
            element.n_corners[id],
            pc,
        )

    def __check_water_layer(self, element_type, id):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.corner_water_layer[id] = check_water_layer(
            element.corner_oil_layer[id],
            element.oil_filled[id],
        )

    def __update_element_pinned_hinging_contact_angles(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat
        calculate_hinging_contact_angle(
            pc,
            element_data.corner_half_angle[id],
            element_data.b_pinned[id],
            element_data.n_corners[id],
            element_data.receding_contact_angle[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
        )

    def __update_element_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.b[id] = calculate_imbibition_element_b(
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
        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
        )
