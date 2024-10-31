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
    calculate_oil_layer_network,
    calculate_r,
    check_water_layer_network,
)
from py_pore_flow.equations.entry_pressure import (
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
)
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation
from py_pore_flow import log


class ImbibitionSimulation(CycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, WATER, calculate_full_conductance)
        self.Rmin = calculate_r(pc_max, 100000)

        self.snap_off_throat_list = []
        self.snap_off_pore_list = []

    def get_pores_left(self):
        return np.logical_not(self.network.pn_core.pore.water_filled | self.network.pn_core.pore.trapped).sum()

    def _update_invasions(self, pc, target_sw=None):
        invaded_pc = super()._update_invasions(pc, target_sw)
        return invaded_pc

    @staticmethod
    @nb.njit
    def __snap_off(
        element_id_list,
        element_n_corners,
        element_conns,
        element_n_conns,
        element_water_filled,
        element_oil_filled,
        element_shape,
        element_r,
        element_beta,
        element_theta_a,
        element_theta_r,
        element_theta_h,
        element_snap_off_viability,
        entry_pressure,
        element_snap_off_pc,
        element_trapped,
        neighbor_water_filled,
        Rmin,
        pc,
        border_elements,
        is_throats=True,
    ):
        snap_off_elements = []
        for element_id in element_id_list:
            if (
                element_snap_off_viability[element_id]
                and not element_water_filled[element_id]
                and element_shape[element_id] != CIRCLE
                and not element_trapped[element_id]
            ):
                if False:  # Which has best performance?
                    if is_throats:
                        if element_id in border_elements.get_throats_ids():
                            continue
                    else:
                        if element_id in border_elements.get_pores_ids():
                            continue
                else:
                    for i in range(element_n_conns[element_id]):
                        pore_id = element_conns[element_id][i]
                        if neighbor_water_filled[pore_id]:
                            continue

                entry_pressure[element_id] = element_snap_off_pc[element_id]
                snap_off_elements.append(element_id)

        return np.array(snap_off_elements, dtype=np.int64)

    def __update_snap_offs_pcs(self, pc):
        self.snap_off_throat_list = self.__snap_off(
            self.network.pn.throats(),
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.conns,
            self.network.pn_core.throat.n_conns,
            self.network.pn_core.throat.water_filled,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.radius,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped,
            self.network.pn_core.pore.water_filled,
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=True,
        )

        self.snap_off_pore_list = self.__snap_off(
            self.network.pn.pores(),
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            self.network.pn_core.pore.water_filled,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped,
            self.network.pn_core.throat.water_filled,
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=False,
        )

    def __update_oil_layers(self, pc):
        calculate_oil_layer_network(
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            pc,
        )

        calculate_oil_layer_network(
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            pc,
        )

    def __update_water_layers(self):
        check_water_layer_network(
            self.network.pn_core.throat.corner_water_layer,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.oil_filled,
        )
        check_water_layer_network(
            self.network.pn_core.pore.corner_water_layer,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.oil_filled,
        )

    def _update_entry_pressures(self, pc):
        self.__update_snap_offs_pcs(pc)
        super()._update_entry_pressures(pc)

    def _update_element_entry_pc(self, element, pc):
        if element.is_pore():
            self._update_pore_entry_pc(element.get_id(), pc)
        else:
            self._update_throat_entry_pc(element.get_id(), pc)

    def _update_pore_entry_pc(self, pore_id, pc):
        K = self.krel_calculator.get_absolute_permeability()
        number_of_oil_filled_neighbors = self.network.pn_core.throat.oil_filled[
            self.network.get_neighbor_throats_ids(pore_id)
        ].sum()
        element_pc = calculate_imbibition_pc(
            self.network.pn_core.pore.shape_factor[pore_id],
            self.network.pn_core.pore.radius[pore_id],
            self.network.pn_core.pore.advancing_contact_angle[pore_id],
            self.network.pn_core.pore.shape[pore_id],
            pc,
            self.network.pn_core.pore.corner_half_angle[pore_id],
            self.network.pn_core.pore.n_corners[pore_id],
            number_of_oil_filled_neighbors,
            K,
        )
        self.network.pn_core.pore.entry_pressure[pore_id] = element_pc

    def _update_throat_entry_pc(self, throat_id, pc):
        element_pc = calculate_imbibition_piston_like_displacement_pc(
            self.network.pn_core.throat.shape_factor[throat_id],
            self.network.pn_core.throat.radius[throat_id],
            self.network.pn_core.throat.advancing_contact_angle[throat_id],
            self.network.pn_core.throat.shape[throat_id],
            pc,
            self.network.pn_core.throat.corner_half_angle[throat_id],
            self.network.pn_core.throat.n_corners[throat_id],
        )
        self.network.pn_core.throat.entry_pressure[throat_id] = element_pc

    def _get_element_to_fill(self, pc, border_elements):
        self.__clear_snapoff_lists()
        pore_id_list = np.concatenate((border_elements.get_pores_ids(), self.snap_off_pore_list))
        throat_id_list = np.concatenate((border_elements.get_throats_ids(), self.snap_off_throat_list))
        elements_to_fill = self._jit_pore_throats_element_to_fill(
            pc,
            pore_id_list,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.spontaneous_imbibition,
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.spontaneous_imbibition,
        )

        if len(elements_to_fill) > 0:
            element_to_fill = elements_to_fill[0]
            return (element_to_fill[0], element_to_fill[1], min(element_to_fill[2], pc))
        else:
            return None

    @staticmethod
    @nb.njit
    def _jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        pore_spontaneous_imbibition,
        throat_id_list,
        throat_entry_pressure,
        throat_spontaneous_imbibition,
    ):
        if pc < 0:
            tuple_list = []
            for id in pore_id_list:
                tuple_list.append((PORE, id, pore_entry_pressure[id]))
            for id in throat_id_list:
                tuple_list.append((THROAT, id, throat_entry_pressure[id]))
            tuple_list.sort(key=lambda element: element[2], reverse=True)
        else:
            tuple_list = []
            for id in pore_id_list:
                if pore_spontaneous_imbibition[id]:
                    tuple_list.append((PORE, id, pore_entry_pressure[id]))
            for id in throat_id_list:
                if throat_spontaneous_imbibition[id]:
                    tuple_list.append((THROAT, id, throat_entry_pressure[id]))
            tuple_list.sort(key=lambda element: element[2], reverse=True)

        return tuple_list

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc <= invasion_pc

    def __update_b(self, pc):
        calculate_imbibition_network_b(
            self.network.pn_core.pore.b,
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
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_corners,
            pc,
        )
        calculate_imbibition_network_bo(
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.n_corners,
            pc,
        )

    def _update_element_filled_geometry(self, pc, element):
        element.corner_oil_layer = calculate_oil_layer(
            element.advancing_contact_angle, element.beta, element.n_corners, pc
        )
        self.__update_element_b(element.type_name, element.id, pc)

        return super()._update_element_filled_geometry(pc, element)

    def _update_network_geometry(self, pc):
        self.__update_b(pc)
        self.__update_oil_layers(pc)
        self.__update_water_layers()
        self.__update_hinging_contact_angles(pc)
        super()._update_network_geometry(pc)

    def __update_element_b(self, type_name, id, pc):
        if type_name == "pore":
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.b[id] = calculate_imbibition_element_b(
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            element.hinging_contact_angle[id],
            element.corner_water_layer[id],
            element.n_corners[id],
            pc,
            self.Rmin,
        )
        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
        )

    def __clear_snapoff_lists(self):
        self.snap_off_pore_list = np.delete(
            self.snap_off_pore_list,
            self.network.pn_core.pore.water_filled[self.snap_off_pore_list]
            | self.network.pn_core.pore.trapped[self.snap_off_pore_list],
        )
        self.snap_off_throat_list = np.delete(
            self.snap_off_throat_list,
            self.network.pn_core.throat.water_filled[self.snap_off_throat_list]
            | self.network.pn_core.throat.trapped[self.snap_off_throat_list],
        )

    def __update_hinging_contact_angles(self, pc):
        calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
        )
        calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
        )
