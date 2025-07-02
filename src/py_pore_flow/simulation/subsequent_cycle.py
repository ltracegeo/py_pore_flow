from abc import abstractmethod

import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    OIL,
    WATER,
    calculate_imbibition_element_b,
    calculate_imbibition_element_bo,
    calculate_imbibition_element_saturation,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_imbibition_network_saturation,
    calculate_network_hinging_contact_angle,
    calculate_oil_layer,
    calculate_oil_layer_network,
    calculate_r,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import PORE_BODY_FILLING_STRATEGY, calculate_pore_body_filling_pc
from py_pore_flow.network import PORE, THROAT
from py_pore_flow.simulation.cycle_simulation import CycleSimulation, SaturationCalculator
from py_pore_flow import log


class SubsequentCycleSimulation(CycleSimulation):
    def __init__(self, network, pc_max, phase, calculate_full_conductance=False):
        super().__init__(network, phase, calculate_full_conductance)
        self.saturation_calculator = SubsequentSaturation(network, self.phase)

        self.Rmin = calculate_r(pc_max, 100000)

        self.use_pore_body_filling = c.use_pore_fill
        self.pore_body_filling_strategy = PORE_BODY_FILLING_STRATEGY[c.pore_fill_alg]

        self.snap_off_throat_list = []
        self.snap_off_pore_list = []

    @staticmethod
    @nb.njit(cache=True)
    def _njit_snap_off(
        element_imminent_snap_off,
        element_id_list,
        element_filled,
        element_shape,
        element_snap_off_viability,
        entry_pressure,
        element_snap_off_pc,
        element_trapped,
    ):
        snap_off_elements = []
        for element_id in element_id_list:
            if (
                element_snap_off_viability[element_id]
                and not element_filled[element_id]
                and element_shape[element_id] != CIRCLE
                and not element_trapped[element_id]
            ):
                entry_pressure[element_id] = element_snap_off_pc[element_id]
                snap_off_elements.append(element_id)
                element_imminent_snap_off[element_id] = True
            else:
                element_imminent_snap_off[element_id] = False

        return np.array(snap_off_elements, dtype=np.int64)

    def _update_throats_snap_offs_pcs(self, throat_ids):
        throat_filled = (
            self.network.pn_core.throat.water_filled if self.phase == WATER else self.network.pn_core.throat.oil_filled
        )
        return self._njit_snap_off(
            self.network.pn_core.throat.imminent_snap_off,
            throat_ids,
            throat_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
        )

    def _update_pores_snap_offs_pcs(self, pore_ids):
        pore_filled = (
            self.network.pn_core.pore.water_filled if self.phase == WATER else self.network.pn_core.pore.oil_filled
        )
        return self._njit_snap_off(
            self.network.pn_core.pore.imminent_snap_off,
            pore_ids,
            pore_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
        )

    def _update_oil_layers(self, pc):
        calculate_oil_layer_network(
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.oil_layer_collapse_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            pc,
        )

        calculate_oil_layer_network(
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.oil_layer_collapse_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            pc,
        )

    def __update_water_layers(self):
        check_water_layer_network(
            self.network.pn_core.throat.corner_water_layer,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.oil_filled,
        )
        check_water_layer_network(
            self.network.pn_core.pore.corner_water_layer,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.oil_filled,
        )

    def _update_entry_pressures(self, pc):
        self._update_throat_entry_pc(self.border_elements.get_throats_ids(), True)
        self._update_pore_entry_pc(self.border_elements.get_pores_ids(), pc, True)

    def _update_pore_entry_pc(self, pore_id_list, pc, replace_snap_off=False):
        if replace_snap_off:
            self.snap_off_pore_list = self._update_pores_snap_offs_pcs(self.network.pn.pores())
        else:
            new_snap_off_pores = self._update_pores_snap_offs_pcs(pore_id_list)
            if len(new_snap_off_pores) > 0:
                self.snap_off_pore_list = np.unique(np.concatenate((self.snap_off_pore_list, new_snap_off_pores)))

        self._update_border_pore_entry_pc(pore_id_list, pc)

    def _update_border_pore_entry_pc(self, pore_id_list, pc):
        pore_limiting_contact_angle = (
            self.network.pn_core.pore.advancing_contact_angle
            if self.phase == WATER
            else self.network.pn_core.pore.receding_contact_angle
        )
        throat_receding_phase_filled = (
            self.network.pn_core.throat.oil_filled if self.phase == WATER else self.network.pn_core.throat.water_filled
        )
        self.__njit_update_pore_entry_pc(
            self.network.pn_core.pore.entry_pressure,
            pore_id_list,
            pc,
            self.use_pore_body_filling,
            self.pore_body_filling_strategy,
            self.phase,
            self.network.pn_core.pore.piston_like_pressure,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.shape_factor,
            self.network.pn_core.pore.imminent_snap_off,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            pore_limiting_contact_angle,
            self.network.pn_core.throat.radius,
            throat_receding_phase_filled,
        )

    @staticmethod
    @nb.njit(cache=True)
    def __njit_update_pore_entry_pc(
        pore_entry_pressure,
        pore_id_list,
        pc,
        use_pore_body_filling,
        pore_body_filling_strategy,
        phase,
        pore_piston_like_pressure,
        pore_radius,
        pore_shape_factor,
        pore_imminent_snap_off,
        pore_conns,
        pore_n_conns,
        pore_limiting_contact_angle,
        throat_radius,
        throat_receding_phase_filled,
    ):
        for pore_id in pore_id_list:
            if not use_pore_body_filling or pc <= 0:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
                continue

            neighbors = pore_conns[pore_id][: pore_n_conns[pore_id]]
            mask = throat_receding_phase_filled[neighbors]
            filled_neighbors = neighbors[mask]
            number_of_filled_neighbors = filled_neighbors.size

            if number_of_filled_neighbors <= 1:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
            else:
                throats_radii = throat_radius[filled_neighbors]

                pore_body_filling_pressure = calculate_pore_body_filling_pc(
                    pore_body_filling_strategy,
                    pore_radius[pore_id],
                    pore_limiting_contact_angle[pore_id],
                    number_of_filled_neighbors,
                    throats_radii,
                    pore_shape_factor[pore_id],
                )
                piston_like_pressure = pore_piston_like_pressure[pore_id]

                if phase == WATER and pore_body_filling_pressure > piston_like_pressure:
                    pore_entry_pressure[pore_id] = piston_like_pressure
                elif phase == OIL and pore_body_filling_pressure < piston_like_pressure:
                    pore_entry_pressure[pore_id] = piston_like_pressure
                else:
                    if pore_imminent_snap_off[pore_id]:
                        if phase == WATER:
                            if pore_body_filling_pressure > pore_entry_pressure[pore_id]:
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                        else:
                            if pore_body_filling_pressure < pore_entry_pressure[pore_id]:
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                    else:
                        pore_entry_pressure[pore_id] = pore_body_filling_pressure

    def _update_throat_entry_pc(self, throat_id_list, replace_snap_off=False):
        if replace_snap_off:
            self.snap_off_throat_list = self._update_throats_snap_offs_pcs(self.network.pn.throats())
        else:
            new_snap_off_throats = self._update_throats_snap_offs_pcs(throat_id_list)
            if len(new_snap_off_throats) > 0:
                self.snap_off_throat_list = np.unique(np.concatenate((self.snap_off_throat_list, new_snap_off_throats)))

        self._update_border_throat_entry_pc(throat_id_list)

    def _update_border_throat_entry_pc(self, throat_id_list):
        self.__njit_update_throat_entry_pc(
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.piston_like_pressure,
            self.network.pn_core.throat.in_border,
        )

    @staticmethod
    @nb.njit(cache=True)
    def __njit_update_throat_entry_pc(
        throat_id_list,
        throat_entry_pressure,
        throat_piston_like_pressure,
        throat_in_border,
    ):
        for throat_id in throat_id_list:
            if throat_in_border[throat_id]:
                throat_entry_pressure[throat_id] = throat_piston_like_pressure[throat_id]

    def _get_element_to_fill(self, pc, border_elements):
        self.__clear_snapoff_lists()
        pore_id_list = np.concatenate((border_elements.get_pores_ids(), self.snap_off_pore_list))
        throat_id_list = np.concatenate((border_elements.get_throats_ids(), self.snap_off_throat_list))
        element_to_fill = self._jit_pore_throats_element_to_fill(
            pc,
            pore_id_list,
            self.network.pn_core.pore.entry_pressure,
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
        )

        return element_to_fill

    @abstractmethod
    def _can_be_filled(self, current_pc, invasion_pc):
        pass

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

    def _update_network_geometry(self, pc):
        self._update_hinging_contact_angles(pc)
        self._update_b(pc)
        self._update_oil_layers(pc)
        self.__update_water_layers()
        super()._update_network_geometry(pc)

    def __clear_snapoff_lists(self):
        filled_pores = (
            self.network.pn_core.pore.water_filled[self.snap_off_pore_list]
            if self.phase == WATER
            else self.network.pn_core.pore.oil_filled[self.snap_off_pore_list]
        )
        self.snap_off_pore_list = np.delete(
            self.snap_off_pore_list,
            filled_pores | self.network.pn_core.pore.trapped_evacuation[self.snap_off_pore_list],
        )
        filled_throats = (
            self.network.pn_core.throat.water_filled[self.snap_off_throat_list]
            if self.phase == WATER
            else self.network.pn_core.throat.oil_filled[self.snap_off_throat_list]
        )
        self.snap_off_throat_list = np.delete(
            self.snap_off_throat_list,
            filled_throats | self.network.pn_core.throat.trapped_evacuation[self.snap_off_throat_list],
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


class SubsequentSaturation(SaturationCalculator):
    def __init__(self, network, phase):
        super().__init__(network, phase)

    def _get_pores_saturations(self, pc):
        return calculate_imbibition_network_saturation(
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.corner_oil_layer,
        )

    def _get_throats_saturations(self, pc):
        return calculate_imbibition_network_saturation(
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.corner_oil_layer,
        )

    def _get_element_saturations(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        return calculate_imbibition_element_saturation(
            area,
            element_data.oil_filled[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
            element_data.corner_half_angle[id],
            element_data.b[id],
            element_data.bo[id],
            element_data.n_corners[id],
            element_data.corner_oil_layer[id],
        )
