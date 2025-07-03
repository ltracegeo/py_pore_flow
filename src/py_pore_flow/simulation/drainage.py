import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    OIL,
    check_water_layer,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
    calculate_multiphase_mask_drainage,
)
from py_pore_flow.network import PORE, THROAT
from py_pore_flow.simulation.cycle_simulation import CycleSimulation, SaturationCalculator


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False):
        super().__init__(network, OIL, calculate_full_conductance)
        self.saturation_calculator = DrainageSaturation(network)

    def _update_multiphase_mask(self, pc, pc_old):
        calculate_multiphase_mask_drainage(
            self.network.pn_core.pore.is_multiphase,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.oil_filled,
        )
        calculate_multiphase_mask_drainage(
            self.network.pn_core.throat.is_multiphase,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.oil_filled,
        )
        super()._update_multiphase_mask(pc, pc_old)

    def _get_element_to_fill(self, pc, border_elements):
        pores_to_fill, throats_to_fill, pressure = self.__jit_pore_throats_element_to_fill(
            pc,
            border_elements.get_pores_ids(),
            self.network.pn_core.pore.entry_pressure,
            border_elements.get_throats_ids(),
            self.network.pn_core.throat.entry_pressure,
        )
        return pores_to_fill, throats_to_fill, pressure

    @staticmethod
    @nb.njit(cache=True)
    def __jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        throat_id_list,
        throat_entry_pressure,
    ):
        min_pressure = np.inf

        pores_to_fill_list = np.empty(len(pore_id_list), np.int64)
        throats_to_fill_list = np.empty(len(throat_id_list), np.int64)
        pore_to_fill_count = 0
        throat_to_fill_count = 0

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), pc

        for id in pore_id_list:
            pressure = pore_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                pores_to_fill_list[0] = id
                pore_to_fill_count = 1

        for id in throat_id_list:
            pressure = throat_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                throats_to_fill_list[0] = id
                throat_to_fill_count = 1
                pore_to_fill_count = 0

        return (
            pores_to_fill_list[:pore_to_fill_count],
            throats_to_fill_list[:throat_to_fill_count],
            max(min_pressure, pc),
        )

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_element_filled_geometry(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
        self._update_element_hinging_contact_angles(element.type, element.id, pc)
        return super()._update_element_filled_geometry(pc, element)


class DrainageSaturation(SaturationCalculator):
    def __init__(self, network):
        super().__init__(network, OIL)

    def _get_pores_saturations(self, pc):
        return calculate_drainage_network_saturation(
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.C_r,
            pc,
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
        )

    def _get_throats_saturations(self, pc):
        return calculate_drainage_network_saturation(
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.C_r,
            pc,
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
        )

    def _get_element_saturations(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        return calculate_drainage_element_saturation(
            element_data.shape[id],
            element_data.C_r[id],
            pc,
            area,
            element_data.oil_filled[id],
        )
