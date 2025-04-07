import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    OIL,
    check_water_layer,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
)
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation, SaturationCalculator


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False):
        super().__init__(network, OIL, calculate_full_conductance)
        self.saturation_calculator = DrainageSaturation(network)

    def _get_element_to_fill(self, pc, border_elements):
        element_to_fill = self.__jit_pore_throats_element_to_fill(
            pc,
            border_elements.get_pores_ids(),
            self.network.pn_core.pore.entry_pressure,
            border_elements.get_throats_ids(),
            self.network.pn_core.throat.entry_pressure,
        )
        return element_to_fill

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
        min_id = -1
        min_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            pressure = pore_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                min_id = id
                min_type = PORE

        for id in throat_id_list:
            pressure = throat_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                min_id = id
                min_type = THROAT

        if min_id != -1:
            return (min_type, min_id, max(min_pressure, pc))
        else:
            return None

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_element_filled_geometry(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
        return super()._update_element_filled_geometry(pc, element)


class DrainageSaturation(SaturationCalculator):
    def __init__(self, network):
        super().__init__(network)

    def update_saturations(self, pc):
        self.__update_drainage_saturations(pc)

    def update_element_saturations(self, type_name, id, pc):
        if type_name == "pore":
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat

        self.__update_drainage_element_saturations(type_name, id, pc)
        return element_data.oil_saturation[id] * element_data.volume[id] / self.total_volume

    def __update_drainage_saturations(self, pc):
        (
            self.network.pn_core["pore.water_saturation"],
            self.network.pn_core["pore.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.C_r,
            pc,
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
        )
        (
            self.network.pn_core["throat.water_saturation"],
            self.network.pn_core["throat.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.C_r,
            pc,
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
        )

    def __update_drainage_element_saturations(self, type_name, id, pc):
        if type_name == "pore":
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        (element_data.water_saturation[id], element_data.oil_saturation[id],) = calculate_drainage_element_saturation(
            element_data.shape[id],
            element_data.C_r[id],
            pc,
            area,
            element_data.oil_filled[id],
        )
