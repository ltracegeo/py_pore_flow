import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import OIL, check_water_layer
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False):
        super().__init__(network, OIL, calculate_full_conductance)

    def get_pores_left(self):
        return np.logical_not(self.network.pn_core.pore.oil_filled | self.network.pn_core.pore.trapped).sum()

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
    @nb.njit
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
