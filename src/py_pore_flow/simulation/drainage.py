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
        pore_entry_pressure = self.network.pn_core.pore.entry_pressure
        throat_entry_pressure = self.network.pn_core.throat.entry_pressure
        tuple_list = []
        for id in border_elements.get_pores_ids():
            tuple_list.append((PORE, id, pore_entry_pressure[id]))
        for id in border_elements.get_throats_ids():
            tuple_list.append((THROAT, id, throat_entry_pressure[id]))

        if len(tuple_list) > 0:
            tuple_list.sort(key=lambda element: element[2], reverse=False)
            element_to_fill = tuple_list[0]
            return (element_to_fill[0], element_to_fill[1], max(element_to_fill[2], pc))
        else:
            return None

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_element_filled_geometry(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
        return super()._update_element_filled_geometry(pc, element)
