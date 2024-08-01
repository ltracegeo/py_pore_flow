import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import OIL, check_water_layer
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import CycleSimulation


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False):
        super().__init__(network, OIL, calculate_full_conductance)

    def get_pores_left(self):
        return np.logical_not(self.network.pn["pore.oil_filled"] | self.network.pn["pore.trapped"]).sum()

    def _get_element_to_fill(self, pc, border_elements):
        pore_pressure_array = self.network.pn["pore.entry_pressure"]
        pore_id_array = border_elements.get_pores_ids()
        if len(pore_id_array) > 0:
            minor_id_index = np.argmin(pore_pressure_array[pore_id_array])
            pore_id = pore_id_array[minor_id_index]
        else:
            pore_id = None

        throat_pressure_array = self.network.pn["throat.entry_pressure"]
        throat_id_array = border_elements.get_throats_ids()
        if len(throat_id_array) > 0:
            minor_id_index = np.argmin(self.network.pn["throat.entry_pressure"][throat_id_array])
            throat_id = throat_id_array[minor_id_index]
        else:
            throat_id = None

        if pore_id is not None and throat_id is not None:
            if throat_pressure_array[throat_id] < pore_pressure_array[pore_id]:
                element_to_fill = self.network.get_throat(throat_id)
                filling_pc = throat_pressure_array[throat_id]
            else:
                element_to_fill = self.network.get_pore(pore_id)
                filling_pc = pore_pressure_array[pore_id]
        elif pore_id is not None:
            element_to_fill = self.network.get_pore(pore_id)
            filling_pc = pore_pressure_array[pore_id]
        elif throat_id is not None:
            element_to_fill = self.network.get_throat(throat_id)
            filling_pc = throat_pressure_array[throat_id]
        else:
            element_to_fill = None
            filling_pc = pc

        return element_to_fill, filling_pc if filling_pc > pc else pc

    def _on_element_invaded(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
