import numpy as np

from py_pore_flow import log
from py_pore_flow.equations import OIL, check_water_layer
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import CycleSimulation


class DrainageSimulation(CycleSimulation):
    def __init__(self, network):
        super().__init__(network, OIL)

    def get_pores_left(self):
        return np.logical_not(self.network.pn["pore.oil_filled"] | self.network.pn["pore.trapped"]).sum()

    def _get_element_to_fill(self, pc, border_elements):
        element_list = []
        for pore_id in border_elements.get_pores_ids():
            element_list.append(self.network.get_pore(pore_id))
        for throat_id in border_elements.get_throats_ids():
            element_list.append(self.network.get_throat(throat_id))
        element_list.sort(key=lambda x: x.get_pc(), reverse=False)

        for element in element_list:
            if pc > element.get_pc() and not element.trapped and not element.oil_filled:
                return element
        return None

    def _on_element_invaded(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
