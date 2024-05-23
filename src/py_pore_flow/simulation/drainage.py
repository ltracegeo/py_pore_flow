import numpy as np

from py_pore_flow import log
from py_pore_flow.equations import OIL
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.simulation_base import SimulationBase


def run_drainage(network, pc_incrementer, network_plot, generate_vtu=False, debug=False):
    if debug:
        log.info("Start drainage")

    simulation = DrainageSimulation(network)

    Pc_array = []
    Sw_array = []
    Krw_array = []
    Kro_array = []

    while True:
        Pc = pc_incrementer.get_pc()

        simulation.update_pc(Pc)
        Sw = simulation.calculate_sw_total()
        Krw, Kro = simulation.calculate_krel()
        pores_filled, total_pores, percentage = simulation.get_oil_filled_percentage()

        Pc_array.append(Pc)
        Sw_array.append(Sw)
        Krw_array.append(Krw)
        Kro_array.append(Kro)

        if debug:
            log.info(
                f"Pc: {Pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage:2}%)."
            )

        if network_plot is not None:
            network_plot.update()

        if generate_vtu:
            pn_to_vtu(network.pn, Pc, Sw, 1)

        if simulation.get_pores_left() == 0:
            break

        if Pc >= pc_incrementer.get_pc_max():
            break
        else:
            pc_incrementer.increment(Sw)

    return Pc_array, Sw_array, Krw_array, Kro_array


class DrainageSimulation(SimulationBase):
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
