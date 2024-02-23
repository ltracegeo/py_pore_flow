import numpy as np

from pore_flow.equations import OIL
from pore_flow.output import pn_to_vtu
from pore_flow.simulation.simulation_base import SimulationBase


def run_drainage(network, pc_incrementer, generate_vtu=False, debug=False):
    if debug:
        print("Start drainage")

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
            print(
                f"Pc: {Pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage:2}%)."
            )

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

    def _check_invasion_condition(self, pc, element):
        return pc > element.get_pc()
