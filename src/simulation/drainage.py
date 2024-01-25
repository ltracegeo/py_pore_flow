from equations import OIL
from simulation.simulation_base import SimulationBase


def run_drainage(network, pc_incrementer, debug=False):
    if debug:
        print("Start drainage")

    simulation = DrainageSimulation(network)

    Sw_array = []
    Krw_array = []
    Kro_array = []

    Pc = pc_incrementer.get_pc()
    while Pc <= pc_incrementer.get_pc_max():
        simulation.update_pc(Pc)
        Sw = simulation.calculate_sw_total(Pc)
        Krw, Kro = simulation.calculate_krel()
        pores_filled, total_pores, percentage = simulation.get_oil_filled_percentage()

        Sw_array.append(Sw)
        Krw_array.append(Krw)
        Kro_array.append(Kro)

        if debug:
            print(
                f"Pc: {Pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage:2}%)."
            )

        Pc = pc_incrementer.increment(Sw)

    return Sw_array, Krw_array, Kro_array


class DrainageSimulation(SimulationBase):
    def __init__(self, network):
        super().__init__(network, OIL)

    def _check_invasion_condition(self, pc, element):
        return pc > element.get_pc()
