from equations import (
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
    WATER,
)
from simulation.simulation_base import SimulationBase


def run_imbibition(network, pc_incrementer, debug=False):
    if debug:
        print("Start imbibition")

    simulation = ImbibitionSimulation(network)

    Sw_array = []
    Krw_array = []
    Kro_array = []

    Pc = pc_incrementer.get_pc()
    while Pc > pc_incrementer.get_pc_min():
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

        Pc = pc_incrementer.decrement(Sw)

    return Sw_array, Krw_array, Kro_array


class ImbibitionSimulation(SimulationBase):
    def __init__(self, network):
        super().__init__(network, WATER)

    def _check_invasion_condition(self, pc, element):
        if element.is_pore():
            K = self.krel_calculator.get_absolute_permeability()
            number_of_oil_filled_neighbors = 0
            for neighbor in self.network.get_neighbor_throats(element.get_id()):
                if neighbor.is_oil_filled():
                    number_of_oil_filled_neighbors += 1
            element_pc = calculate_imbibition_pc(
                element.G,
                element.r,
                element.advancing_contact_angle,
                element.shape,
                element.C_a,
                pc,
                0,
                number_of_oil_filled_neighbors,
                K,
            )
        else:
            element_pc = calculate_imbibition_piston_like_displacement_pc(
                element.G, element.r, element.advancing_contact_angle, element.shape, element.C_a, pc, 0
            )
        return element_pc > pc
