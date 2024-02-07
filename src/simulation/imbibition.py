import numpy as np

from equations import (
    CIRCLE,
    WATER,
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_r,
    calculate_snap_off_pc,
)
from simulation.simulation_base import SimulationBase


def run_imbibition(network, pc_incrementer, debug=False):
    if debug:
        print("Start imbibition")

    simulation = ImbibitionSimulation(network, pc_incrementer.get_pc())

    Sw_array = []
    Krw_array = []
    Kro_array = []

    while True:
        Pc = pc_incrementer.get_pc()

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

        if simulation.get_pores_left() == 0:
            break

        if Pc <= pc_incrementer.get_pc_min():
            break
        else:
            pc_incrementer.decrement(Sw)

    return Sw_array, Krw_array, Kro_array


class ImbibitionSimulation(SimulationBase):
    def __init__(self, network, pc_max):
        super().__init__(network, WATER)
        self.Rmin = calculate_r(pc_max)

    def get_pores_left(self):
        return np.logical_not(self.network.pn["pore.water_filled"] | self.network.pn["pore.trapped"]).sum()

    def _update_invasions(self, pc):
        super()._update_invasions(pc)
        self.__update_snap_offs(pc)

    def __update_snap_offs(self, pc):
        for throat in self.network.get_throat_list():
            if not throat.is_filled(self.phase) and throat.shape != CIRCLE:
                for pore in self.network.get_neighbor_pores(throat.get_id()):
                    if pore.is_filled(self.phase):
                        continue
                if (
                    calculate_snap_off_pc(
                        throat.r,
                        throat.beta,
                        throat.advancing_contact_angle,
                        throat.receding_contact_angle,
                        throat.theta_h,
                        self.Rmin,
                        pc,
                        0,
                    )
                    > pc
                ):
                    throat.set_filled(self.phase, True)

        for pore in self.network.get_pore_list():
            if not pore.is_filled(self.phase) and pore.shape != CIRCLE:
                for throat in self.network.get_neighbor_throats(pore.get_id()):
                    if throat.is_filled(self.phase):
                        continue
                    if (
                        calculate_snap_off_pc(
                            pore.r,
                            pore.beta,
                            pore.advancing_contact_angle,
                            pore.receding_contact_angle,
                            pore.theta_h,
                            self.Rmin,
                            pc,
                            0,
                        )
                        > pc
                    ):
                        pore.set_filled(self.phase, True)

    def _check_invasion_condition(self, pc, element):
        if element.is_pore():
            K = self.krel_calculator.get_absolute_permeability()
            number_of_oil_filled_neighbors = 0
            for neighbor in self.network.get_neighbor_throats(element.get_id()):
                if neighbor.oil_filled:
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
