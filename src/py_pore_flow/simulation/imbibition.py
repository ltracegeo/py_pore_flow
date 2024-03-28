import numpy as np

from py_pore_flow import log
from py_pore_flow.equations import (
    CIRCLE,
    WATER,
    calculate_imbibition_network_b,
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_oil_layer,
    calculate_oil_layer_network,
    calculate_r,
    calculate_snap_off_pc,
)
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.simulation_base import SimulationBase


def run_imbibition(network, pc_incrementer, end_at_sor=False, generate_vtu=False, debug=False):
    if debug:
        log.info("Start imbibition")

    simulation = ImbibitionSimulation(network, pc_incrementer.get_pc())

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

        if generate_vtu:
            pn_to_vtu(network.pn, Pc, Sw, 2)

        if simulation.get_pores_left() == 0:
            break

        if end_at_sor and Kro <= 0:
            break

        if Pc <= pc_incrementer.get_pc_min():
            break
        else:
            pc_incrementer.decrement(Sw)

    return Pc_array, Sw_array, Krw_array, Kro_array


class ImbibitionSimulation(SimulationBase):
    def __init__(self, network, pc_max):
        super().__init__(network, WATER)
        self.Rmin = calculate_r(pc_max)

    def get_pores_left(self):
        return np.logical_not(self.network.pn["pore.water_filled"] | self.network.pn["pore.trapped"]).sum()

    def _update_invasions(self, pc):
        super()._update_invasions(pc)
        self.__update_snap_offs(pc)
        self.__update_oil_layers(pc)

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
                    self.border_elements.remove_throat(throat.get_id())
                    neighbor_pore_list = self.network.get_neighbor_pores(throat.get_id())
                    for neighbor_pore in neighbor_pore_list:
                        if not neighbor_pore.is_filled(self.phase):
                            self.border_elements.add_pore(neighbor_pore.get_id())

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
                        self.border_elements.remove_pore(pore.get_id())
                        neighbor_throat_list = self.network.get_neighbor_throats(pore.get_id())
                        for neighbor_throat in neighbor_throat_list:
                            if not neighbor_throat.is_filled(self.phase):
                                self.border_elements.add_throat(neighbor_throat.get_id())

    def __update_oil_layers(self, pc):
        self.network.pn[f"pore.corner_oil_layer"] = calculate_oil_layer_network(
            self.network.pn[f"pore.corner_oil_layer"],
            self.network.pn[f"pore.corner_half_angle"],
            self.network.pn[f"pore.advancing_contact_angle"],
            self.network.pn[f"pore.b"],
            self.network.pn[f"pore.n_corners"],
            pc,
        )

        self.network.pn[f"throat.corner_oil_layer"] = calculate_oil_layer_network(
            self.network.pn[f"throat.corner_oil_layer"],
            self.network.pn[f"throat.corner_half_angle"],
            self.network.pn[f"throat.advancing_contact_angle"],
            self.network.pn[f"throat.b"],
            self.network.pn[f"throat.n_corners"],
            pc,
        )

    def _check_invasion_condition(self, pc, element):
        if pc > 0 and not element.spontaneous_imbibition:
            return False

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
                element.beta,
                element.n_corners,
                number_of_oil_filled_neighbors,
                K,
            )
        else:
            element_pc = calculate_imbibition_piston_like_displacement_pc(
                element.G,
                element.r,
                element.advancing_contact_angle,
                element.shape,
                element.C_a,
                pc,
                element.beta,
                element.n_corners,
            )
        return element_pc > pc

    def _update_b(self, pc):
        self.network.pn["pore.b"] = calculate_imbibition_network_b(
            self.network.pn["pore.corner_half_angle"],
            self.network.pn["pore.receding_contact_angle"],
            self.network.pn["pore.advancing_contact_angle"],
            self.network.pn["pore.hinging_contact_angle"],
            self.network.pn["pore.n_corners"],
            pc,
            self.Rmin,
        )

        self.network.pn["throat.b"] = calculate_imbibition_network_b(
            self.network.pn["throat.corner_half_angle"],
            self.network.pn["throat.receding_contact_angle"],
            self.network.pn["throat.advancing_contact_angle"],
            self.network.pn["throat.hinging_contact_angle"],
            self.network.pn["throat.n_corners"],
            pc,
            self.Rmin,
        )

    def _on_element_invaded(self, pc, element):
        element.corner_oil_layer = calculate_oil_layer(
            element.advancing_contact_angle, element.beta, element.n_corners, pc
        )
