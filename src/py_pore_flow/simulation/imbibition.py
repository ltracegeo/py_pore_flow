import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations import (
    CIRCLE,
    WATER,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_oil_layer,
    calculate_oil_layer_network,
    calculate_r,
    calculate_snap_off_pc,
    check_water_layer_network,
)
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import CycleSimulation


class ImbibitionSimulation(CycleSimulation):
    def __init__(self, network, pc_max):
        super().__init__(network, WATER)
        self.Rmin = calculate_r(pc_max, 100000)

    def get_pores_left(self):
        return np.logical_not(self.network.pn["pore.water_filled"] | self.network.pn["pore.trapped"]).sum()

    def _update_invasions(self, pc):
        super()._update_invasions(pc)
        self.__update_snap_offs(pc)
        self.__update_oil_layers(pc)
        self.__update_water_layers()

    @staticmethod
    @nb.njit
    def snap_off(
        throat_id_list,
        throat_n_corners,
        throat_conns,
        throat_n_conns,
        throat_water_filled,
        throat_oil_filled,
        throat_shape,
        throat_r,
        throat_beta,
        throat_theta_a,
        throat_theta_r,
        throat_theta_h,
        throat_snap_off_viability,
        pore_water_filled,
        Rmin,
        pc,
        border_elements,
        is_throats=True,
    ):
        for throat_id in throat_id_list:
            if (
                throat_snap_off_viability[throat_id]
                and not throat_water_filled[throat_id]
                and throat_shape[throat_id] != CIRCLE
            ):
                for i in range(throat_n_conns[throat_id]):
                    pore_id = throat_conns[throat_id][i]
                    if pore_water_filled[pore_id]:
                        continue
                snap_off_pc = calculate_snap_off_pc(
                    throat_r[throat_id],
                    throat_beta[throat_id],
                    throat_theta_a[throat_id],
                    throat_theta_r[throat_id],
                    throat_theta_h[throat_id],
                    throat_n_corners[throat_id],
                    Rmin,
                    pc,
                    0,
                )
                if snap_off_pc > pc:
                    throat_water_filled[throat_id] = True
                    throat_oil_filled[throat_id] = False
                    if is_throats:
                        border_elements.remove_throat(throat_id)
                    else:
                        border_elements.remove_pore(throat_id)
                    for i in range(throat_n_conns[throat_id]):
                        pore_id = throat_conns[throat_id][i]
                        if not pore_water_filled[pore_id]:
                            if is_throats:
                                border_elements.add_pore(pore_id)
                            else:
                                border_elements.add_throat(pore_id)

    def __update_snap_offs(self, pc):
        self.snap_off(
            self.network.pn.throats(),
            self.network.pn["throat.n_corners"],
            self.network.pn["throat.conns"],
            self.network.pn["throat.n_conns"],
            self.network.pn["throat.water_filled"],
            self.network.pn["throat.oil_filled"],
            self.network.pn["throat.shape"],
            self.network.pn["throat.radius"],
            self.network.pn["throat.corner_half_angle"],
            self.network.pn["throat.advancing_contact_angle"],
            self.network.pn["throat.receding_contact_angle"],
            self.network.pn["throat.hinging_contact_angle"],
            self.network.pn["throat.snap_off_viability"],
            self.network.pn["pore.water_filled"],
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=True,
        )

        self.snap_off(
            self.network.pn.pores(),
            self.network.pn["pore.n_corners"],
            self.network.pn["pore.conns"],
            self.network.pn["pore.n_conns"],
            self.network.pn["pore.water_filled"],
            self.network.pn["pore.oil_filled"],
            self.network.pn["pore.shape"],
            self.network.pn["pore.radius"],
            self.network.pn["pore.corner_half_angle"],
            self.network.pn["pore.advancing_contact_angle"],
            self.network.pn["pore.receding_contact_angle"],
            self.network.pn["pore.hinging_contact_angle"],
            self.network.pn["pore.snap_off_viability"],
            self.network.pn["throat.water_filled"],
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=False,
        )

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

    def __update_water_layers(self):
        self.network.pn[f"throat.corner_water_layer"] = check_water_layer_network(
            self.network.pn[f"throat.corner_oil_layer"], self.network.pn[f"throat.oil_filled"]
        )
        self.network.pn[f"pore.corner_water_layer"] = check_water_layer_network(
            self.network.pn[f"pore.corner_oil_layer"], self.network.pn[f"pore.oil_filled"]
        )

    def _update_pc(self, element, pc):
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
        element.set_pc(element_pc)

    def _get_element_to_fill(self, pc, border_elements):
        element_list = []
        for pore_id in border_elements.get_pores_ids():
            element_list.append(self.network.get_pore(pore_id))
        for throat_id in border_elements.get_throats_ids():
            element_list.append(self.network.get_throat(throat_id))
        element_list.sort(key=lambda x: x.get_pc(), reverse=True)

        for element in element_list:
            if pc < 0 or element.spontaneous_imbibition:
                if element.get_pc() > pc and not element.trapped and not element.water_filled:
                    return element
        return None

    def _update_b(self, pc):
        self.network.pn["pore.b"] = calculate_imbibition_network_b(
            self.network.pn["pore.corner_half_angle"],
            self.network.pn["pore.max_b"],
            self.network.pn["pore.receding_contact_angle"],
            self.network.pn["pore.advancing_contact_angle"],
            self.network.pn["pore.hinging_contact_angle"],
            self.network.pn["pore.corner_water_layer"],
            self.network.pn["pore.n_corners"],
            pc,
            self.Rmin,
        )
        self.network.pn["throat.b"] = calculate_imbibition_network_b(
            self.network.pn["throat.corner_half_angle"],
            self.network.pn["throat.max_b"],
            self.network.pn["throat.receding_contact_angle"],
            self.network.pn["throat.advancing_contact_angle"],
            self.network.pn["throat.hinging_contact_angle"],
            self.network.pn["throat.corner_water_layer"],
            self.network.pn["throat.n_corners"],
            pc,
            self.Rmin,
        )

        self.network.pn["pore.bo"] = calculate_imbibition_network_bo(
            self.network.pn["pore.corner_half_angle"],
            self.network.pn["pore.max_b"],
            self.network.pn["pore.advancing_contact_angle"],
            self.network.pn["pore.corner_oil_layer"],
            self.network.pn["pore.n_corners"],
            pc,
        )
        self.network.pn["throat.bo"] = calculate_imbibition_network_bo(
            self.network.pn["throat.corner_half_angle"],
            self.network.pn["throat.max_b"],
            self.network.pn["throat.advancing_contact_angle"],
            self.network.pn["throat.corner_oil_layer"],
            self.network.pn["throat.n_corners"],
            pc,
        )

    def _on_element_invaded(self, pc, element):
        element.corner_oil_layer = calculate_oil_layer(
            element.advancing_contact_angle, element.beta, element.n_corners, pc
        )
