import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    WATER,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_oil_layer,
    calculate_oil_layer_network,
    calculate_r,
    check_water_layer_network,
)
from py_pore_flow.equations.entry_pressure import (
    calculate_imbibition_pc,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_snap_off_pc,
)
from py_pore_flow.simulation.cycle_simulation import CycleSimulation
from py_pore_flow import log


class ImbibitionSimulation(CycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, WATER, calculate_full_conductance)
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
        element_id_list,
        element_n_corners,
        element_conns,
        element_n_conns,
        element_water_filled,
        element_oil_filled,
        element_shape,
        element_r,
        element_beta,
        element_theta_a,
        element_theta_r,
        element_theta_h,
        element_snap_off_viability,
        neighbor_water_filled,
        Rmin,
        pc,
        border_elements,
        is_throats=True,
    ):
        invaded_elements = []
        for element_id in element_id_list:
            if (
                element_snap_off_viability[element_id]
                and not element_water_filled[element_id]
                and element_shape[element_id] != CIRCLE
            ):
                for i in range(element_n_conns[element_id]):
                    pore_id = element_conns[element_id][i]
                    if neighbor_water_filled[pore_id]:
                        continue
                snap_off_pc = calculate_snap_off_pc(
                    element_r[element_id],
                    element_beta[element_id],
                    element_theta_a[element_id],
                    element_theta_r[element_id],
                    element_theta_h[element_id],
                    element_n_corners[element_id],
                    Rmin,
                    pc,
                    0,
                )
                if snap_off_pc > pc:
                    element_water_filled[element_id] = True
                    element_oil_filled[element_id] = False
                    if is_throats:
                        border_elements.remove_throat(element_id)
                    else:
                        border_elements.remove_pore(element_id)

                    invaded_elements.append([element_id, snap_off_pc])

                    for i in range(element_n_conns[element_id]):
                        neighbor_id = element_conns[element_id][i]
                        if not neighbor_water_filled[neighbor_id]:
                            if is_throats:
                                border_elements.add_pore(neighbor_id)
                            else:
                                border_elements.add_throat(neighbor_id)

        return invaded_elements

    def __update_snap_offs(self, pc):
        invaded_throats = self.snap_off(
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

        invaded_pores = self.snap_off(
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

        throat_r = self.network.pn["throat.radius"]
        for throat in invaded_throats:
            log.debug(
                f"Invasion {'throat':6} ID: {int(throat[0]):6} Pc: {throat[1]} Radius: {throat_r[int(throat[0])]}"
            )

        pore_r = self.network.pn["pore.radius"]
        for pore in invaded_pores:
            log.debug(f"Invasion {'pore':6} ID: {int(pore[0]):6} Pc: {pore[1]} Radius: {pore_r[int(pore[0])]}")

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

    def _update_element_entry_pc(self, element, pc):
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
