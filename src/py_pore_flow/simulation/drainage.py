import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    OIL,
    check_water_layer,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
)
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT, CycleSimulation, SaturationCalculator
from py_pore_flow.simulation.algorithms.sorted_root_tree import SortedRootTree


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False):
        super().__init__(network, OIL, calculate_full_conductance)
        self.saturation_calculator = DrainageSaturation(network)
        self.sorted_root_tree = SortedRootTree(
            self.network.pn_core.throat.entry_pressure, self.network.pn_core.pore.entry_pressure
        )
        self.previous_pore_ids = np.array([])
        self.previous_throat_ids = np.array([])

        self.initialize()

    def _get_element_to_fill(self, pc, border_elements):
        current_pores_ids = border_elements.get_pores_ids()
        current_throats_ids = border_elements.get_throats_ids()
        if len(current_pores_ids) == 0 and len(current_throats_ids) == 0:
            return None

        new_pore_ids = np.setdiff1d(current_pores_ids, self.previous_pore_ids)
        new_throat_ids = np.setdiff1d(current_throats_ids, self.previous_throat_ids)
        element_to_fill = self.__jit_pore_throats_element_to_fill(
            pc,
            new_pore_ids,
            self.network.pn_core.pore.entry_pressure,
            new_throat_ids,
            self.network.pn_core.throat.entry_pressure,
            self.sorted_root_tree,
        )
        self.previous_pore_ids = current_pores_ids.copy()
        self.previous_throat_ids = current_throats_ids.copy()

        return element_to_fill

    def _add_element_to_fill(self, pore_id_list, throat_id_list):
        for pore_id in pore_id_list:
            self.sorted_root_tree.add_node(self.network.pn_core.pore.entry_pressure[pore_id], pore_id + 1)
        for throat_id in throat_id_list:
            self.sorted_root_tree.add_node(
                self.network.pn_core.throat.entry_pressure[throat_id], throat_id + self.sorted_root_tree.pore_n + 1
            )

    def _pop_element_to_fill(self):
        self.sorted_root_tree.pop_root()

    @staticmethod
    # @nb.njit(cache=True)
    def __jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        throat_id_list,
        throat_entry_pressure,
        sorted_root_tree,
    ):
        # if len(pore_id_list) == 0 and len(throat_id_list) == 0:
        #     return None

        # for pore_id in pore_id_list:
        #     sorted_root_tree.add_node(pore_entry_pressure[pore_id], pore_id + 1)
        # for throat_id in throat_id_list:
        #     sorted_root_tree.add_node(throat_entry_pressure[throat_id], throat_id + sorted_root_tree.pore_n + 1)
        root_label = sorted_root_tree.get_root()
        if root_label != 0:
            if root_label > sorted_root_tree.pore_n:
                min_id = root_label - sorted_root_tree.pore_n - 1
                min_pressure = throat_entry_pressure[min_id]
                min_type = THROAT
            else:
                min_id = root_label - 1
                min_pressure = pore_entry_pressure[min_id]
                min_type = PORE
        else:
            min_pressure = -np.inf
            min_id = -1
            min_type = -1

        if min_id != -1:
            return (min_type, min_id, max(min_pressure, pc))
        else:
            return None

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_element_filled_geometry(self, pc, element):
        element.corner_water_layer = check_water_layer(element.corner_oil_layer, element.oil_filled)
        return super()._update_element_filled_geometry(pc, element)


class DrainageSaturation(SaturationCalculator):
    def __init__(self, network):
        super().__init__(network, OIL)

    def _get_pores_saturations(self, pc):
        return calculate_drainage_network_saturation(
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.C_r,
            pc,
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
        )

    def _get_throats_saturations(self, pc):
        return calculate_drainage_network_saturation(
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.C_r,
            pc,
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
        )

    def _get_element_saturations(self, type_name, id, pc):
        if type_name == "pore":
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        return calculate_drainage_element_saturation(
            element_data.shape[id],
            element_data.C_r[id],
            pc,
            area,
            element_data.oil_filled[id],
        )
