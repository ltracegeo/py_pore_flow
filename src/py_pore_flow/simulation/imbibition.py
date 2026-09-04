import numba as nb
import numpy as np

import py_pore_flow.consts as c
from py_pore_flow.equations.geometry import (
    CIRCLE,
    WATER,
    calculate_hinging_contact_angle,
    calculate_imbibition_element_b,
    calculate_imbibition_element_bo,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_multiphase_mask_imbibition,
    calculate_network_hinging_contact_angle,
    calculate_oil_layer,
    calculate_oil_layer_element,
    calculate_oil_layer_network,
    check_water_layer,
)
from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log


class ImbibitionSimulation(SubsequentCycleSimulation):
    def __init__(
        self,
        network,
        pc_max,
        calculate_full_conductance=False,
        batch_invasions=False,
        calculate_krel=True,
        calculate_trapping=True,
    ):
        super().__init__(
            network,
            pc_max,
            WATER,
            calculate_full_conductance,
            batch_invasions=batch_invasions,
            calculate_krel=calculate_krel,
            calculate_trapping=calculate_trapping,
        )

    def _update_multiphase_mask(self, pc, pc_old):
        calculate_multiphase_mask_imbibition(
            self.network.pn_core.pore.is_multiphase,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.oil_layer_collapse_pressure,
            self.network.pn_core.pore.water_filled,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.trapped,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.oil_filled_multiphase,
            pc,
        )
        calculate_multiphase_mask_imbibition(
            self.network.pn_core.throat.is_multiphase,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.oil_layer_collapse_pressure,
            self.network.pn_core.throat.water_filled,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.trapped,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.oil_filled_multiphase,
            pc,
        )
        super()._update_multiphase_mask(pc, pc_old)

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc <= invasion_pc

    @staticmethod
    @nb.njit(cache=True)
    def _jit_pore_throats_element_to_fill(
        pc,
        pore_id_list,
        pore_entry_pressure,
        pore_trapped,
        throat_id_list,
        throat_entry_pressure,
        throat_trapped,
        verbose=False,
    ):
        max_pressure = -np.inf
        max_id = -1
        max_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            if pore_trapped[id]:
                continue
            pressure = pore_entry_pressure[id]
            if pressure > max_pressure:
                max_pressure = pressure
                max_id = id
                max_type = PORE

        for id in throat_id_list:
            if throat_trapped[id]:
                continue
            pressure = throat_entry_pressure[id]
            if pressure > max_pressure:
                max_pressure = pressure
                max_id = id
                max_type = THROAT

        if max_id != -1:
            if verbose:
                print("  Selected element: ", "pore" if max_type == PORE else "throat", max_id, " with PC: ", max_pressure)
            return (max_type, max_id, min(max_pressure, pc))
        else:
            return None

    def _update_element_filled_geometry(self, pc, element, element_id, verbose=False):
        self._update_element_pinned_hinging_contact_angles(element.type, element_id, pc)
        self._update_element_pinned_b(element.type, element_id, pc)
        element.set_corner_oil_layer(
            element_id,
            calculate_oil_layer(
                element.get_advancing_contact_angle(element_id),
                element.get_beta(element_id),
                element.get_n_corners(element_id),
                pc,
            ),
        )
        self._update_oil_layer_element(element.type, element_id, pc)

        self._update_element_bo(element.type, element_id, pc)
        self._check_water_layer(element.type, element_id)
        self._update_element_hinging_contact_angles(element.type, element_id, pc, verbose=verbose)
        self._update_element_b(element.type, element_id, pc)

        return super()._update_element_filled_geometry(pc, element, element_id, verbose=verbose)

    def _update_element_pinned_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        n_corners = element.n_corners[id]
        element.b[id] = calculate_imbibition_element_b(
            element.b[id],
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            element.hinging_contact_angle[id],
            np.full(n_corners, True, np.bool_),
            n_corners,
            self.Rmin,
            pc,
            c.sigma,
        )

    def _update_element_bo(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
            c.sigma,
        )

    def _update_oil_layer_element(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        calculate_oil_layer_element(
            element.corner_oil_layer[id],
            element.oil_layer_collapse_pressure[id],
            element.b[id],
            element.n_corners[id],
            pc,
        )

    def _check_water_layer(self, element_type, id):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.corner_water_layer[id] = check_water_layer(
            element.corner_oil_layer[id],
            element.oil_filled[id],
        )

    def _update_element_pinned_hinging_contact_angles(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat
        calculate_hinging_contact_angle(
            pc,
            element_data.corner_half_angle[id],
            element_data.b_pinned[id],
            element_data.n_corners[id],
            element_data.receding_contact_angle[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
            c.sigma,
        )

    def _update_element_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.b[id] = calculate_imbibition_element_b(
            element.b[id],
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            element.hinging_contact_angle[id],
            element.corner_water_layer[id],
            element.n_corners[id],
            self.Rmin,
            pc,
            c.sigma,
        )
        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
            c.sigma,
        )

    def _update_b(self, pore_ids, throat_ids, pc):
        calculate_imbibition_network_b(
            self.network.pn_core.pore.b,
            pore_ids,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_water_layer,
            self.network.pn_core.pore.n_corners,
            pc,
            self.Rmin,
            c.sigma,
        )
        calculate_imbibition_network_b(
            self.network.pn_core.throat.b,
            throat_ids,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_water_layer,
            self.network.pn_core.throat.n_corners,
            pc,
            self.Rmin,
            c.sigma,
        )

        calculate_imbibition_network_bo(
            self.network.pn_core.pore.bo,
            pore_ids,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_corners,
            pc,
            c.sigma,
        )
        calculate_imbibition_network_bo(
            self.network.pn_core.throat.bo,
            throat_ids,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.n_corners,
            pc,
            c.sigma,
        )

    def _update_hinging_contact_angles(self, pore_ids, throat_ids, pc):
        calculate_network_hinging_contact_angle(
            pore_ids,
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.trapped_evacuation,
            c.sigma,
        )
        calculate_network_hinging_contact_angle(
            throat_ids,
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.trapped_evacuation,
            c.sigma,
        )

    def _update_oil_layers(self, pore_ids, throat_ids, pc):
        calculate_oil_layer_network(
            self.network.pn_core.pore.corner_oil_layer,
            pore_ids,
            self.network.pn_core.pore.oil_layer_collapse_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            pc,
        )

        calculate_oil_layer_network(
            self.network.pn_core.throat.corner_oil_layer,
            throat_ids,
            self.network.pn_core.throat.oil_layer_collapse_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            pc,
        )
