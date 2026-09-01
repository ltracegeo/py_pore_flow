import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    OIL,
    check_water_layer,
    calculate_multiphase_mask_2nd_drainage,
    calculate_2nd_drainage_element_b,
    calculate_2nd_drainage_network_b,
    calculate_2nd_drainage_network_bo,
    calculate_network_hinging_contact_angle,
    calculate_network_outter_hinging_contact_angle,
)
import py_pore_flow.consts as c
from py_pore_flow.equations.entry_pressure import PORE_BODY_FILLING_STRATEGY
from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log


class SecondDrainageSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False, batch_invasions=False):
        super().__init__(network, pc_max, OIL, calculate_full_conductance, batch_invasions=batch_invasions)

        if c.pore_fill_alg == "oren2":
            self.pore_body_filling_strategy = PORE_BODY_FILLING_STRATEGY["oren1"]

    def _update_multiphase_mask(self, pc, pc_old):
        calculate_multiphase_mask_2nd_drainage(
            self.network.pn_core.pore.is_multiphase,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.water_filled,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.trapped,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.water_filled_multiphase,
            self.network.pn_core.pore.oil_filled_multiphase,
            self.network.pn_core.pore.water_corner_collapse_pressure,
            self.network.pn_core.pore.b_pinned,
            pc,
        )
        calculate_multiphase_mask_2nd_drainage(
            self.network.pn_core.throat.is_multiphase,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.water_filled,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.trapped,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.water_filled_multiphase,
            self.network.pn_core.throat.oil_filled_multiphase,
            self.network.pn_core.throat.water_corner_collapse_pressure,
            self.network.pn_core.throat.b_pinned,
            pc,
        )
        super()._update_multiphase_mask(pc, pc_old)

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

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
        min_pressure = np.inf
        min_id = -1
        min_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            if pore_trapped[id]:
                continue
            pressure = pore_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                min_id = id
                min_type = PORE

        for id in throat_id_list:
            if throat_trapped[id]:
                continue
            pressure = throat_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                min_id = id
                min_type = THROAT

        if min_id != -1:
            if verbose:
                print("  Selected element: ", "pore" if min_type == PORE else "throat", min_id, " with PC: ", min_pressure)
            return (min_type, min_id, max(min_pressure, pc))
        else:
            return None

    def _update_element_filled_geometry(self, pc, element, element_id, verbose=False):
        element.set_corner_oil_layer(element_id, False)
        element.set_bo(element_id, 0.0)
        element.set_corner_water_layer(
            element_id,
            check_water_layer(
                element.get_corner_oil_layer(element_id),
                element.get_oil_filled(element_id),
            ),
        )
        self._update_element_pinned_b(element.type, element_id, pc)
        self._update_element_hinging_contact_angles(element.type, element_id, pc, verbose=verbose)
        self._update_element_b(element.type, element_id, pc)
        return super()._update_element_filled_geometry(pc, element, element_id, verbose=verbose)

    def _update_element_pinned_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        n_corners = element.n_corners[id]
        element.b[id] = calculate_2nd_drainage_element_b(
            element.b[id],
            element.corner_half_angle[id],
            element.max_b[id],
            element.receding_contact_angle[id],
            element.advancing_contact_angle[id],
            np.full(n_corners, element.advancing_contact_angle[id], np.float64),
            element.corner_water_layer[id],
            n_corners,
            self.Rmin,
            pc,
            c.sigma,
        )

    def _update_element_b(self, element_type, id, pc):
        if element_type == PORE:
            element = self.network.pn_core.pore
        else:
            element = self.network.pn_core.throat

        element.b[id] = calculate_2nd_drainage_element_b(
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

    def _update_b(self, pore_ids, throat_ids, pc):
        calculate_2nd_drainage_network_b(
            self.network.pn_core.pore.b,
            pore_ids,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.oil_filled,
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
        calculate_2nd_drainage_network_b(
            self.network.pn_core.throat.b,
            throat_ids,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.oil_filled,
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

        calculate_2nd_drainage_network_bo(
            self.network.pn_core.pore.bo,
            pore_ids,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.max_b,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle_o,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.n_corners,
            pc,
            c.sigma,
        )
        calculate_2nd_drainage_network_bo(
            self.network.pn_core.throat.bo,
            throat_ids,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.max_b,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle_o,
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

        calculate_network_outter_hinging_contact_angle(
            pore_ids,
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle_o,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.max_b,
            c.sigma,
        )
        calculate_network_outter_hinging_contact_angle(
            throat_ids,
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle_o,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.max_b,
            c.sigma,
        )

    def _update_oil_layers(self, pore_ids, throat_ids, pc):
        pass
