import numba as nb
import numpy as np

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
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation, _find_max_entry_pressure
from py_pore_flow import log


class ImbibitionSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, pc_max, WATER, calculate_full_conductance)

    def _get_border_candidate(self, border_elements):
        """Imbibition selects the element with the highest entry pressure via njit scan."""
        pore_ids = border_elements.get_pores_ids()
        throat_ids = border_elements.get_throats_ids()
        if len(pore_ids) == 0 and len(throat_ids) == 0:
            return None
        etype, eid, epc = _find_max_entry_pressure(
            pore_ids,
            self.network.pn_core.pore.entry_pressure,
            throat_ids,
            self.network.pn_core.throat.entry_pressure,
        )
        if eid == -1:
            return None
        return (int(etype), int(eid), float(epc))

    def _pick_best_candidate(self, a, b):
        """Pick the candidate with the higher PC (imbibition direction)."""
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return a if a[2] >= b[2] else b

    def _clamp_pc(self, entry_pc: float, pc: float) -> float:
        """Imbibition clamps invasion PC to be at most the current PC."""
        return min(entry_pc, pc)

    def _scan_snap_off_raw(self):
        """Scan snap-off lists for the element with the highest raw entry pressure."""
        if len(self.snap_off_pore_list) == 0 and len(self.snap_off_throat_list) == 0:
            return None
        etype, eid, epc = _find_max_entry_pressure(
            self.snap_off_pore_list,
            self.network.pn_core.pore.entry_pressure,
            self.snap_off_throat_list,
            self.network.pn_core.throat.entry_pressure,
        )
        if eid == -1:
            return None
        return (int(etype), int(eid), float(epc))

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
        throat_id_list,
        throat_entry_pressure,
        verbose=False,
    ):
        max_pressure = -np.inf
        max_id = -1
        max_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            pressure = pore_entry_pressure[id]
            if pressure > max_pressure:
                max_pressure = pressure
                max_id = id
                max_type = PORE

        for id in throat_id_list:
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
        )
        element.bo[id] = calculate_imbibition_element_bo(
            element.corner_half_angle[id],
            element.max_b[id],
            element.advancing_contact_angle[id],
            element.corner_oil_layer[id],
            element.n_corners[id],
            pc,
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
