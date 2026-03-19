from abc import abstractmethod

import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    OIL,
    WATER,
    calculate_imbibition_element_saturation,
    calculate_imbibition_network_saturation,
    calculate_r,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import PORE_BODY_FILLING_STRATEGY, calculate_pore_body_filling_pc
from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.cycle_simulation import CycleSimulation
from py_pore_flow.simulation.saturation_calculator import SaturationCalculator
from py_pore_flow import log


@nb.njit(cache=True)
def _find_min_entry_pressure(
    pore_id_list, pore_entry_pressure, throat_id_list, throat_entry_pressure
):
    """Find the element with the minimum entry pressure across pores and throats.

    Returns ``(element_type, element_id, raw_pc)`` or ``(-1, -1, inf)`` if empty.
    """
    min_pc = np.inf
    min_id = np.int64(-1)
    min_type = np.int64(-1)
    for pid in pore_id_list:
        pc = pore_entry_pressure[pid]
        if pc < min_pc:
            min_pc = pc
            min_id = pid
            min_type = PORE
    for tid in throat_id_list:
        pc = throat_entry_pressure[tid]
        if pc < min_pc:
            min_pc = pc
            min_id = tid
            min_type = THROAT
    return min_type, min_id, min_pc


@nb.njit(cache=True)
def _find_max_entry_pressure(
    pore_id_list, pore_entry_pressure, throat_id_list, throat_entry_pressure
):
    """Find the element with the maximum entry pressure across pores and throats.

    Returns ``(element_type, element_id, raw_pc)`` or ``(-1, -1, -inf)`` if empty.
    """
    max_pc = -np.inf
    max_id = np.int64(-1)
    max_type = np.int64(-1)
    for pid in pore_id_list:
        pc = pore_entry_pressure[pid]
        if pc > max_pc:
            max_pc = pc
            max_id = pid
            max_type = PORE
    for tid in throat_id_list:
        pc = throat_entry_pressure[tid]
        if pc > max_pc:
            max_pc = pc
            max_id = tid
            max_type = THROAT
    return max_type, max_id, max_pc


class SubsequentCycleSimulation(CycleSimulation):
    def __init__(self, network, pc_max, phase, calculate_full_conductance=False):
        super().__init__(network, phase, calculate_full_conductance)
        self.saturation_calculator = SubsequentSaturation(network, self.phase)

        self.Rmin = calculate_r(pc_max, 100000)

        self.use_pore_body_filling = c.use_pore_fill
        self.pore_body_filling_strategy = PORE_BODY_FILLING_STRATEGY[c.pore_fill_alg]

        self.snap_off_throat_list = []
        self.snap_off_pore_list = []

        self.new_snap_off_throats_array = np.empty(len(self.network.pn_core.throat.all), np.int64)
        self.new_snap_off_pores_array = np.empty(len(self.network.pn_core.pore.all), np.int64)

        self.pores_to_update = np.empty(len(self.network.pn_core.pore.all), np.int64)
        self.throats_to_update = np.empty(len(self.network.pn_core.throat.all), np.int64)

    @staticmethod
    @nb.njit(cache=True)
    def _njit_snap_off(
        snap_off_elements,
        element_imminent_snap_off,
        element_id_list,
        element_filled,
        element_shape,
        element_snap_off_viability,
        entry_pressure,
        element_snap_off_pc,
        element_trapped,
    ):
        snap_off_elements_count = 0
        for element_id in element_id_list:
            if (
                element_snap_off_viability[element_id]
                and not element_filled[element_id]
                and element_shape[element_id] != CIRCLE
                and not element_trapped[element_id]
            ):
                entry_pressure[element_id] = element_snap_off_pc[element_id]
                snap_off_elements[snap_off_elements_count] = element_id
                snap_off_elements_count += 1
                element_imminent_snap_off[element_id] = True
            else:
                element_imminent_snap_off[element_id] = False

        return snap_off_elements[:snap_off_elements_count]

    def _update_throats_snap_offs_pcs(self, throat_ids):
        throat_filled = (
            self.network.pn_core.throat.water_filled if self.phase == WATER else self.network.pn_core.throat.oil_filled
        )
        return self._njit_snap_off(
            self.new_snap_off_throats_array,
            self.network.pn_core.throat.imminent_snap_off,
            throat_ids,
            throat_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
        )

    def _update_pores_snap_offs_pcs(self, pore_ids):
        pore_filled = (
            self.network.pn_core.pore.water_filled if self.phase == WATER else self.network.pn_core.pore.oil_filled
        )
        return self._njit_snap_off(
            self.new_snap_off_pores_array,
            self.network.pn_core.pore.imminent_snap_off,
            pore_ids,
            pore_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
        )

    def _update_water_layers(self, pore_ids, throat_ids):
        check_water_layer_network(
            self.network.pn_core.throat.corner_water_layer,
            throat_ids,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.corner_oil_layer,
            self.network.pn_core.throat.oil_filled,
        )
        check_water_layer_network(
            self.network.pn_core.pore.corner_water_layer,
            pore_ids,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.corner_oil_layer,
            self.network.pn_core.pore.oil_filled,
        )

    def _update_entry_pressures(self, pc, verbose=False):
        self._update_throat_entry_pc(self.border_elements.get_throats_ids(), True, verbose=verbose)
        self._update_pore_entry_pc(self.border_elements.get_pores_ids(), pc, True, verbose=verbose)
        # Sync tree PCs after the entry_pressure arrays were modified externally
        self.border_elements.sync_pore_pcs()
        self.border_elements.sync_throat_pcs()

    def _update_pore_entry_pc(self, pore_id_list, pc, replace_snap_off=False, verbose=False):
        if replace_snap_off:
            self.snap_off_pore_list = self._update_pores_snap_offs_pcs(self.network.pn.pores())
        else:
            new_snap_off_pores = self._update_pores_snap_offs_pcs(pore_id_list)
            if len(new_snap_off_pores) > 0:
                self.snap_off_pore_list = np.unique(np.concatenate((self.snap_off_pore_list, new_snap_off_pores)))

        self._update_border_pore_entry_pc(pore_id_list, pc, verbose=verbose)

    def _update_border_pore_entry_pc(self, pore_id_list, pc, verbose=False):
        pore_limiting_contact_angle = (
            self.network.pn_core.pore.advancing_contact_angle
            if self.phase == WATER
            else self.network.pn_core.pore.receding_contact_angle
        )
        throat_receding_phase_filled = (
            self.network.pn_core.throat.oil_filled if self.phase == WATER else self.network.pn_core.throat.water_filled
        )
        self._njit_update_pore_entry_pc(
            self.network.pn_core.pore.entry_pressure,
            pore_id_list,
            pc,
            self.use_pore_body_filling,
            self.pore_body_filling_strategy,
            self.phase,
            self.network.pn_core.pore.piston_like_pressure,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.shape_factor,
            self.network.pn_core.pore.imminent_snap_off,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            pore_limiting_contact_angle,
            self.network.pn_core.throat.radius,
            throat_receding_phase_filled,
            verbose=verbose,
        )
        # Sync tree PCs for the updated pores
        self.border_elements.sync_pore_pcs(pore_id_list)

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_pore_entry_pc(
        pore_entry_pressure,
        pore_id_list,
        pc,
        use_pore_body_filling,
        pore_body_filling_strategy,
        phase,
        pore_piston_like_pressure,
        pore_radius,
        pore_shape_factor,
        pore_imminent_snap_off,
        pore_conns,
        pore_n_conns,
        pore_limiting_contact_angle,
        throat_radius,
        throat_receding_phase_filled,
        verbose=False,
    ):
        for pore_id in pore_id_list:
            if not use_pore_body_filling or pc <= 0:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
                if verbose:
                    print("  Pore ", pore_id, " PC calculation: Piston-like = ", element_pc)
                continue

            neighbors = pore_conns[pore_id][: pore_n_conns[pore_id]]
            mask = throat_receding_phase_filled[neighbors]
            filled_neighbors = neighbors[mask]
            number_of_filled_neighbors = filled_neighbors.size

            if number_of_filled_neighbors <= 1:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
                if verbose:
                    print("  Pore ", pore_id, " PC calculation: Piston-like = ", element_pc, " (Filled neighbors <= 1)")
            else:
                throats_radii = throat_radius[filled_neighbors]

                pore_body_filling_pressure = calculate_pore_body_filling_pc(
                    pore_body_filling_strategy,
                    pore_radius[pore_id],
                    pore_limiting_contact_angle[pore_id],
                    number_of_filled_neighbors,
                    throats_radii,
                    pore_shape_factor[pore_id],
                )
                piston_like_pressure = pore_piston_like_pressure[pore_id]

                if phase == WATER and pore_body_filling_pressure > piston_like_pressure:
                    element_pc = piston_like_pressure
                    pore_entry_pressure[pore_id] = piston_like_pressure
                elif phase == OIL and pore_body_filling_pressure < piston_like_pressure:
                    element_pc = piston_like_pressure
                    pore_entry_pressure[pore_id] = piston_like_pressure
                else:
                    if pore_imminent_snap_off[pore_id]:
                        if phase == WATER:
                            if pore_body_filling_pressure > pore_entry_pressure[pore_id]:
                                element_pc = pore_body_filling_pressure
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                            else:
                                element_pc = pore_entry_pressure[pore_id]
                        else:
                            if pore_body_filling_pressure < pore_entry_pressure[pore_id]:
                                element_pc = pore_body_filling_pressure
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                            else:
                                element_pc = pore_entry_pressure[pore_id]
                    else:
                        element_pc = pore_body_filling_pressure
                        pore_entry_pressure[pore_id] = pore_body_filling_pressure

                if verbose:
                    print("  Pore ", pore_id, " PC calculation: Piston-like = ", piston_like_pressure, ", Body-filling = ", pore_body_filling_pressure, " -> Selected = ", element_pc)

    def _update_throat_entry_pc(self, throat_id_list, replace_snap_off=False, verbose=False):
        if replace_snap_off:
            self.snap_off_throat_list = self._update_throats_snap_offs_pcs(self.network.pn.throats())
        else:
            new_snap_off_throats = self._update_throats_snap_offs_pcs(throat_id_list)
            if len(new_snap_off_throats) > 0:
                self.snap_off_throat_list = np.unique(np.concatenate((self.snap_off_throat_list, new_snap_off_throats)))

        self._update_border_throat_entry_pc(throat_id_list, verbose=verbose)

    def _update_border_throat_entry_pc(self, throat_id_list, verbose=False):
        self._njit_update_throat_entry_pc(
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.piston_like_pressure,
            self.network.pn_core.throat.in_border,
            verbose=verbose,
        )
        # Sync tree PCs for the updated throats
        self.border_elements.sync_throat_pcs(throat_id_list)

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_throat_entry_pc(
        throat_id_list,
        throat_entry_pressure,
        throat_piston_like_pressure,
        throat_in_border,
        verbose=False,
    ):
        for throat_id in throat_id_list:
            if throat_in_border[throat_id]:
                pressure = throat_piston_like_pressure[throat_id]
                throat_entry_pressure[throat_id] = pressure
                if verbose:
                    print("  Throat ", throat_id, " PC calculation: Piston-like = ", pressure)

    def _get_element_to_fill(self, pc, border_elements, verbose=False):
        """Find the best border or snap-off element to fill.

        Uses O(log N) tree peek for border elements and linear scan for
        snap-off lists (which are typically much smaller than border elements).
        """
        self._clear_snapoff_lists()

        # O(log N) peek from the border element trees
        border_result = self._get_border_candidate(border_elements)

        # Linear scan of snap-off lists for raw (unclamped) best candidate
        snap_off_result = self._scan_snap_off_raw()

        # Pick the better candidate (both use raw PCs for fair comparison)
        best = self._pick_best_candidate(border_result, snap_off_result)
        if best is None:
            return None

        element_type, element_id, entry_pc = best
        invasion_pc = self._clamp_pc(entry_pc, pc)
        if verbose:
            type_name = "pore" if element_type == PORE else "throat"
            log.info(f"  Selected element: {type_name} {element_id} with PC: {entry_pc}")
        return (element_type, element_id, invasion_pc)

    @abstractmethod
    def _get_border_candidate(self, border_elements):
        """Get the best border element candidate. Subclasses implement for min/max."""
        pass

    @abstractmethod
    def _pick_best_candidate(self, a, b):
        """Pick the better of two candidates. Subclasses implement for min/max comparison."""
        pass

    @abstractmethod
    def _clamp_pc(self, entry_pc: float, pc: float) -> float:
        """Clamp invasion PC. Subclasses implement for min/max direction."""
        pass

    @abstractmethod
    def _scan_snap_off_raw(self):
        """Scan snap-off lists for the best candidate with raw (unclamped) PC.

        Returns ``(element_type, element_id, raw_pc)`` or ``None``.
        """
        pass

    @abstractmethod
    def _can_be_filled(self, current_pc, invasion_pc):
        pass

    def _update_network_geometry(self, pc):
        pore_ids = self._get_elements_to_update(
            self.pores_to_update,
            self.phase,
            pc,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.imminent_snap_off,
            self.network.pn_core.pore.snap_off_pressure,
        )
        throat_ids = self._get_elements_to_update(
            self.throats_to_update,
            self.phase,
            pc,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.imminent_snap_off,
            self.network.pn_core.throat.snap_off_pressure,
        )
        self._update_hinging_contact_angles(pore_ids, throat_ids, pc)
        self._update_b(pore_ids, throat_ids, pc)
        self._update_oil_layers(pore_ids, throat_ids, pc)
        self._update_water_layers(pore_ids, throat_ids)
        super()._update_network_geometry(pc)

    @staticmethod
    @nb.njit(cache=True)
    def _get_elements_to_update(
        elements_to_update, phase, pc, trapped_evacuation, imminent_snap_off, snap_off_pressure
    ):
        j = 0
        if phase == WATER:
            for i in range(len(trapped_evacuation)):
                if not trapped_evacuation[i] and not (imminent_snap_off[i] and (snap_off_pressure[i] >= pc)):
                    elements_to_update[j] = i
                    j += 1
        else:
            for i in range(len(trapped_evacuation)):
                if not trapped_evacuation[i] and not (imminent_snap_off[i] and (snap_off_pressure[i] <= pc)):
                    elements_to_update[j] = i
                    j += 1
        return elements_to_update[:j]

    def _clear_snapoff_lists(self):
        filled_pores = (
            self.network.pn_core.pore.water_filled[self.snap_off_pore_list]
            if self.phase == WATER
            else self.network.pn_core.pore.oil_filled[self.snap_off_pore_list]
        )
        self.snap_off_pore_list = np.delete(
            self.snap_off_pore_list,
            filled_pores | self.network.pn_core.pore.trapped_evacuation[self.snap_off_pore_list],
        )
        filled_throats = (
            self.network.pn_core.throat.water_filled[self.snap_off_throat_list]
            if self.phase == WATER
            else self.network.pn_core.throat.oil_filled[self.snap_off_throat_list]
        )
        self.snap_off_throat_list = np.delete(
            self.snap_off_throat_list,
            filled_throats | self.network.pn_core.throat.trapped_evacuation[self.snap_off_throat_list],
        )

    @abstractmethod
    def _update_hinging_contact_angles(self, pore_ids, throat_ids, pc):
        pass

    @abstractmethod
    def _update_b(self, pore_ids, throat_ids, pc):
        pass

    @abstractmethod
    def _update_oil_layers(self, pore_ids, throat_ids, pc):
        pass


class SubsequentSaturation(SaturationCalculator):
    def __init__(self, network, phase):
        super().__init__(network, phase)

    def _get_pores_saturations(self, pc, verbose=False):
        return calculate_imbibition_network_saturation(
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.corner_oil_layer,
            verbose=verbose,
        )

    def _get_throats_saturations(self, pc, verbose=False):
        return calculate_imbibition_network_saturation(
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.corner_oil_layer,
            verbose=verbose,
        )

    def _get_element_saturations(self, element_type, id, pc, verbose=False):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        return calculate_imbibition_element_saturation(
            area,
            element_data.oil_filled[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
            element_data.corner_half_angle[id],
            element_data.b[id],
            element_data.bo[id],
            element_data.n_corners[id],
            element_data.corner_oil_layer[id],
            verbose=verbose,
        )
