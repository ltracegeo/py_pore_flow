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
import py_pore_flow.consts as c
from py_pore_flow.equations.entry_pressure import PORE_BODY_FILLING_STRATEGY, calculate_pore_body_filling_pc
from py_pore_flow.data_structures.sparse_set import SparseSet
from py_pore_flow.network.constants import PISTON_LIKE, PORE, PORE_BODY_FILLING, SNAP_OFF, THROAT
from py_pore_flow.simulation.cycle_simulation import CycleSimulation
from py_pore_flow.simulation.saturation_calculator import SaturationCalculator
from py_pore_flow import log


class SubsequentCycleSimulation(CycleSimulation):
    def __init__(self, network, pc_max, phase, calculate_full_conductance=False, batch_invasions=False):
        super().__init__(network, phase, calculate_full_conductance, batch_invasions=batch_invasions)
        self.saturation_calculator = SubsequentSaturation(network, self.phase)

        self.Rmin = calculate_r(pc_max, c.sigma, 100000)

        self.use_pore_body_filling = c.use_pore_fill
        self.pore_body_filling_strategy = PORE_BODY_FILLING_STRATEGY[c.pore_fill_alg]

        self.snap_off_throat_list = SparseSet(len(self.network.pn_core.throat.all))
        self.snap_off_pore_list = SparseSet(len(self.network.pn_core.pore.all))

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
        invasion_method,
        element_snap_off_pc,
        element_trapped_evacuation,
        element_trapped,
    ):
        snap_off_elements_count = 0
        for element_id in element_id_list:
            if (
                element_snap_off_viability[element_id]
                and not element_filled[element_id]
                and element_shape[element_id] != CIRCLE
                and not element_trapped_evacuation[element_id]
                and not element_trapped[element_id]
            ):
                entry_pressure[element_id] = element_snap_off_pc[element_id]
                invasion_method[element_id] = SNAP_OFF
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
            self.network.pn_core.throat.invasion_method,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.trapped,
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
            self.network.pn_core.pore.invasion_method,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.trapped,
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

    def _get_batch_seed_ids(self):
        """Seed batched invasions with border + snap-off candidates.

        Snap-off elements are interior (not on the border between filled and
        unfilled regions) but are still invadable.  If we only seed from the
        border, the BFS in ``_update_invasions_batched`` /
        ``_update_invasions_partial_batch`` can miss them entirely.  When the
        next element to fill is a snap-off candidate, ``_get_element_to_fill``
        still finds it and clamps ``invasion_pc`` to the current ``pc``,
        causing the caller's ``update_pc`` loop to spin with ``sw_delta=0``
        and ``no_invasion_left=False``.  Including snap-off ids here lets
        Layer-0 fill them and break the spin.
        """
        border_pores, border_throats = super()._get_batch_seed_ids()
        snap_off_pores = self.snap_off_pore_list.list()
        snap_off_throats = self.snap_off_throat_list.list()

        if len(snap_off_pores) == 0:
            pore_ids = border_pores
        elif len(border_pores) == 0:
            pore_ids = snap_off_pores
        else:
            pore_ids = np.concatenate((border_pores, snap_off_pores))

        if len(snap_off_throats) == 0:
            throat_ids = border_throats
        elif len(border_throats) == 0:
            throat_ids = snap_off_throats
        else:
            throat_ids = np.concatenate((border_throats, snap_off_throats))

        return pore_ids, throat_ids

    def _update_pore_entry_pc(self, pore_id_list, pc, replace_snap_off=False, verbose=False):
        if replace_snap_off:
            self.snap_off_pore_list.set_from_array(self._update_pores_snap_offs_pcs(self.network.pn.pores()))
        else:
            new_snap_off_pores = self._update_pores_snap_offs_pcs(pore_id_list)
            if len(new_snap_off_pores) > 0:
                self.snap_off_pore_list.add(new_snap_off_pores)

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
            self.network.pn_core.pore.invasion_method,
            self.network.pn_core.pore.piston_like_pressure,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.shape_factor,
            self.network.pn_core.pore.imminent_snap_off,
            self.network.pn_core.pore.conns_values,
            self.network.pn_core.pore.conns_offsets,
            self.network.pn_core.pore.n_conns,
            pore_limiting_contact_angle,
            self.network.pn_core.throat.radius,
            throat_receding_phase_filled,
            c.sigma,
            c.pore_fill_weights,
            verbose=verbose,
        )

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_pore_entry_pc(
        pore_entry_pressure,
        pore_id_list,
        pc,
        use_pore_body_filling,
        pore_body_filling_strategy,
        phase,
        pore_invasion_method,
        pore_piston_like_pressure,
        pore_radius,
        pore_shape_factor,
        pore_imminent_snap_off,
        pore_conns_values,
        pore_conns_offsets,
        pore_n_conns,
        pore_limiting_contact_angle,
        throat_radius,
        throat_receding_phase_filled,
        c_sigma,
        c_pore_fill_weights,
        verbose=False,
    ):
        for pore_id in pore_id_list:
            if not use_pore_body_filling or pc <= 0:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
                pore_invasion_method[pore_id] = PISTON_LIKE
                if verbose:
                    print("  Pore ", pore_id, " PC calculation: Piston-like = ", element_pc)
                continue

            p_start = pore_conns_offsets[pore_id]
            p_end = p_start + pore_n_conns[pore_id]
            neighbors = pore_conns_values[p_start:p_end]
            mask = throat_receding_phase_filled[neighbors]
            filled_neighbors = neighbors[mask]
            number_of_filled_neighbors = filled_neighbors.size

            if number_of_filled_neighbors <= 1:
                element_pc = pore_piston_like_pressure[pore_id]
                pore_entry_pressure[pore_id] = element_pc
                pore_invasion_method[pore_id] = PISTON_LIKE
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
                    c_sigma,
                    c_pore_fill_weights,
                )
                piston_like_pressure = pore_piston_like_pressure[pore_id]

                if phase == WATER and pore_body_filling_pressure > piston_like_pressure:
                    element_pc = piston_like_pressure
                    pore_entry_pressure[pore_id] = piston_like_pressure
                    pore_invasion_method[pore_id] = PISTON_LIKE
                elif phase == OIL and pore_body_filling_pressure < piston_like_pressure:
                    element_pc = piston_like_pressure
                    pore_entry_pressure[pore_id] = piston_like_pressure
                    pore_invasion_method[pore_id] = PISTON_LIKE
                else:
                    if pore_imminent_snap_off[pore_id]:
                        if phase == WATER:
                            if pore_body_filling_pressure > pore_entry_pressure[pore_id]:
                                element_pc = pore_body_filling_pressure
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                                pore_invasion_method[pore_id] = PORE_BODY_FILLING
                            else:
                                element_pc = pore_entry_pressure[pore_id]
                        else:
                            if pore_body_filling_pressure < pore_entry_pressure[pore_id]:
                                element_pc = pore_body_filling_pressure
                                pore_entry_pressure[pore_id] = pore_body_filling_pressure
                                pore_invasion_method[pore_id] = PORE_BODY_FILLING
                            else:
                                element_pc = pore_entry_pressure[pore_id]
                    else:
                        element_pc = pore_body_filling_pressure
                        pore_entry_pressure[pore_id] = pore_body_filling_pressure
                        pore_invasion_method[pore_id] = PORE_BODY_FILLING

                if verbose:
                    print("  Pore ", pore_id, " PC calculation: Piston-like = ", piston_like_pressure, ", Body-filling = ", pore_body_filling_pressure, " -> Selected = ", element_pc)

    def _update_throat_entry_pc(self, throat_id_list, replace_snap_off=False, verbose=False):
        if replace_snap_off:
            self.snap_off_throat_list.set_from_array(self._update_throats_snap_offs_pcs(self.network.pn.throats()))
        else:
            new_snap_off_throats = self._update_throats_snap_offs_pcs(throat_id_list)
            if len(new_snap_off_throats) > 0:
                self.snap_off_throat_list.add(new_snap_off_throats)

        self._update_border_throat_entry_pc(throat_id_list, verbose=verbose)

    def _update_border_throat_entry_pc(self, throat_id_list, verbose=False):
        self._njit_update_throat_entry_pc(
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.invasion_method,
            self.network.pn_core.throat.piston_like_pressure,
            self.network.pn_core.throat.in_border,
            verbose=verbose,
        )

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_throat_entry_pc(
        throat_id_list,
        throat_entry_pressure,
        throat_invasion_method,
        throat_piston_like_pressure,
        throat_in_border,
        verbose=False,
    ):
        for throat_id in throat_id_list:
            if throat_in_border[throat_id]:
                pressure = throat_piston_like_pressure[throat_id]
                throat_entry_pressure[throat_id] = pressure
                throat_invasion_method[throat_id] = PISTON_LIKE
                if verbose:
                    print("  Throat ", throat_id, " PC calculation: Piston-like = ", pressure)

    def _get_element_to_fill(self, pc, border_elements, verbose=False):
        self._clear_snapoff_lists()
        pore_id_list = np.concatenate((border_elements.get_pores_ids(), self.snap_off_pore_list.list()))
        throat_id_list = np.concatenate((border_elements.get_throats_ids(), self.snap_off_throat_list.list()))
        element_to_fill = self._jit_pore_throats_element_to_fill(
            pc,
            pore_id_list,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.trapped,
            throat_id_list,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.trapped,
            verbose=verbose,
        )

        return element_to_fill

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
        # `trapped` is tracked separately from `trapped_evacuation`; an element
        # flagged `trapped` after it was added to the snap-off list would
        # otherwise be picked by _get_element_to_fill and spin update_pc,
        # because the batch seeder correctly rejects trapped elements.
        pore_filled = (
            self.network.pn_core.pore.water_filled if self.phase == WATER else self.network.pn_core.pore.oil_filled
        )
        self.snap_off_pore_list.remove_where3(
            pore_filled,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.pore.trapped,
        )
        throat_filled = (
            self.network.pn_core.throat.water_filled if self.phase == WATER else self.network.pn_core.throat.oil_filled
        )
        self.snap_off_throat_list.remove_where3(
            throat_filled,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.throat.trapped,
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
