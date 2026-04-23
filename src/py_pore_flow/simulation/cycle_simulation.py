from abc import abstractmethod
import numba as nb
import numpy as np
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    calculate_hinging_contact_angle,
)
from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.clusters import PhaseConnectionChecker
from py_pore_flow.simulation.krel_calculator import KrelCalculator
from py_pore_flow.simulation.trap_checker import TrapChecker
from py_pore_flow.simulation.border_elements import BorderElements


class CycleSimulation:
    def __init__(self, network, phase, calculate_full_conductance=False, batch_invasions="none"):
        self.network = network
        self.phase = phase
        self.batch_invasions = batch_invasions

        self.krel_calculator = KrelCalculator(self.network, calculate_full_conductance)

        self.phase_connection_checker = PhaseConnectionChecker(network, self.phase)
        self.phase_connection_checker.reset_phase_connections()

        n_pores = network.get_number_of_pores()
        n_throats = network.get_number_of_throats()

        self.new_border_pores_array = np.empty(n_pores, np.int64)
        self.new_border_throats_array = np.empty(n_throats, np.int64)
        self.old_border_pores_array = np.empty(n_pores, np.int64)
        self.old_border_throats_array = np.empty(n_throats, np.int64)

        self.trap_checker = TrapChecker(network, self.phase)

        self.border_elements = BorderElements(
            n_throats=n_throats,
            n_pores=n_pores
        )
        self._refresh_border_elements(True, None, None)

        # First trap-check may mark pores already in the border as trapped;
        # refresh again so the border never contains trapped elements.  Without
        # this, the batch seeder (which correctly rejects trapped) can produce
        # an empty layer while _get_element_to_fill still picks the trapped
        # border pore, and update_pc spins forever at the same pc.
        self.trap_checker.update_trapped_pores()
        self._refresh_border_elements(True, None, None)

        self.saturation_calculator = None

        self.current_pc = None

        # Pre-allocated work buffers for batch invasion modes.
        if batch_invasions != "none":
            self._batch_pore_queue = np.empty(n_pores, np.int64)
            self._batch_throat_queue = np.empty(n_throats, np.int64)
            self._batch_pore_visited = np.zeros(n_pores, np.bool_)
            self._batch_throat_visited = np.zeros(n_throats, np.bool_)
        if batch_invasions == "partial":
            self._batch_next_pore_queue = np.empty(n_pores, np.int64)
            self._batch_next_throat_queue = np.empty(n_throats, np.int64)

    def _get_batch_seed_ids(self):
        """Return (pore_ids, throat_ids) to seed Layer-0 of batched invasions.

        By default only the current border is returned. Subclasses that track
        additional invadable elements outside the border (e.g. snap-off
        candidates in subsequent cycles) must override this to include them,
        otherwise those elements may never be picked up by the BFS and the
        caller can spin forever trying to advance Pc through them.
        """
        return (
            self.border_elements.get_pores_ids(),
            self.border_elements.get_throats_ids(),
        )

    def update_pc(self, pc, target_sw=None, verbose=False):
        invasion_pc = pc
        old_pc = pc
        total_sw_delta = 0.0
        invasion_target_sw = None

        if verbose: yield "Updating network geometry"
        self._update_network_geometry(pc)

        while True:
            if target_sw is not None:
                invasion_target_sw = target_sw - total_sw_delta
            invaded_pc = invasion_pc
            
            if verbose: yield f"Updating invasions at Pc: {invasion_pc:.2f}"
            sw_delta, invasion_pc, no_invasion_left = yield from self._update_invasions(invasion_pc, invasion_target_sw, verbose=verbose)
            
            if verbose: yield "Updating multiphase mask"
            self._update_multiphase_mask(invaded_pc, old_pc)
            old_pc = invaded_pc
            total_sw_delta += sw_delta

            if (target_sw is None) or (total_sw_delta >= target_sw) or (no_invasion_left):
                break

        if verbose: yield "Finalizing network geometry update"
        self._update_network_geometry(invaded_pc)

        return total_sw_delta, invaded_pc, invasion_pc, no_invasion_left

    def _update_multiphase_mask(self, pc, pc_old):
        pass

    def calculate_sw_total(self):
        return self.saturation_calculator.calculate_sw_total()

    def calculate_krel(self):
        return self.krel_calculator.calculate_krw(), self.krel_calculator.calculate_kro()

    def get_pores_left(self):
        if self.phase == WATER:
            return np.logical_not(self.network.pn_core.pore.water_filled | self.network.pn_core.pore.trapped).sum()
        else:
            return np.logical_not(self.network.pn_core.pore.oil_filled | self.network.pn_core.pore.trapped).sum()

    def get_throats_left(self):
        if self.phase == WATER:
            return np.logical_not(self.network.pn_core.throat.water_filled | self.network.pn_core.throat.trapped).sum()
        else:
            return np.logical_not(self.network.pn_core.throat.oil_filled | self.network.pn_core.throat.trapped).sum()

    def get_oil_filled_percentage(self):
        total_pores = self.network.get_number_of_pores()
        pores_filled = self.network.pn_core.pore.oil_filled.sum()
        return pores_filled, total_pores, (pores_filled / total_pores)

    @abstractmethod
    def _check_invasion_condition(self, pc, element):
        pass

    def _update_element_filled_geometry(self, pc, element, element_id, verbose=False):
        self._update_element_hinging_contact_angles(element.type, element_id, pc, verbose=verbose)
        return self.saturation_calculator.update_element_saturations(element.type, element_id, pc, verbose=verbose)

    def _update_network_geometry(self, pc, verbose=False):
        self.saturation_calculator.update_saturations(pc, verbose=verbose)

    @abstractmethod
    def _update_pore_entry_pc(self, pore_id, pc):
        pass

    @abstractmethod
    def _update_border_pore_entry_pc(self, pore_id, pc):
        pass

    @abstractmethod
    def _update_throat_entry_pc(self, throat_id):
        pass

    @abstractmethod
    def _update_border_throat_entry_pc(self, throat_id):
        pass

    @abstractmethod
    def _get_element_to_fill(self, pc, border_elements, verbose=False):
        pass

    @abstractmethod
    def _update_entry_pressures(self, pc, verbose=False):
        pass

    def _update_invasions(self, pc, target_sw=None, verbose=False):
        if self.batch_invasions == "full":
            return (yield from self._update_invasions_batched(pc, target_sw, verbose))
        if self.batch_invasions == "partial":
            return (yield from self._update_invasions_partial_batch(pc, target_sw, verbose))

        sw_delta = 0.0

        if verbose: yield "Updating entry pressures"
        self._update_entry_pressures(pc, verbose=verbose)

        no_invasion_left = False
        invasion_pc = pc
        
        while True:
            try:
                element_to_fill = self._get_element_to_fill(pc, self.border_elements, verbose=verbose)
            except TypeError as e:
                print(f"DEBUG: Actual class is {type(self)}")
                print(f"DEBUG: Inheritance MRO: {type(self).mro()}")
                print(e)
                raise e

            if element_to_fill is None:
                no_invasion_left = True
                break

            element_type, element_id, invasion_pc = element_to_fill

            if (target_sw is not None and sw_delta >= target_sw) or not self._can_be_filled(pc, invasion_pc):
                break

            if element_type == PORE:
                element = self.network.pore
            else:
                element = self.network.throat
                
            if verbose: yield f"Filling {'pore' if element_type == PORE else 'throat'} {element_id} at Pc: {invasion_pc:.2f}"

            self._fill_element(element, element_id, invasion_pc)
            reached_bubble = self._check_for_bubble(element, element_id)
            sw_delta += self._update_element_filled_geometry(invasion_pc, element, element_id)
            if element.is_pore():
                self.phase_connection_checker.update_phase_connections(PORE, element_id)
            else:
                self.phase_connection_checker.update_phase_connections(THROAT, element_id)
            if (self.phase == OIL and element.get_So(element_id) == 1.0) or (
                self.phase == WATER and element.get_Sw(element_id) == 1.0
            ):
                if verbose: yield "Updating trapping"
                (
                    new_trapped_pores,
                    new_trapped_throats,
                    new_untrapped_pores,
                    new_untrapped_throats,
                ) = self.trap_checker.update_trapped_pores()
                
                if verbose: yield "Refreshing border elements"
                new_border_pores, new_border_throats = self._refresh_border_elements(
                    reached_bubble, element, element_id, new_trapped_pores, new_trapped_throats
                )
                
                if verbose: yield "Updating untrapped entry pressures"
                self._update_pore_entry_pc(new_untrapped_pores, pc)
                self._update_throat_entry_pc(new_untrapped_throats)
            else:
                if verbose: yield "Refreshing border elements"
                new_border_pores, new_border_throats = self._refresh_border_elements(
                    reached_bubble, element, element_id
                )
            
            if verbose: yield "Updating border entry pressures"
            self._update_border_pore_entry_pc(new_border_pores, pc)
            self._update_border_throat_entry_pc(new_border_throats)
        return sw_delta, invasion_pc, no_invasion_left

    def _update_invasions_batched(self, pc, target_sw=None, verbose=False):
        sw_delta = 0.0

        if verbose: yield "Updating entry pressures (batch mode)"
        self._update_entry_pressures(pc, verbose=verbose)

        pc_tolerant = pc

        pn = self.network.pn_core
        seed_pore_ids, seed_throat_ids = self._get_batch_seed_ids()

        if self.phase == OIL:
            pore_filled = pn.pore.oil_filled
            throat_filled = pn.throat.oil_filled
        else:
            pore_filled = pn.pore.water_filled
            throat_filled = pn.throat.water_filled

        # Reset visited buffers.
        self._batch_pore_visited[:] = False
        self._batch_throat_visited[:] = False

        # Collect all elements to invade via BFS wavefront.
        pore_fill_count, throat_fill_count = self._njit_collect_batch_elements(
            pc_tolerant,
            self.phase,
            seed_pore_ids,
            seed_throat_ids,
            pn.pore.entry_pressure,
            pn.throat.entry_pressure,
            pore_filled,
            throat_filled,
            pn.pore.trapped,
            pn.throat.trapped,
            pn.pore.conns_values,
            pn.pore.conns_offsets,
            pn.pore.n_conns,
            pn.throat.conns_values,
            pn.throat.conns_offsets,
            self._batch_pore_queue,
            self._batch_throat_queue,
            self._batch_pore_visited,
            self._batch_throat_visited,
        )

        pores_to_fill = self._batch_pore_queue[:pore_fill_count]
        throats_to_fill = self._batch_throat_queue[:throat_fill_count]

        if verbose:
            yield f"Batch: invading {pore_fill_count} pores and {throat_fill_count} throats"

        # Invade collected elements, stopping if target_sw is reached.
        any_filled = False
        for i in range(pore_fill_count):
            if target_sw is not None and sw_delta >= target_sw:
                break
            pore_id = pores_to_fill[i]
            self._fill_element(self.network.pore, pore_id, pc)
            sw_delta += self._update_element_filled_geometry(pc, self.network.pore, pore_id)
            any_filled = True

        if not (target_sw is not None and sw_delta >= target_sw):
            for i in range(throat_fill_count):
                if target_sw is not None and sw_delta >= target_sw:
                    break
                throat_id = throats_to_fill[i]
                self._fill_element(self.network.throat, throat_id, pc)
                sw_delta += self._update_element_filled_geometry(pc, self.network.throat, throat_id)
                any_filled = True

        # Post-batch recalculations (once).
        if any_filled:
            if verbose: yield "Batch: rebuilding phase connections"
            self.phase_connection_checker.reset_phase_connections()

            if verbose: yield "Batch: updating trapping"
            self.trap_checker.update_trapped_pores()

            if verbose: yield "Batch: resetting border elements"
            new_border_pores, new_border_throats = self._refresh_border_elements(True, None, None)

            if verbose: yield "Batch: updating border entry pressures"
            self._update_border_pore_entry_pc(new_border_pores, pc)
            self._update_border_throat_entry_pc(new_border_throats)

        # Determine invasion_pc for the caller (_step_by_sw uses it to
        # advance Pc to the next candidate).  Reuse the existing
        # _get_element_to_fill which also accounts for snap-off elements.
        element_to_fill = self._get_element_to_fill(pc, self.border_elements)
        if element_to_fill is not None:
            _, _, invasion_pc = element_to_fill
            no_invasion_left = False
        else:
            invasion_pc = pc
            no_invasion_left = True

        return sw_delta, invasion_pc, no_invasion_left

    def _update_invasions_partial_batch(self, pc, target_sw=None, verbose=False):
        total_sw_delta = 0.0

        if verbose: yield "Updating entry pressures (partial batch mode)"
        self._update_entry_pressures(pc, verbose=verbose)

        # Compute tolerant Pc (1% shift in invasion direction).
        if self.phase == OIL:
            pc_tolerant = pc * 1.01 if pc > 0 else pc * 0.99
        else:
            pc_tolerant = pc * 0.99 if pc > 0 else pc * 1.01

        pn = self.network.pn_core
        if self.phase == OIL:
            pore_filled = pn.pore.oil_filled
            throat_filled = pn.throat.oil_filled
        else:
            pore_filled = pn.pore.water_filled
            throat_filled = pn.throat.water_filled

        self._batch_pore_visited[:] = False
        self._batch_throat_visited[:] = False

        # --- Layer 0: seed from current border elements (and any extras such
        # as snap-off candidates in subsequent cycles) ---
        seed_pore_ids, seed_throat_ids = self._get_batch_seed_ids()

        pore_fill_count, throat_fill_count, next_pore_count, next_throat_count = (
            self._njit_collect_single_layer(
                pc_tolerant,
                self.phase,
                seed_pore_ids,
                seed_throat_ids,
                pn.pore.entry_pressure,
                pn.throat.entry_pressure,
                pore_filled,
                throat_filled,
                pn.pore.trapped,
                pn.throat.trapped,
                pn.pore.conns_values,
                pn.pore.conns_offsets,
                pn.pore.n_conns,
                pn.throat.conns_values,
                pn.throat.conns_offsets,
                self._batch_pore_queue,
                self._batch_throat_queue,
                self._batch_pore_visited,
                self._batch_throat_visited,
                self._batch_next_pore_queue,
                self._batch_next_throat_queue,
            )
        )

        sw_limit_reached = False

        # Process layers until no more elements qualify or target_sw reached.
        while pore_fill_count > 0 or throat_fill_count > 0:
            if verbose:
                yield f"Partial batch layer: invading {pore_fill_count} pores and {throat_fill_count} throats"

            # Invade this layer, stopping if target_sw is reached.
            for i in range(pore_fill_count):
                if target_sw is not None and total_sw_delta >= target_sw:
                    sw_limit_reached = True
                    break
                pore_id = self._batch_pore_queue[i]
                self._fill_element(self.network.pore, pore_id, pc)
                total_sw_delta += self._update_element_filled_geometry(pc, self.network.pore, pore_id)

            if not sw_limit_reached:
                for i in range(throat_fill_count):
                    if target_sw is not None and total_sw_delta >= target_sw:
                        sw_limit_reached = True
                        break
                    throat_id = self._batch_throat_queue[i]
                    self._fill_element(self.network.throat, throat_id, pc)
                    total_sw_delta += self._update_element_filled_geometry(pc, self.network.throat, throat_id)

            # Post-layer recalculations.
            if verbose: yield "Partial batch: rebuilding phase connections"
            self.phase_connection_checker.reset_phase_connections()

            if verbose: yield "Partial batch: updating trapping"
            self.trap_checker.update_trapped_pores()

            if verbose: yield "Partial batch: resetting border elements"
            new_border_pores, new_border_throats = self._refresh_border_elements(True, None, None)

            if verbose: yield "Partial batch: updating border entry pressures"
            self._update_border_pore_entry_pc(new_border_pores, pc)
            self._update_border_throat_entry_pc(new_border_throats)

            if sw_limit_reached:
                break

            # --- Next layer: seed from the neighbor candidates found above ---
            if next_pore_count == 0 and next_throat_count == 0:
                break

            # Re-read filled/trapped since they changed after recalculations.
            if self.phase == OIL:
                pore_filled = pn.pore.oil_filled
                throat_filled = pn.throat.oil_filled
            else:
                pore_filled = pn.pore.water_filled
                throat_filled = pn.throat.water_filled

            next_pores = self._batch_next_pore_queue[:next_pore_count]
            next_throats = self._batch_next_throat_queue[:next_throat_count]

            # Reset visited so next-layer candidates aren't skipped.
            # pore_filled/throat_filled already prevent re-invading.
            self._batch_pore_visited[:] = False
            self._batch_throat_visited[:] = False

            pore_fill_count, throat_fill_count, next_pore_count, next_throat_count = (
                self._njit_collect_single_layer(
                    pc_tolerant,
                    self.phase,
                    next_pores,
                    next_throats,
                    pn.pore.entry_pressure,
                    pn.throat.entry_pressure,
                    pore_filled,
                    throat_filled,
                    pn.pore.trapped,
                    pn.throat.trapped,
                    pn.pore.conns_values,
                    pn.pore.conns_offsets,
                    pn.pore.n_conns,
                    pn.throat.conns_values,
                    pn.throat.conns_offsets,
                    self._batch_pore_queue,
                    self._batch_throat_queue,
                    self._batch_pore_visited,
                    self._batch_throat_visited,
                    self._batch_next_pore_queue,
                    self._batch_next_throat_queue,
                )
            )

        # Determine invasion_pc for _step_by_sw advancement.
        element_to_fill = self._get_element_to_fill(pc, self.border_elements)
        if element_to_fill is not None:
            _, _, invasion_pc = element_to_fill
            no_invasion_left = False
        else:
            invasion_pc = pc
            no_invasion_left = True

        return total_sw_delta, invasion_pc, no_invasion_left

    @staticmethod
    @nb.njit(cache=True)
    def _njit_collect_single_layer(
        pc_tolerant,
        phase,
        candidate_pore_ids,
        candidate_throat_ids,
        pore_entry_pressure,
        throat_entry_pressure,
        pore_filled,
        throat_filled,
        pore_trapped,
        throat_trapped,
        pore_conns_values,
        pore_conns_offsets,
        pore_n_conns,
        throat_conns_values,
        throat_conns_offsets,
        pore_queue,
        throat_queue,
        pore_visited,
        throat_visited,
        next_pore_queue,
        next_throat_queue,
    ):
        """Collect one layer of qualifying elements from candidates and discover
        their qualifying neighbors as the next layer.

        Returns (pore_fill_count, throat_fill_count,
                 next_pore_count, next_throat_count).

        pore_queue[:pore_fill_count]   = pore IDs to invade this layer.
        throat_queue[:throat_fill_count] = throat IDs to invade this layer.
        next_pore_queue[:next_pore_count]   = neighbor pore candidates for next layer.
        next_throat_queue[:next_throat_count] = neighbor throat candidates for next layer.
        """
        is_drainage = phase == OIL
        pore_fill_count = 0
        throat_fill_count = 0
        next_pore_count = 0
        next_throat_count = 0

        # Filter candidate pores that qualify.
        for i in range(len(candidate_pore_ids)):
            pid = candidate_pore_ids[i]
            if pore_visited[pid] or pore_filled[pid] or pore_trapped[pid]:
                continue
            ep = pore_entry_pressure[pid]
            if is_drainage:
                qualifies = pc_tolerant >= ep
            else:
                qualifies = pc_tolerant <= ep
            if qualifies:
                pore_queue[pore_fill_count] = pid
                pore_fill_count += 1
                pore_visited[pid] = True

        # Filter candidate throats that qualify.
        for i in range(len(candidate_throat_ids)):
            tid = candidate_throat_ids[i]
            if throat_visited[tid] or throat_filled[tid] or throat_trapped[tid]:
                continue
            ep = throat_entry_pressure[tid]
            if is_drainage:
                qualifies = pc_tolerant >= ep
            else:
                qualifies = pc_tolerant <= ep
            if qualifies:
                throat_queue[throat_fill_count] = tid
                throat_fill_count += 1
                throat_visited[tid] = True

        # Discover neighbors of qualifying pores → throat neighbors.
        for i in range(pore_fill_count):
            pid = pore_queue[i]
            start = pore_conns_offsets[pid]
            end = start + pore_n_conns[pid]
            for j in range(start, end):
                tid = pore_conns_values[j]
                if throat_visited[tid] or throat_filled[tid] or throat_trapped[tid]:
                    continue
                ep = throat_entry_pressure[tid]
                if is_drainage:
                    qualifies = pc_tolerant >= ep
                else:
                    qualifies = pc_tolerant <= ep
                if qualifies:
                    next_throat_queue[next_throat_count] = tid
                    next_throat_count += 1
                    throat_visited[tid] = True

        # Discover neighbors of qualifying throats → pore neighbors.
        for i in range(throat_fill_count):
            tid = throat_queue[i]
            t_start = throat_conns_offsets[tid]
            pore_0 = throat_conns_values[t_start]
            pore_1 = throat_conns_values[t_start + 1]
            for pid in (pore_0, pore_1):
                if pore_visited[pid] or pore_filled[pid] or pore_trapped[pid]:
                    continue
                ep = pore_entry_pressure[pid]
                if is_drainage:
                    qualifies = pc_tolerant >= ep
                else:
                    qualifies = pc_tolerant <= ep
                if qualifies:
                    next_pore_queue[next_pore_count] = pid
                    next_pore_count += 1
                    pore_visited[pid] = True

        return pore_fill_count, throat_fill_count, next_pore_count, next_throat_count

    @staticmethod
    @nb.njit(cache=True)
    def _njit_collect_batch_elements(
        pc_tolerant,
        phase,
        border_pore_ids,
        border_throat_ids,
        pore_entry_pressure,
        throat_entry_pressure,
        pore_filled,
        throat_filled,
        pore_trapped,
        throat_trapped,
        pore_conns_values,
        pore_conns_offsets,
        pore_n_conns,
        throat_conns_values,
        throat_conns_offsets,
        pore_queue,
        throat_queue,
        pore_visited,
        throat_visited,
    ):
        """BFS wavefront collection of elements to invade in batch mode.

        Returns (pore_fill_count, throat_fill_count).  The first
        pore_fill_count entries of pore_queue and throat_fill_count entries
        of throat_queue contain the element IDs to invade.
        """
        is_drainage = phase == OIL
        pore_head = 0
        pore_tail = 0
        throat_head = 0
        throat_tail = 0

        # Seed from border pores.
        for i in range(len(border_pore_ids)):
            pid = border_pore_ids[i]
            if pore_visited[pid]:
                continue
            ep = pore_entry_pressure[pid]
            if is_drainage:
                qualifies = pc_tolerant >= ep
            else:
                qualifies = pc_tolerant <= ep
            if qualifies:
                pore_queue[pore_tail] = pid
                pore_tail += 1
                pore_visited[pid] = True

        # Seed from border throats.
        for i in range(len(border_throat_ids)):
            tid = border_throat_ids[i]
            if throat_visited[tid]:
                continue
            ep = throat_entry_pressure[tid]
            if is_drainage:
                qualifies = pc_tolerant >= ep
            else:
                qualifies = pc_tolerant <= ep
            if qualifies:
                throat_queue[throat_tail] = tid
                throat_tail += 1
                throat_visited[tid] = True

        # BFS: alternate between pore and throat queues until both are empty.
        while pore_head < pore_tail or throat_head < throat_tail:
            # Process pore queue → discover throat neighbors.
            while pore_head < pore_tail:
                pid = pore_queue[pore_head]
                pore_head += 1
                start = pore_conns_offsets[pid]
                end = start + pore_n_conns[pid]
                for j in range(start, end):
                    tid = pore_conns_values[j]
                    if throat_visited[tid] or throat_filled[tid] or throat_trapped[tid]:
                        continue
                    ep = throat_entry_pressure[tid]
                    if is_drainage:
                        qualifies = pc_tolerant >= ep
                    else:
                        qualifies = pc_tolerant <= ep
                    if qualifies:
                        throat_queue[throat_tail] = tid
                        throat_tail += 1
                        throat_visited[tid] = True

            # Process throat queue → discover pore neighbors.
            while throat_head < throat_tail:
                tid = throat_queue[throat_head]
                throat_head += 1
                t_start = throat_conns_offsets[tid]
                pore_0 = throat_conns_values[t_start]
                pore_1 = throat_conns_values[t_start + 1]
                for pid in (pore_0, pore_1):
                    if pore_visited[pid] or pore_filled[pid] or pore_trapped[pid]:
                        continue
                    ep = pore_entry_pressure[pid]
                    if is_drainage:
                        qualifies = pc_tolerant >= ep
                    else:
                        qualifies = pc_tolerant <= ep
                    if qualifies:
                        pore_queue[pore_tail] = pid
                        pore_tail += 1
                        pore_visited[pid] = True

        return pore_tail, throat_tail

    def _check_for_bubble(self, element, element_id):

        if element.type == PORE:
            element_data = self.network.pn_core.pore
            neighbor_data = self.network.pn_core.throat
        else:
            element_data = self.network.pn_core.throat
            neighbor_data = self.network.pn_core.pore

        if self.phase == WATER:
            element_saturation = element_data.water_saturation
            neighbor_saturation = neighbor_data.water_saturation
            connected_to_inlet = neighbor_data.water_connected_to_inlet
        else:
            element_saturation = element_data.oil_saturation
            neighbor_saturation = neighbor_data.oil_saturation
            connected_to_inlet = neighbor_data.oil_connected_to_inlet

        if element_saturation[element_id] > 0:
            return False

        neighbor_ids = element_data.conns[element_id][: element_data.n_conns[element_id]]
        if ((neighbor_saturation[neighbor_ids] > 0) & ~connected_to_inlet[neighbor_ids]).any():
            return True
        else:
            return False

    def _refresh_border_elements(
        self,
        reached_bubble,
        filled_element,
        filled_element_id,
        new_trapped_pores=np.empty(0, np.int64),
        new_trapped_throats=np.empty(0, np.int64),
    ):
        if reached_bubble:
            new_pores, new_throats = self._reset_border_elements(
                self.new_border_pores_array,
                self.new_border_throats_array,
                self.network.pn_core,
                self.border_elements,
                self.phase,
            )
        else:
            new_pores, new_throats = self._update_border_elements(
                self.new_border_pores_array,
                self.new_border_throats_array,
                self.old_border_pores_array,
                self.old_border_throats_array,
                self.network.pn_core,
                self.border_elements,
                self.phase,
                filled_element,
                filled_element_id,
                new_trapped_pores,
                new_trapped_throats,
            )

        return new_pores, new_throats

    def _fill_element(self, element, element_id, pc):
        element.set_filled(element_id, self.phase, True)
        if element.type == PORE:
            self.network.pn_core.pore.trapped[element_id] = False
            self.network.pn_core.pore.trapped_evacuation[element_id] = False
        else:
            self.network.pn_core.throat.trapped[element_id] = False
            self.network.pn_core.throat.trapped_evacuation[element_id] = False

    def _update_element_hinging_contact_angles(self, element_type, id, pc, verbose=False):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat
        calculate_hinging_contact_angle(
            pc,
            element_data.corner_half_angle[id],
            element_data.b[id],
            element_data.n_corners[id],
            element_data.receding_contact_angle[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
            verbose=verbose,
        )

    @staticmethod
    def _reset_border_elements(
        new_border_pores,
        new_border_throats,
        pn,
        border_elements,
        phase,
    ):
        border_pore_ids, border_throat_ids = CycleSimulation._njit_reset_border_elements(
            new_border_pores,
            new_border_throats,
            phase,
            pn.pore.water_filled,
            pn.pore.oil_filled,
            pn.pore.water_connected_to_inlet,
            pn.pore.oil_connected_to_inlet,
            pn.pore.trapped,
            pn.pore.inlets,
            pn.pore.conns_values,
            pn.pore.conns_offsets,
            pn.pore.n_conns,
            pn.pore.in_border,
            pn.throat.water_filled,
            pn.throat.oil_filled,
            pn.throat.water_connected_to_inlet,
            pn.throat.oil_connected_to_inlet,
            pn.throat.trapped,
            pn.throat.conns_values,
            pn.throat.conns_offsets,
            pn.throat.n_conns,
            pn.throat.in_border,
        )

        if len(border_throat_ids) > 0:
            new_throats = np.setdiff1d(border_throat_ids, border_elements.get_throats_ids())
            border_elements.set_throats_ids(border_throat_ids)
        else:
            new_throats = np.array([], np.int64)
            border_elements.set_throats_ids(np.array([], np.int64))
        if len(border_pore_ids) > 0:
            new_pores = np.setdiff1d(border_pore_ids, border_elements.get_pores_ids())
            border_elements.set_pores_ids(border_pore_ids)
        else:
            new_pores = np.array([], np.int64)
            border_elements.set_pores_ids(np.array([], np.int64))

        return new_pores, new_throats

    @staticmethod
    def _update_border_elements(
        new_border_pores_array,
        new_border_throats_array,
        old_border_pores_array,
        old_border_throats_array,
        pn,
        border_elements,
        phase,
        filled_element,
        filled_element_id,
        new_trapped_pores,
        new_trapped_throats,
    ):
        new_border_pores, new_border_throats, old_pores, old_throats = CycleSimulation._njit_update_border_elements(
            new_border_pores_array,
            new_border_throats_array,
            old_border_pores_array,
            old_border_throats_array,
            phase,
            filled_element.type,
            filled_element_id,
            pn.pore.trapped,
            pn.pore.conns_values,
            pn.pore.conns_offsets,
            pn.pore.n_conns,
            pn.pore.water_filled,
            pn.pore.oil_filled,
            pn.pore.in_border,
            pn.throat.trapped,
            pn.throat.conns_values,
            pn.throat.conns_offsets,
            pn.throat.n_conns,
            pn.throat.water_filled,
            pn.throat.oil_filled,
            pn.throat.in_border,
            new_trapped_pores,
            new_trapped_throats,
        )

        border_elements.add_pores_ids(new_border_pores)
        border_elements.add_throats_ids(new_border_throats)
        border_elements.remove_pores_ids(old_pores)
        border_elements.remove_throats_ids(old_throats)

        return new_border_pores, new_border_throats

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_border_elements(
        new_border_pores_array,
        new_border_throats_array,
        old_border_pores_array,
        old_border_throats_array,
        phase,
        filled_element_type,
        filled_element_id,
        pore_trapped,
        pore_conns_values,
        pore_conns_offsets,
        pore_n_conns,
        pore_water_filled,
        pore_oil_filled,
        pore_in_border,
        throat_trapped,
        throat_conns_values,
        throat_conns_offsets,
        throat_n_conns,
        throat_water_filled,
        throat_oil_filled,
        throat_in_border,
        new_trapped_pores,
        new_trapped_throats,
    ):
        # setup
        if filled_element_type == PORE:
            element_conns_values = pore_conns_values
            element_conns_offsets = pore_conns_offsets
            element_n_conns = pore_n_conns
            neighbor_trapped = throat_trapped
            neighbor_water_filled = throat_water_filled
            neighbor_oil_filled = throat_oil_filled
            neighbor_in_border = throat_in_border
            new_border_neighbors = new_border_throats_array
        else:
            element_conns_values = throat_conns_values
            element_conns_offsets = throat_conns_offsets
            element_n_conns = throat_n_conns
            neighbor_trapped = pore_trapped
            neighbor_water_filled = pore_water_filled
            neighbor_oil_filled = pore_oil_filled
            neighbor_in_border = pore_in_border
            new_border_neighbors = new_border_pores_array

        if phase == WATER:
            neighbor_filled = neighbor_water_filled
        else:
            neighbor_filled = neighbor_oil_filled

        # pores to add
        start = element_conns_offsets[filled_element_id]
        end = start + element_n_conns[filled_element_id]
        neighbor_ids = element_conns_values[start:end].astype(np.int64)
        new_neighbors_count = 0
        for id in neighbor_ids:
            if not neighbor_filled[id] and not neighbor_trapped[id] and not neighbor_in_border[id]:
                new_border_neighbors[new_neighbors_count] = id
                new_neighbors_count += 1
                neighbor_in_border[id] = True

        if filled_element_type == PORE:
            new_pores = np.empty(0, np.int64)
            new_throats = new_border_neighbors[:new_neighbors_count]
        else:
            new_pores = new_border_neighbors[:new_neighbors_count]
            new_throats = np.empty(0, np.int64)

        # pores to remove
        old_pores_count = 0
        for pore_id in new_trapped_pores:
            if pore_in_border[pore_id]:
                old_border_pores_array[old_pores_count] = pore_id
                old_pores_count += 1
                pore_in_border[pore_id] = False
        old_throats_count = 0
        for throat_id in new_trapped_throats:
            if throat_in_border[throat_id]:
                old_border_throats_array[old_throats_count] = throat_id
                old_throats_count += 1
                throat_in_border[throat_id] = False
        if filled_element_type == PORE:
            if pore_in_border[filled_element_id]:
                old_border_pores_array[old_pores_count] = filled_element_id
                old_pores_count += 1
                pore_in_border[filled_element_id] = False
        else:
            if throat_in_border[filled_element_id]:
                old_border_throats_array[old_throats_count] = filled_element_id
                old_throats_count += 1
                throat_in_border[filled_element_id] = False

        return (
            new_pores,
            new_throats,
            old_border_pores_array[:old_pores_count],
            old_border_throats_array[:old_throats_count],
        )

    @staticmethod
    @nb.njit(cache=True)
    def _njit_reset_border_elements(
        new_border_pores,
        new_border_throats,
        phase,
        pore_water_filled,
        pore_oil_filled,
        pore_water_connected_to_inlet,
        pore_oil_connected_to_inlet,
        pore_trapped,
        pore_inlets,
        pore_conns_values,
        pore_conns_offsets,
        pore_n_conns,
        pore_in_border,
        throat_water_filled,
        throat_oil_filled,
        throat_water_connected_to_inlet,
        throat_oil_connected_to_inlet,
        throat_trapped,
        throat_conns_values,
        throat_conns_offsets,
        throat_n_conns,
        throat_in_border,
    ):
        if phase == WATER:
            pore_phase_filled = pore_water_filled
            throat_phase_filled = throat_water_filled
            pore_phase_connected_to_inlet = pore_water_connected_to_inlet
            throat_phase_connected_to_inlet = throat_water_connected_to_inlet
        else:
            pore_phase_filled = pore_oil_filled
            throat_phase_filled = throat_oil_filled
            pore_phase_connected_to_inlet = pore_oil_connected_to_inlet
            throat_phase_connected_to_inlet = throat_oil_connected_to_inlet

        # Normal border elements
        border_pore_count = 0
        for pore_id in range(len(pore_phase_filled)):
            is_in_border = False
            if (not pore_phase_filled[pore_id]) and not pore_trapped[pore_id]:
                p_start = pore_conns_offsets[pore_id]
                p_end = p_start + pore_n_conns[pore_id]
                for throat_id in pore_conns_values[p_start:p_end]:
                    if throat_phase_connected_to_inlet[throat_id] and throat_phase_filled[throat_id]:
                        new_border_pores[border_pore_count] = pore_id
                        border_pore_count += 1
                        is_in_border = True
                        break
            pore_in_border[pore_id] = is_in_border

        border_throat_count = 0
        for throat_id in range(len(throat_phase_filled)):
            is_in_border = False
            if (not throat_phase_filled[throat_id]) and not throat_trapped[throat_id]:
                t_start = throat_conns_offsets[throat_id]
                t_end = t_start + throat_n_conns[throat_id]
                for pore_id in throat_conns_values[t_start:t_end]:
                    if pore_phase_connected_to_inlet[pore_id] and pore_phase_filled[pore_id]:
                        new_border_throats[border_throat_count] = throat_id
                        border_throat_count += 1
                        is_in_border = True
                        break
            throat_in_border[throat_id] = is_in_border

        # inlets
        for pore_id in range(len(pore_inlets)):
            if pore_inlets[pore_id]:
                if not pore_phase_filled[pore_id] and not pore_trapped[pore_id]:
                    new_border_pores[border_pore_count] = pore_id
                    border_pore_count += 1
                    pore_in_border[pore_id] = True

        border_pore_ids = np.unique(new_border_pores[:border_pore_count])
        border_throat_ids = np.unique(new_border_throats[:border_throat_count])

        return border_pore_ids, border_throat_ids

    def _check_for_inconsistent_elements(self):
        wrong_pores = np.where(
            (self.network.pn_core.pore.water_filled & (self.network.pn_core.pore.water_saturation == 0))
            | (self.network.pn_core.pore.oil_filled & (self.network.pn_core.pore.oil_saturation == 0))
        )[0]
        wrong_throats = np.where(
            (self.network.pn_core.throat.water_filled & (self.network.pn_core.throat.water_saturation == 0))
            | (self.network.pn_core.throat.oil_filled & (self.network.pn_core.throat.oil_saturation == 0))
        )[0]
        if len(wrong_pores) > 0 or len(wrong_throats) > 0:
            print("Wrong pores", wrong_pores)
            print("Wrong throats", wrong_throats)
