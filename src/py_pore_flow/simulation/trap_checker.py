import numba as nb
import numpy as np
from py_pore_flow.equations.geometry import WATER

ENABLE_TRAPPING_FROM_INLET = True


class TrapChecker:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase
        self.network.pn_core.pore.trapped[:] = False
        self.network.pn_core.throat.trapped[:] = False
        self.network.pn_core.pore.trapped_evacuation[:] = False
        self.network.pn_core.throat.trapped_evacuation[:] = False
        self.pore_new_trapped = np.empty(len(self.network.pn_core.pore.trapped), np.int64)
        self.throat_new_trapped = np.empty(len(self.network.pn_core.throat.trapped), np.int64)
        self.pore_new_untrapped = np.empty(len(self.network.pn_core.pore.trapped), np.int64)
        self.throat_new_untrapped = np.empty(len(self.network.pn_core.throat.trapped), np.int64)

    def update_trapped_pores(self):
        if self.phase == WATER:
            evading_filled_pore = self.network.pn_core.pore.oil_filled
            evading_connected_to_outlet_pore = self.network.pn_core.pore.oil_connected_to_outlet
            if not ENABLE_TRAPPING_FROM_INLET:
                evading_connected_to_outlet_pore = (
                    evading_connected_to_outlet_pore | self.network.pn_core.pore.oil_connected_to_inlet
                )

            new_trapped_pores, new_untrapped_pores = self._njit_update_trapped_elements(
                self.pore_new_trapped,
                self.pore_new_untrapped,
                self.network.pn_core.pore.trapped,
                self.network.pn_core.pore.trapped_evacuation,
                evading_filled_pore,
                evading_connected_to_outlet_pore,
                self.network.pn_core.pore.water_connected_to_inlet,
            )

            evading_filled_throat = self.network.pn_core.throat.oil_filled
            evading_connected_to_outlet_throat = self.network.pn_core.throat.oil_connected_to_outlet
            if not ENABLE_TRAPPING_FROM_INLET:
                evading_connected_to_outlet_throat = (
                    evading_connected_to_outlet_throat | self.network.pn_core.throat.oil_connected_to_inlet
                )

            new_trapped_throats, new_untrapped_throats = self._njit_update_trapped_elements(
                self.throat_new_trapped,
                self.throat_new_untrapped,
                self.network.pn_core.throat.trapped,
                self.network.pn_core.throat.trapped_evacuation,
                evading_filled_throat,
                evading_connected_to_outlet_throat,
                self.network.pn_core.throat.water_connected_to_inlet,
            )
        else:
            evading_filled_pore = self.network.pn_core.pore.water_filled
            evading_connected_to_outlet_pore = self.network.pn_core.pore.water_connected_to_outlet
            if not ENABLE_TRAPPING_FROM_INLET:
                evading_connected_to_outlet_pore = (
                    evading_connected_to_outlet_pore | self.network.pn_core.pore.water_connected_to_inlet
                )

            new_trapped_pores, new_untrapped_pores = self._njit_update_trapped_elements(
                self.pore_new_trapped,
                self.pore_new_untrapped,
                self.network.pn_core.pore.trapped,
                self.network.pn_core.pore.trapped_evacuation,
                evading_filled_pore,
                evading_connected_to_outlet_pore,
                self.network.pn_core.pore.oil_connected_to_inlet,
            )

            evading_filled_throat = self.network.pn_core.throat.water_filled
            evading_connected_to_outlet_throat = self.network.pn_core.throat.water_connected_to_outlet
            if not ENABLE_TRAPPING_FROM_INLET:
                evading_connected_to_outlet_throat = (
                    evading_connected_to_outlet_throat | self.network.pn_core.throat.water_connected_to_inlet
                )

            new_trapped_throats, new_untrapped_throats = self._njit_update_trapped_elements(
                self.throat_new_trapped,
                self.throat_new_untrapped,
                self.network.pn_core.throat.trapped,
                self.network.pn_core.throat.trapped_evacuation,
                evading_filled_throat,
                evading_connected_to_outlet_throat,
                self.network.pn_core.throat.oil_connected_to_inlet,
            )
        return new_trapped_pores, new_trapped_throats, new_untrapped_pores, new_untrapped_throats
        

    @staticmethod
    @nb.njit(cache=True)
    def _njit_update_trapped_elements(
        new_trapped,
        new_untrapped,
        element_trapped,
        element_trapped_evacuation,
        evading_filled,
        evading_connected_to_outlet,
        invading_connected_to_inlet,
    ):
        new_trapped_count = 0
        new_untrapped_count = 0
        for element_id in range(len(element_trapped)):
            new_trapped_status = evading_filled[element_id] and not evading_connected_to_outlet[element_id]
            if not element_trapped[element_id] and new_trapped_status:
                new_trapped[new_trapped_count] = element_id
                new_trapped_count += 1
            element_trapped[element_id] = new_trapped_status
            new_trapped_evacuation = element_trapped[element_id] or not invading_connected_to_inlet[element_id]
            if element_trapped_evacuation[element_id] and not new_trapped_evacuation:
                new_untrapped[new_untrapped_count] = element_id
                new_untrapped_count += 1
            element_trapped_evacuation[element_id] = new_trapped_evacuation
        return new_trapped[:new_trapped_count], new_untrapped[:new_untrapped_count]
