import numba as nb
import numpy as np

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    OIL,
    check_water_layer,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
    calculate_multiphase_mask_drainage,
)
from py_pore_flow.network.network import PORE, THROAT
from py_pore_flow.simulation.cycle_simulation import CycleSimulation
from py_pore_flow.simulation.saturation_calculator import SaturationCalculator


class DrainageSimulation(CycleSimulation):
    def __init__(self, network, calculate_full_conductance=False, batch_invasions=False):
        super().__init__(network, OIL, calculate_full_conductance, batch_invasions=batch_invasions)
        self.saturation_calculator = DrainageSaturation(network)

    def _update_multiphase_mask(self, pc, pc_old):
        calculate_multiphase_mask_drainage(
            self.network.pn_core.pore.is_multiphase,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.oil_filled,
        )
        calculate_multiphase_mask_drainage(
            self.network.pn_core.throat.is_multiphase,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.oil_filled,
        )
        super()._update_multiphase_mask(pc, pc_old)

    def _update_entry_pressures(self, pc, verbose=False):
        # Entry pressures in drainage are constant, so no need to update them.
        pass

    def _update_pore_entry_pc(self, pore_id, pc):
        pass

    def _update_border_pore_entry_pc(self, pore_id, pc):
        pass

    def _update_throat_entry_pc(self, throat_id):
        pass

    def _update_border_throat_entry_pc(self, throat_id):
        pass

    def _get_element_to_fill(self, pc, border_elements, verbose=False):
        element_to_fill = self._jit_pore_throats_element_to_fill(
            pc,
            border_elements.get_pores_ids(),
            self.network.pn_core.pore.entry_pressure,
            border_elements.get_throats_ids(),
            self.network.pn_core.throat.entry_pressure,
            verbose=verbose,
        )
        return element_to_fill

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
        min_pressure = np.inf
        min_id = -1
        min_type = -1

        if len(pore_id_list) == 0 and len(throat_id_list) == 0:
            return None

        for id in pore_id_list:
            pressure = pore_entry_pressure[id]
            if pressure < min_pressure:
                min_pressure = pressure
                min_id = id
                min_type = PORE

        for id in throat_id_list:
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

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_element_filled_geometry(self, pc, element, element_id, verbose=False):
        element.set_corner_water_layer(
            element_id,
            check_water_layer(
                element.get_corner_oil_layer(element_id),
                element.get_oil_filled(element_id),
            ),
        )
        self._update_element_hinging_contact_angles(element.type, element_id, pc, verbose=verbose)
        return super()._update_element_filled_geometry(pc, element, element_id, verbose=verbose)


class DrainageSaturation(SaturationCalculator):
    def __init__(self, network):
        super().__init__(network, OIL)

    def _get_pores_saturations(self, pc, verbose=False):
        return calculate_drainage_network_saturation(
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.C_r,
            pc,
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
            verbose=verbose,
        )

    def _get_throats_saturations(self, pc, verbose=False):
        return calculate_drainage_network_saturation(
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.C_r,
            pc,
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
            verbose=verbose,
        )

    def _get_element_saturations(self, element_type, id, pc, verbose=False):
        if element_type == PORE:
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
            verbose=verbose,
        )
