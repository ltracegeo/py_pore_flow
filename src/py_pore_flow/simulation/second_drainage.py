import numba as nb
import numpy as np

from py_pore_flow.equations.geometry import (
    CIRCLE,
    OIL,
    calculate_imbibition_element_b,
    calculate_imbibition_element_bo,
    calculate_imbibition_network_b,
    calculate_imbibition_network_bo,
    calculate_network_hinging_contact_angle,
    calculate_oil_layer,
    calculate_oil_layer_network,
    calculate_r,
    check_water_layer_network,
)
import py_pore_flow.constants as c
from py_pore_flow.equations.entry_pressure import calculate_imbibition_pc, PORE_BODY_FILLING_STRATEGY
from py_pore_flow.simulation.cycle_simulation import PORE, THROAT
from py_pore_flow.simulation.subsequent_cycle import SubsequentCycleSimulation
from py_pore_flow import log


class SecondDrainageSimulation(SubsequentCycleSimulation):
    def __init__(self, network, pc_max, calculate_full_conductance=False):
        super().__init__(network, pc_max, OIL, calculate_full_conductance)

        if c.pore_fill_alg == "oren2":
            self.calculate_pore_body_filling_pc = PORE_BODY_FILLING_STRATEGY["oren1"]

    def _can_be_filled(self, current_pc, invasion_pc):
        return current_pc >= invasion_pc

    def _update_snap_offs_pcs(self, pc):
        throat_snap_off_viability = np.empty(len(self.network.pn.throats()), np.bool_)
        for i in range(len(self.network.pn.throats())):
            throat_snap_off_viability[i] = self.network.pn_core.throat.corner_oil_layer[i][
                : self.network.pn_core.throat.n_conns[i]
            ].any()
        self.snap_off_throat_list = self._snap_off(
            self.network.pn.throats(),
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.conns,
            self.network.pn_core.throat.n_conns,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.radius,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            throat_snap_off_viability,
            self.network.pn_core.throat.entry_pressure,
            self.network.pn_core.throat.snap_off_pressure,
            self.network.pn_core.throat.trapped_evacuation,
            self.network.pn_core.pore.oil_filled,
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=True,
        )

        pore_snap_off_viability = np.empty(len(self.network.pn.pores()), np.bool_)
        for i in range(len(self.network.pn.pores())):
            pore_snap_off_viability[i] = self.network.pn_core.pore.corner_oil_layer[i][
                : self.network.pn_core.pore.n_conns[i]
            ].any()
        self.snap_off_pore_list = self._snap_off(
            self.network.pn.pores(),
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.conns,
            self.network.pn_core.pore.n_conns,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.radius,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            pore_snap_off_viability,
            self.network.pn_core.pore.entry_pressure,
            self.network.pn_core.pore.snap_off_pressure,
            self.network.pn_core.pore.trapped_evacuation,
            self.network.pn_core.throat.oil_filled,
            self.Rmin,
            pc,
            self.border_elements,
            is_throats=False,
        )
