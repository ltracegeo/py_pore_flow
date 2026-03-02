import unittest

import numpy as np
import openpnm as op

from py_pore_flow.equations.geometry import WATER
from py_pore_flow.simulation.cycle_simulation import (
    BorderElements,
    CycleSimulation,
    PhaseConnectionChecker,
    TrapChecker,
)
from py_pore_flow.network import initialize_pn, Network
from tests.unit.simulation.resources.simple_network import get_simple_network


class TestCycleSimulation(unittest.TestCase):
    def test_find_initial_border_elements(self):
        pn = get_simple_network(initial_saturation="mix_filled")
        
        # Save original state for re-application after initialize_pn
        water_sat_p = pn["pore.water_saturation"].copy()
        water_sat_t = pn["throat.water_saturation"].copy()
        water_filled_p = pn["pore.water_filled"].copy()
        water_filled_t = pn["throat.water_filled"].copy()

        initialize_pn(pn)
        
        # Re-apply the specific test state
        pn["pore.water_saturation"] = water_sat_p
        pn["throat.water_saturation"] = water_sat_t
        pn["pore.water_filled"] = water_filled_p
        pn["throat.water_filled"] = water_filled_t
        
        # After update, we need to ensure all properties are synced correctly
        pn["pore.oil_saturation"] = 1.0 - pn["pore.water_saturation"]
        pn["throat.oil_saturation"] = 1.0 - pn["throat.water_saturation"]
        pn["pore.oil_filled"] = ~pn["pore.water_filled"]
        pn["throat.oil_filled"] = ~pn["throat.water_filled"]
        
        # Re-calculate is_multiphase after update because it depends on saturation
        for element_type in ["pore", "throat"]:
             pn[f"{element_type}.is_multiphase"] = (pn[f"{element_type}.water_saturation"] > 0) & (
                pn[f"{element_type}.water_saturation"] < 1.0
            )

        network = Network(pn)

        
        cicle_simluation = CycleSimulation(
            network=network, 
            phase=WATER, 
            calculate_full_conductance=False
            )
        
        cicle_simluation._refresh_border_elements(True, None, None)

        np.testing.assert_equal(cicle_simluation.border_elements.get_pores_ids(), [3, 6, 16])
        np.testing.assert_equal(cicle_simluation.border_elements.get_throats_ids(), [])

        # Trapping
        trap_checker = TrapChecker(network, WATER)
        trap_checker.update_trapped_pores()
        np.testing.assert_equal(
            pn["throat.trapped"],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        )
        np.testing.assert_equal(
            pn["pore.trapped"],
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
            ],
        )
