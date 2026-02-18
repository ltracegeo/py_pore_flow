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


class TestCycleSimulation(unittest.TestCase):
    def test_find_initial_border_elements(self):
        #  -> 0-----0----0---O---O ->
        #            \
        #             0----O
        #            /      \
        #  -> O--O--O        O---O ->
        #                   /
        #  -> 0--@--@--0--O
        #
        #  -> 0-----1----2---3---4 ----->
        #            \
        #             5-------6
        #            /         \
        #  -> 7--8--9          10---11 ->
        #                      /
        #  -> 12--13--14--15--16
        #
        #
        #  Legend:
        #  -> inlet/outlet
        #  0  pore with water in center
        #  O  pore with oil in center
        #  @  pore with oil in center and water in corners
        #  -- throat

        Np = 17
        Nt = 16
        pn_dict = {
            "throat.conns": np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [1, 5],
                    [5, 6],
                    [5, 9],
                    [6, 10],
                    [7, 8],
                    [8, 9],
                    [10, 11],
                    [16, 10],
                    [12, 13],
                    [13, 14],
                    [14, 15],
                    [15, 16],
                ]
            ),
            "pore.coords": np.array(
                [
                    [0, 0, 0],
                    [6, 0, 0],
                    [11, 0, 0],
                    [15, 0, 0],
                    [19, 0, 0],
                    [8, 2, 0],
                    [13, 2, 0],
                    [0, 4, 0],
                    [3, 4, 0],
                    [6, 4, 0],
                    [15, 4, 0],
                    [19, 4, 0],
                    [0, 6, 0],
                    [3, 6, 0],
                    [6, 6, 0],
                    [9, 6, 0],
                    [12, 6, 0],
                ]
            ),
            "pore.outlets": np.array(
                [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            ),
            "pore.inlets": np.array(
                [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                ]
            ),
            "pore.water_saturation": np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.5,
                    0.5,
                    1.0,
                    0.0,
                ]
            ),
            "throat.water_saturation": np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ),
            "pore.water_filled": np.array(
                [
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    True,
                    False,
                ]
            ),
            "throat.water_filled": np.array(
                [
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
            "throat.shape_factor": np.ones(Nt)*0.1,
            "throat.radius": np.ones(Nt),
            "throat.volume": np.ones(Nt),
            "throat.clay_volume": np.zeros(Nt),
            "throat.cross_sectional_area": np.ones(Nt),
            "throat.length": np.ones(Nt),
            "pore.shape_factor": np.ones(Np)*0.1,
            "pore.radius": np.ones(Np),
            "pore.volume": np.ones(Np),
            "pore.clay_volume": np.zeros(Np),
            "pore.area": np.ones(Np),
            "pore.length": np.ones(Np),
        }

        pn = op.network.Network()
        pn.update(pn_dict)
        initialize_pn(pn)
        pn.update(pn_dict)
        
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
