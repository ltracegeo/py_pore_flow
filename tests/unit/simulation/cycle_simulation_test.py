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
        return
        #  -> 0-----0----0---O---O ->
        #            \
        #             0----O
        #            /      \
        #  -> O--O--O        O---O ->
        #                   /
        #  -> 0--@--@--0--O
        #
        #
        #  Legend:
        #  -> inlet/outlet
        #  0  pore with water in center
        #  O  pore with oil in center
        #  @  pore with oil in center and water in corners
        #  -- throat

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
            "throat.shape_factor": np.array([]),
            "throat.radius": np.array([]),
            "throat.volume": np.array([]),
            "throat.clay_volume": np.array([]),
            "throat.cross_sectional_area": np.array([]),
            "throat.length": np.array([]),
            "pore.shape_factor": np.array([]),
            "pore.radius": np.array([]),
            "pore.volume": np.array([]),
            "pore.clay_volume": np.array([]),
            "pore.area": np.array([]),
            "pore.length": np.array([]),
        }

        pn = op.network.Network()
        pn.update(pn_dict)
        initialize_pn(pn)
        pn.update(pn_dict)
        pn["pore.oil_saturation"] = 1.0 - pn["pore.water_saturation"]
        pn["throat.oil_saturation"] = 1.0 - pn["throat.water_saturation"]
        pn["pore.oil_filled"] = ~pn["pore.water_filled"]
        pn["throat.oil_filled"] = ~pn["throat.water_filled"]

        network = Network(pn)
        phase_connection_checker = PhaseConnectionChecker(network, WATER)
        phase_connection_checker.update_phase_connections()

        border_elements = BorderElements()
        CycleSimulation._find_border_elements(network.pn_core, border_elements, WATER)

        np.testing.assert_equal(border_elements.get_pores_ids(), [3, 6, 7, 9, 13, 14, 16])
        np.testing.assert_equal(border_elements.get_throats_ids(), [])

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
