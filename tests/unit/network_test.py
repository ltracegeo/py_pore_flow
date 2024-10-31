import unittest

from numpy.testing import assert_almost_equal, assert_equal
import openpnm as op

import global_random
from network import initialize_pn, Network
from parameters.parameter_manager import ParameterManager
from simulation.simulation import calculate_initial_values

from equations.geometry import (
    calculate_C,
)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_random.set_seed(0)
        cls.pn = op.io.network_from_statoil("tests/resources/a", "a")
        initialize_pn(cls.pn)
        cls.network = Network(cls.pn)
        parameter_manager = ParameterManager()
        parameter_manager.update_initial_cas(cls.network)
        calculate_initial_values(cls.network)

    def setUp(self):
        global_random.set_seed(0)

    def test_properties(self):
        self.assertEqual(self.network.get_pore(948).G, self.pn["pore.shape_factor"][948])
        self.assertEqual(self.network.get_pore(666).G, self.pn["pore.shape_factor"][666])
        self.assertEqual(self.network.get_pore(889).G, self.pn["pore.shape_factor"][889])
        assert_equal(self.network.get_pore(948).C_r, self.pn["pore.C_r"][948])
        assert_equal(self.network.get_pore(666).C_r, self.pn["pore.C_r"][666])
        assert_equal(self.network.get_pore(889).C_r, self.pn["pore.C_r"][889])

    def test_values(self):
        element = self.network.get_pore(948)
        assert_almost_equal(element.G, 0.0443548387096774)
        assert_almost_equal(element.r, 7.607076e-06)
        assert_almost_equal(element.advancing_contact_angle, 0.0)  # pq é zero?
        assert_almost_equal(element.receding_contact_angle, 0.0)
        assert_almost_equal(element.shape, 0)
        assert_almost_equal(element.beta[:3], [0.35751046, 0.51436028, 0.69892559])
        assert_almost_equal(element.C_r, 0.0)
        assert_almost_equal(element.C_a, 0.0)
        assert_almost_equal(element.A / 1e-10, 3.2616286606601026)

        element = self.network.get_throat(435)
        assert_almost_equal(element.G, 0.07957747154594767)
        assert_almost_equal(element.r, 4.98e-06)
        pore_list = self.network.get_neighbor_pores(element.get_id())
        assert_almost_equal(len(pore_list), 2)
        assert_almost_equal([pore_list[0].get_id(), pore_list[1].get_id()], [34, 948])
