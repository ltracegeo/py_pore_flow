import unittest

from numpy.testing import assert_equal
import openpnm as op

import global_random
from network import initialize_pn
from parameters.parameter_manager import ParameterManager
from simulation.simulation import construct_network


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_random.set_seed(0)
        cls.pn = op.io.network_from_statoil("tests/resources/statoil", "a")
        initialize_pn(cls.pn)
        cls.network = construct_network(cls.pn, 40000)
        parameter_manager = ParameterManager()
        parameter_manager.update_initial_cas(cls.network)

    def setUp(self):
        global_random.set_seed(0)

    def test_shape_factor(self):
        self.assertEqual(self.network.get_pore(948).G, self.pn["pore.shape_factor"][948])
        self.assertEqual(self.network.get_pore(666).G, self.pn["pore.shape_factor"][666])
        self.assertEqual(self.network.get_pore(889).G, self.pn["pore.shape_factor"][889])

    def test_C(self):
        assert_equal(self.network.get_pore(948).C, self.pn["pore.C"][948])
        assert_equal(self.network.get_pore(666).C, self.pn["pore.C"][666])
        assert_equal(self.network.get_pore(889).C, self.pn["pore.C"][889])
