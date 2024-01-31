import unittest

import openpnm as op

from network import initialize_pn, Network
from numpy.testing import assert_equal
from simulation.simulation import construct_network


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.pn = op.io.network_from_statoil("tests/resources/statoil", "a")
        initialize_pn(self.pn)
        self.network = construct_network(self.pn, 40000)

    def test_shape_factor(self):
        self.assertEqual(self.network.get_pore(948).G, self.pn["pore.shape_factor"][948])
        self.assertEqual(self.network.get_pore(666).G, self.pn["pore.shape_factor"][666])
        self.assertEqual(self.network.get_pore(889).G, self.pn["pore.shape_factor"][889])

    def test_C(self):
        assert_equal(self.network.get_pore(948).C, self.pn["pore.C"][948])
        assert_equal(self.network.get_pore(666).C, self.pn["pore.C"][666])
        assert_equal(self.network.get_pore(889).C, self.pn["pore.C"][889])
