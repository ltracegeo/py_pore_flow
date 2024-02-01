import unittest

from numpy.testing import assert_almost_equal, assert_equal
import openpnm as op

import global_random
from network import initialize_pn, Network
from parameters.parameter_manager import ParameterManager
from simulation.simulation import calculate_initial_values

from equations import (
    calculate_C,
)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_random.set_seed(0)
        cls.pn = op.io.network_from_statoil("tests/resources/statoil", "a")
        initialize_pn(cls.pn)
        cls.network = Network(cls.pn)
        parameter_manager = ParameterManager()
        parameter_manager.update_initial_cas(cls.network)
        calculate_initial_values(cls.network, 40000)

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
        assert_almost_equal(element.advancing_contact_angle, 0.300665613626673)
        assert_almost_equal(element.receding_contact_angle, 0.014092640613489886)
        assert_almost_equal(element.shape, 0)
        assert_almost_equal(element.beta, [0.35751046, 0.51436028, 0.69892559])
        assert_almost_equal(element.C_r, 2.4936572583035845)
        assert_almost_equal(element.C_a, 2.0538004103939693)
        assert_almost_equal(element.A / 1e-10, 3.2616286606601026)

        element = self.network.get_pore(666)
        assert_almost_equal(element.G, 0.0490196078431372)
        assert_almost_equal(element.r, 9.092194e-06)
        assert_almost_equal(element.advancing_contact_angle, 0.2883398466808009)
        assert_almost_equal(element.receding_contact_angle, 0.013387853642248211)
        assert_almost_equal(element.shape, 1)
        assert_almost_equal(element.beta, [0.78539816, 0.78539816, 0.78539816, 0.78539816])
        assert_almost_equal(element.C_r, 0.8576968493577684)
        assert_almost_equal(element.C_a, 0.597834802625202)
        assert_almost_equal(element.A / 1e-10, 4.216067578415440)

        element = self.network.get_pore(889)
        assert_almost_equal(element.G, 0.0677922077922077)
        assert_almost_equal(element.r, 5.985215e-06)
        assert_almost_equal(element.advancing_contact_angle, 0.3379879648141799)
        assert_almost_equal(element.receding_contact_angle, 0.01630421505269301)
        assert_almost_equal(element.shape, 2)
        assert_almost_equal(element.beta, [])
        assert_almost_equal(element.C_r, 0.0)
        assert_almost_equal(element.C_a, 0.0)
        assert_almost_equal(element.A / 1e-10, 1.3210514807994869)

        element = self.network.get_throat(435)
        assert_almost_equal(element.G, 0.07258065)
        assert_almost_equal(element.r, 4.98e-06)
        assert_almost_equal(element.advancing_contact_angle, 0.14033475216582803)
        assert_almost_equal(element.receding_contact_angle, 0.005828954625540091)
        assert_almost_equal(element.shape, 2)
        assert_almost_equal(element.beta, [])
        assert_almost_equal(element.C_r, 0.0)
        assert_almost_equal(element.C_a, 0.0)
        assert_almost_equal(element.A / 1e-10, 0.854235943050937)
        assert_almost_equal(element.L, 2.620349e-05)
        pore_list = self.network.get_neighbor_pores(element.get_id())
        assert_almost_equal(len(pore_list), 2)
        assert_almost_equal([pore_list[0].get_id(), pore_list[1].get_id()], [34, 948])

        element = self.network.get_pore(34)
        assert_almost_equal(element.G, 0.0354226214781543)
        assert_almost_equal(element.r, 2.903814e-05)
        assert_almost_equal(element.advancing_contact_angle, 0.10274746834472614)
        assert_almost_equal(element.receding_contact_angle, 0.004151172630209948)
        assert_almost_equal(element.shape, 0)
        assert_almost_equal(element.beta, [0.233417, 0.4593866, 0.8779927])
        assert_almost_equal(element.C_r, 3.9159227589325503)
        assert_almost_equal(element.C_a, 3.8439631311119697)
        assert_almost_equal(element.A / 1e-10, 59.5109522864946)
