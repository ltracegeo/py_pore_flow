import unittest

from numpy.testing import assert_almost_equal, assert_equal
import openpnm as op

from py_pore_flow import global_random
from py_pore_flow.network.network import initialize_pn, NetworkManager, PoreNetwork
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.simulation import calculate_initial_values

from py_pore_flow.equations.geometry import (
    calculate_C,
)


class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_random.set_seed(0)
        cls.pn = op.io.network_from_statoil("tests/resources/a", "a")
        cls.pn = PoreNetwork.from_openpnm(cls.pn)
        initialize_pn(cls.pn)
        cls.network = NetworkManager(cls.pn)
        parameter_manager = ParameterManager()
        parameter_manager.update_initial_cas(cls.network)
        calculate_initial_values(cls.network)

    def setUp(self):
        global_random.set_seed(0)

    def test_properties(self):
        pore_ids = [948, 666, 889]
        pore = self.network.get_pore()
        for pore_id in pore_ids:
            self.assertEqual(pore.get_G(pore_id), self.pn["pore.shape_factor"][pore_id])
            assert_equal(pore.get_C_r(pore_id), self.pn["pore.C_r"][pore_id])


    def test_values(self):
        pore_id = 948
        pore = self.network.get_pore()
        assert_almost_equal(pore.get_G(pore_id), 0.0443548387096774)
        assert_almost_equal(pore.get_r(pore_id), 7.607076e-06)
        assert_almost_equal(pore.get_advancing_contact_angle(pore_id), 0.0)  # pq é zero?
        assert_almost_equal(pore.get_receding_contact_angle(pore_id), 0.0)
        assert_almost_equal(pore.get_shape(pore_id), 0)
        assert_almost_equal(pore.get_beta(pore_id)[:3], [0.35751046, 0.51436028, 0.69892559])
        assert_almost_equal(pore.get_C_r(pore_id), 0.0)
        assert_almost_equal(pore.get_C_a(pore_id), 0.0)
        assert_almost_equal(pore.get_A(pore_id) / 1e-10, 3.2616286606601026)

        throat_id = 435
        throat = self.network.get_throat()
        assert_almost_equal(throat.get_G(throat_id), 0.07957747154594767)
        assert_almost_equal(throat.get_r(throat_id), 4.98e-06)
        pore_ids = self.network.get_neighbor_pores_ids(throat_id)
        assert_almost_equal(len(pore_ids), 2)
        assert_almost_equal(pore_ids, [34, 948])
