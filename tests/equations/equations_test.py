import unittest

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import openpnm as op

from equations import (
    CIRCLE,
    SQUARE,
    TRIANGLE,
    calculate_beta,
    calculate_drainage_displacement,
    calculate_shape,
)
import global_random
from network import initialize_pn, Network
from simulation.simulation import construct_network


class TestEquations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global_random.set_seed(0)
        cls.pn = op.io.network_from_statoil("tests/resources/statoil", "a")
        initialize_pn(cls.pn)
        cls.network = construct_network(cls.pn, 40000)

    def setUp(self):
        global_random.set_seed(0)

    def test_shape(self):
        self.assertEqual(calculate_shape(0.0443548387096774), TRIANGLE)
        self.assertEqual(calculate_shape(0.0490196078431372), SQUARE)
        self.assertEqual(calculate_shape(0.0677922077922077), CIRCLE)

    def test_half_corner_angles(self):
        beta = calculate_beta(0.0443548387096774, TRIANGLE)
        assert_almost_equal(beta, np.array([0.35483625, 0.52387179, 0.69208829]))
        beta = calculate_beta(0.0490196078431372, SQUARE)
        assert_almost_equal(beta, np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816]))
        beta = calculate_beta(0.0677922077922077, CIRCLE)
        assert_almost_equal(beta, np.array([]))

    def test_saturation(self):
        Pc = 5000

        C = 2.4947709827738445
        A = 3.261628660660103e-10
        Sw, So = calculate_drainage_displacement(TRIANGLE, C, Pc, A)
        self.assertEqual(Sw, 0.27535861596728156)

        C = 0.8584073464102069
        A = 4.2160675784154405e-10
        Sw, So = calculate_drainage_displacement(SQUARE, C, Pc, A)
        self.assertEqual(Sw, 0.07329736512995329)

        C = 0.0
        A = 1.321051480799487e-10
        Sw, So = calculate_drainage_displacement(CIRCLE, C, Pc, A)
        self.assertEqual(Sw, 0.0)
