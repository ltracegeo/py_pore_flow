import unittest

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import openpnm as op

from equations import (
    CIRCLE,
    SQUARE,
    TRIANGLE,
    WATER,
    OIL,
    calculate_beta,
    calculate_drainage_b,
    calculate_drainage_displacement,
    calculate_drainage_pc,
    calculate_hinging_contact_angle,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_multiphase_oil_conductance,
    calculate_multiphase_water_conductance,
    calculate_singlephase_conductance,
    calculate_shape,
)
import global_random
from network import initialize_pn, Network
from parameters.parameter_manager import ParameterManager
from simulation.simulation import calculate_initial_values


class TestEquations(unittest.TestCase):
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

        C_r = 2.4947709827738445
        A = 3.261628660660103e-10
        Sw, So = calculate_drainage_displacement(TRIANGLE, C_r, Pc, A)
        self.assertEqual(Sw, 0.27535861596728156)

        C_r = 0.8584073464102069
        A = 4.2160675784154405e-10
        Sw, So = calculate_drainage_displacement(SQUARE, C_r, Pc, A)
        self.assertEqual(Sw, 0.07329736512995329)

        C_r = 0.0
        A = 1.321051480799487e-10
        Sw, So = calculate_drainage_displacement(CIRCLE, C_r, Pc, A)
        self.assertEqual(Sw, 0.0)

    def test_drainage_entry_pc(self):
        shape_factor = 0.0443548387096774
        radius = 7.607076e-06
        receding_contact_angle = 0.014092640613489886
        shape = 0
        C_r = 2.4947709827738445
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, C_r, 0)
        self.assertEqual(pc, 5628.935432844781)

        shape_factor = 0.0490196078431372
        radius = 9.092194e-06
        receding_contact_angle = 0.013387853642248211
        shape = 1
        C_r = 0.8584073464102069
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, C_r, 0)
        self.assertEqual(pc, 4653.5748971834755)

        shape_factor = 0.0677922077922077
        radius = 5.985215e-06
        receding_contact_angle = 0.01630421505269301
        shape = 2
        C_r = 0.0
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, C_r, 0)
        self.assertEqual(pc, 9637.397115065623)

    def test_imbibition_entry_pc(self):
        pc = 5000

        shape_factor = 0.0443548387096774
        radius = 7.607076e-06
        advancing_contact_angle = 0.300665613626673
        shape = 0
        C_a = 2.0538004103939693
        entry_pc = calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, C_a, pc, 0
        )
        self.assertEqual(entry_pc, 5351.140845431006)

        shape_factor = 0.0490196078431372
        radius = 9.092194e-06
        advancing_contact_angle = 0.2883398466808009
        shape = 1
        C_a = 0.597834802625202
        entry_pc = calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, C_a, pc, 0
        )
        self.assertEqual(entry_pc, 4447.019531129279)

        shape_factor = 0.0677922077922077
        radius = 5.985215e-06
        advancing_contact_angle = 0.3379879648141799
        shape = 2
        C_a = 0.0
        entry_pc = calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, C_a, pc, 0
        )
        self.assertEqual(entry_pc, 9093.357879396488)

    def test_b(self):
        max_pc = 40000

        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        receding_contact_angle = 0.014092640613489886
        b = calculate_drainage_b(beta, receding_contact_angle, max_pc)
        assert_almost_equal(b, np.array([2.0134182e-06, 1.2875214e-06, 8.9420597e-07]))

        beta = np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816])
        receding_contact_angle = 0.013387853642248211
        b = calculate_drainage_b(beta, receding_contact_angle, max_pc)
        assert_almost_equal(b, np.array([7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07]))

    def test_hinging_contact_angle(self):
        pc = 5000

        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        b = np.array([2.0134182e-06, 1.2875214e-06, 8.9420597e-07])
        theta_h = calculate_hinging_contact_angle(pc, beta, b, 3)
        assert_almost_equal(theta_h, np.array([1.099105, 0.9393731, 0.7834584]))

        beta = np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816])
        b = np.array([7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07])
        theta_h = calculate_hinging_contact_angle(pc, beta, b, 4)
        assert_almost_equal(theta_h, np.array([0.6980902, 0.6980902, 0.6980902, 0.6980902]))

    def test_conductance(self):
        A = np.empty(3, dtype=np.float64)
        A[0] = 3.2616286606601026e-10
        A[1] = 0.85423594305093e-10
        A[2] = 59.5109522864946e-10
        G = np.empty(3, dtype=np.float64)
        G[0] = 0.0443548387096774
        G[1] = 0.07258065
        G[2] = 0.0354226214781543
        L = np.empty(3, dtype=np.float64)
        L[0] = 7.607076e-06
        L[1] = 2.620349e-05
        L[2] = 2.903814e-05
        shape = np.empty(3, dtype=np.float64)
        shape[0] = 0
        shape[1] = 2
        shape[2] = 0

        assert_equal(calculate_singlephase_conductance(L, A, G, shape, WATER), 9.366933989647119e-15)
        assert_equal(calculate_singlephase_conductance(L, A, G, shape, OIL), 7.0757415029708445e-15)

        So = np.empty(3, dtype=np.float64)
        So[0] = 0.7246413840327184
        So[1] = 1
        So[2] = 0.9267026348700467
        assert_equal(calculate_multiphase_oil_conductance(L, A, G, shape, So), 7.005185434381609e-15)

        Sw = np.empty(3, dtype=np.float64)
        Sw[0] = 0.27535861596728156
        Sw[1] = 1
        Sw[2] = 0.07329736512995329
        theta_h = [None] * 3
        theta_h[0] = np.array([1.099105, 0.9393731, 0.7834584])
        theta_h[1] = np.array([])
        theta_h[2] = np.array([0.6980902, 0.6980902, 0.6980902, 0.6980902])
        b = [None] * 3
        b[0] = np.array([2.0134182e-06, 1.2875214e-06, 8.9420597e-07])
        b[1] = np.array([])
        b[2] = np.array([7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07])
        beta = [None] * 3
        beta[0] = np.array([0.35483625, 0.52387179, 0.69208829])
        beta[1] = np.array([])
        beta[2] = np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816])
        assert_equal(calculate_multiphase_water_conductance(L, A, G, shape, Sw, theta_h, b, beta), 3.76155495677152e-17)
