import unittest

from numpy.testing import assert_almost_equal, assert_equal
import numpy as np
import openpnm as op

from equations.geometry import (
    CIRCLE,
    SQUARE,
    TRIANGLE,
    WATER,
    OIL,
    calculate_beta,
    calculate_drainage_b,
    calculate_drainage_element_saturation,
    calculate_hinging_contact_angle,
    calculate_shape,
)
from equations.entry_pressure import (
    calculate_drainage_pc,
    calculate_snap_off_pc_element,
    calculate_imbibition_piston_like_displacement_pc_element,
)
from equations.conductance import (
    calculate_network_singlephase_conductivity,
    calculate_network_multiphase_water_conductivity,
    calculate_network_multiphase_oil_conductivity,
    calculate_conductance,
)
import global_random
from network import initialize_pn, Network
from parameters.parameter_manager import ParameterManager
from simulation.simulation import calculate_initial_values
import constants as c


class TestEquations(unittest.TestCase):
    def setUp(self):
        global_random.set_seed(0)

    def test_shape(self):
        self.assertEqual(calculate_shape(0.0443548387096774)[0], TRIANGLE)
        self.assertEqual(calculate_shape(0.0490196078431372)[0], SQUARE)
        self.assertEqual(calculate_shape(0.0677922077922077)[0], CIRCLE)

    def test_half_corner_angles(self):
        beta = calculate_beta(0.0443548387096774, TRIANGLE)
        assert_almost_equal(beta[:3], np.array([0.35483625, 0.52387179, 0.69208829]))
        beta = calculate_beta(0.0490196078431372, SQUARE)
        assert_almost_equal(beta[:4], np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816]))
        beta = calculate_beta(0.0677922077922077, CIRCLE)
        assert_almost_equal(beta[0], np.array([]))

    def test_saturation(self):
        Pc = 5000

        C_r = 2.4947709827738445
        A = 3.261628660660103e-10
        oil_filled = True
        Sw, So = calculate_drainage_element_saturation(TRIANGLE, C_r, Pc, A, oil_filled)
        self.assertEqual(Sw, 0.27535861596728156)

        C_r = 0.8584073464102069
        A = 4.2160675784154405e-10
        oil_filled = True
        Sw, So = calculate_drainage_element_saturation(SQUARE, C_r, Pc, A, oil_filled)
        self.assertEqual(Sw, 0.07329736512995329)

        C_r = 0.0
        A = 1.321051480799487e-10
        oil_filled = True
        Sw, So = calculate_drainage_element_saturation(CIRCLE, C_r, Pc, A, oil_filled)
        self.assertEqual(Sw, 0.0)

        C_r = 0.0
        A = 1.321051480799487e-10
        oil_filled = False
        Sw, So = calculate_drainage_element_saturation(CIRCLE, C_r, Pc, A, oil_filled)
        self.assertEqual(Sw, 1.0)

    def test_drainage_entry_pc(self):
        shape_factor = 0.0443548387096774
        radius = 7.607076e-06
        receding_contact_angle = 0.014092640613489886
        shape = 0
        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, beta, len(beta))
        self.assertEqual(pc, 6887.581229783431)

        shape_factor = 0.0490196078431372
        radius = 9.092194e-06
        receding_contact_angle = 0.013387853642248211
        shape = 1
        beta = np.array([0.35483625, 0.52387179, 0.69208829, 0.4])
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, beta, len(beta))
        self.assertEqual(pc, 5034.958069238447)

        shape_factor = 1.0 / (4.0 * np.pi)
        radius = 5.985215e-06
        receding_contact_angle = 0.01630421505269301
        shape = 2
        beta = np.array([0.35483625])
        pc = calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, beta, len(beta))
        self.assertEqual(pc, 2 * c.sigma * np.cos(receding_contact_angle) / radius)

    def test_imbibition_entry_pc(self):
        pc = 5000

        shape_factor = 0.0443548387096774
        radius = 7.607076e-06
        b = np.array([0.1, 0.1, 0.1])
        advancing_contact_angle = 0.300665613626673
        receding_contact_angle = 0.300665613626673
        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        shape = 0
        Rmin = c.sigma / pc
        entry_pc = calculate_imbibition_piston_like_displacement_pc_element(
            shape_factor, radius, b, advancing_contact_angle, receding_contact_angle, shape, beta, len(beta), Rmin
        )
        self.assertEqual(entry_pc, 6810.556807178534)

        shape_factor = 0.0490196078431372
        radius = 9.092194e-06
        b = np.array([0.1, 0.1, 0.1, 0.1])
        advancing_contact_angle = 0.2883398466808009
        receding_contact_angle = 0.300665613626673
        beta = np.array([0.35483625, 0.52387179, 0.69208829, 0.4])
        shape = 1
        Rmin = c.sigma / pc
        entry_pc = calculate_imbibition_piston_like_displacement_pc_element(
            shape_factor, radius, b, advancing_contact_angle, receding_contact_angle, shape, beta, len(beta), Rmin
        )
        self.assertEqual(entry_pc, 5001.326568915203)

        shape_factor = 0.0677922077922077
        radius = 5.985215e-06
        b = np.array([0.1])
        advancing_contact_angle = 0.3379879648141799
        receding_contact_angle = 0.300665613626673
        beta = np.array([0.35483625])
        shape = 2
        Rmin = c.sigma / pc
        entry_pc = calculate_imbibition_piston_like_displacement_pc_element(
            shape_factor, radius, b, advancing_contact_angle, receding_contact_angle, shape, beta, len(beta), Rmin
        )
        self.assertEqual(entry_pc, 9283.62521689095)

    def test_snap_off_pc(self):
        pc = 5000
        radius = 1e-06
        Rmin = c.sigma / pc
        h = 0.0
        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        n_corners = len(beta)

        # two moving menisci
        advancing_ca = 0.3
        receding_ca = 0.3
        snap_off_pc = calculate_snap_off_pc_element(radius, beta, advancing_ca, receding_ca, n_corners, Rmin, h)
        self.assertEqual(snap_off_pc, 24657.452515046127)

        # single moving menisci
        advancing_ca = 0.3
        receding_ca = 0.6
        snap_off_pc = calculate_snap_off_pc_element(radius, beta, advancing_ca, receding_ca, n_corners, Rmin, h)
        self.assertEqual(snap_off_pc, 51933.37546591764)

        # oil-wet 1
        advancing_ca = 1.3
        receding_ca = 0.6
        snap_off_pc = calculate_snap_off_pc_element(radius, beta, advancing_ca, receding_ca, n_corners, Rmin, h)
        self.assertEqual(snap_off_pc, -726.4572419676624)

        # oil-wet 2
        advancing_ca = 2.9
        receding_ca = 0.3
        snap_off_pc = calculate_snap_off_pc_element(radius, beta, advancing_ca, receding_ca, n_corners, Rmin, h)
        self.assertEqual(snap_off_pc, -6303.996350456403)

    def test_b(self):
        max_pc = 40000

        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        n_corners = len(beta)
        receding_contact_angle = np.full(n_corners, 0.014092640613489886)
        b = calculate_drainage_b(beta, receding_contact_angle, n_corners, max_pc)
        assert_almost_equal(b[:n_corners], np.array([2.0134182e-06, 1.2875214e-06, 8.9420597e-07]))

        beta = np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816])
        n_corners = len(beta)
        receding_contact_angle = np.full(n_corners, 0.013387853642248211)
        b = calculate_drainage_b(beta, receding_contact_angle, n_corners, max_pc)
        assert_almost_equal(b[:n_corners], np.array([7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07]))

    def test_hinging_contact_angle(self):
        pc = 5000

        beta = np.array([0.35483625, 0.52387179, 0.69208829])
        b = np.array([2.0134182e-06, 1.2875214e-06, 8.9420597e-07])
        n_corners = len(beta)
        receding_contact_angle = np.full(n_corners, 0.011)
        advancing_contact_angle = np.full(n_corners, 0.014)
        theta_h = np.zeros(n_corners)
        calculate_hinging_contact_angle(
            pc, beta, b, n_corners, receding_contact_angle, advancing_contact_angle, theta_h
        )
        assert_almost_equal(theta_h, np.array([0.014, 0.014, 0.014]))

        beta = np.array([0.78539816, 0.78539816, 0.78539816, 0.78539816])
        b = np.array([7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07])
        n_corners = len(beta)
        receding_contact_angle = np.full(n_corners, 0.011)
        advancing_contact_angle = np.full(n_corners, 0.014)
        theta_h = np.zeros(n_corners)
        calculate_hinging_contact_angle(
            pc, beta, b, n_corners, receding_contact_angle, advancing_contact_angle, theta_h
        )
        assert_almost_equal(theta_h, np.array([0.014, 0.014, 0.014, 0.014]))

    def test_singlephase_conductance(self):
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

        conductance_water_template = 1 / (
            L[0] / (0.6 * A[0] ** 2 * G[0] / c.miw)
            + L[1] / (0.5 * A[1] ** 2 * G[1] / c.miw)
            + L[2] / (0.6 * A[2] ** 2 * G[2] / c.miw)
        )
        conductance_oil_template = 1 / (
            L[0] / (0.6 * A[0] ** 2 * G[0] / c.mio)
            + L[1] / (0.5 * A[1] ** 2 * G[1] / c.mio)
            + L[2] / (0.6 * A[2] ** 2 * G[2] / c.mio)
        )

        gp = np.empty(3, dtype=np.float64)

        calculate_network_singlephase_conductivity(gp, A, G, shape, WATER)
        assert_equal(calculate_conductance(L, gp), conductance_water_template)

        calculate_network_singlephase_conductivity(gp, A, G, shape, OIL)
        assert_equal(calculate_conductance(L, gp), conductance_oil_template)

    def test_multiphase_oil_conductance(self):
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
        So = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        beta = np.array(
            [
                [0.35483625, 0.52387179, 0.69208829],
                [0.35483625, 0.52387179, 0.69208829],
                [0.35483625, 0.52387179, 0.69208829],
            ]
        )
        n_corners = np.array([3, 3, 3])
        b = np.array(
            [
                [2.0134182e-06, 1.2875214e-06, 8.9420597e-07],
                [2.0134182e-06, 1.2875214e-06, 8.9420597e-07],
                [2.0134182e-06, 1.2875214e-06, 8.9420597e-07],
            ]
        )
        bo = np.array(
            [
                [8.9420597e-07, 2.0134182e-06, 1.2875214e-06],
                [8.9420597e-07, 2.0134182e-06, 1.2875214e-06],
                [8.9420597e-07, 2.0134182e-06, 1.2875214e-06],
            ]
        )
        theta_h = np.array(
            [[0.35483625, 0.52387179, 0.1], [0.35483625, 0.52387179, 0.1], [0.35483625, 0.52387179, 0.1]]
        )
        theta_a = np.array(
            [[0.35483625, 0.52387179, 0.1], [0.35483625, 0.52387179, 0.1], [0.35483625, 0.52387179, 0.1]]
        )
        corner_oil_layer = np.array([[True, True, True], [True, True, True], [True, True, True]])
        has_oil_in_center = np.array([False, False, False])

        gp = np.empty(3, dtype=np.float64)
        calculate_network_multiphase_oil_conductivity(
            gp, A, G, shape, So, b, bo, beta, theta_h, theta_a, corner_oil_layer, n_corners, has_oil_in_center
        )
        oil_conductance = calculate_conductance(L, gp)
        assert_equal(oil_conductance, 3.350602871904144e-18)

    def test_multiphase_water_conductance(self):
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

        # Water conductance without water in center
        Sw = np.empty(3, dtype=np.float64)
        theta_h = np.array(
            [
                [1.099105, 0.9393731, 0.7834584, 0.0],
                [1.099105, 0.9393731, 0.7834584, 0.0],
                [0.6980902, 0.6980902, 0.6980902, 0.6980902],
            ]
        )
        b = np.array(
            [
                [2.0134182e-06, 1.2875214e-06, 8.9420597e-07, 0.0],
                [2.0134182e-06, 1.2875214e-06, 8.9420597e-07, 0.0],
                [7.398922e-07, 7.398922e-07, 7.398922e-07, 7.398922e-07],
            ]
        )
        beta = np.array(
            [
                [0.35483625, 0.52387179, 0.69208829, 0.0],
                [0.35483625, 0.52387179, 0.69208829, 0.0],
                [0.78539816, 0.78539816, 0.78539816, 0.78539816],
            ],
        )

        has_water_in_center = np.array([False, False, False])
        n_corners = np.array([len(beta[i]) for i in range(len(beta))])

        gp = np.empty(3, dtype=np.float64)
        calculate_network_multiphase_water_conductivity(
            gp, A, G, shape, Sw, theta_h, b, beta, n_corners, has_water_in_center
        )
        water_conductance = calculate_conductance(L, gp)
        assert_almost_equal(water_conductance, 2.0648208672078985e-19)
