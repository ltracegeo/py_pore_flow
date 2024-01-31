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
    calculate_drainage_pc,
    calculate_imbibition_piston_like_displacement_pc,
    calculate_shape,
)
import global_random
from network import initialize_pn, Network
from parameters.parameter_manager import ParameterManager
from simulation.simulation import calculate_initial_values


class TestEquations(unittest.TestCase):
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
