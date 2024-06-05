import math
from math import cos, log, pi, sin, sqrt

from numba import njit

from .common_equations import CIRCLE, SQUARE, TRIANGLE, calculate_corner_C, calculate_dimensionless_layer_area
import py_pore_flow.constants as c


@njit
def calculate_element_singlephase_conductance(A, G, shape, mi):
    if shape == CIRCLE:
        v = 0.5
    elif shape == TRIANGLE:
        v = 0.6
    else:
        v = 0.5623
    g = v * (A**2 * G) / (mi)
    return g


@njit
def calculate_element_multiphase_oil_conductance(bi, bo, beta, theta_i, theta_o, corner_oil_layer, n_corners):
    go = 0
    for i in range(n_corners):
        if corner_oil_layer[i]:
            go += calculate_dimensionless_layer_conductance(bi[i], bo[i], beta[i], theta_i[i], theta_o[i])
    return go


@njit
def calculate_pc(shape_factor, radius, theta, shape, C, beta, n_corners):
    G = shape_factor
    r = radius

    if shape == CIRCLE:
        Fd = 1
    else:
        S1 = 0
        S2 = 0
        S3 = 0
        for i in range(n_corners):
            if beta[i] < pi / 2 - theta:
                S1 += calculate_corner_C(theta, beta[i])
                S2 += cos(theta + beta[i]) / sin(beta[i])
                S3 += pi / 2 - theta - beta[i]
        S3 = 2 * S3
        D = S1 - 2 * S2 * cos(theta) + S3
        Fd = (1 + sqrt(1 + (4 * G * D / cos(theta) ** 2))) / (1 + 2 * sqrt(pi * G))

    Pc = c.sigma * cos(theta) / r * (1 + 2 * sqrt(pi * G)) * Fd
    return Pc


@njit
def calculate_dimensionless_layer_conductance(bi, bo, beta, theta_i, theta_o):
    b_tilde = bi / bo

    Al = calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o)

    Loow = 2 * sin(beta) / cos(theta_o - beta) * (pi / 2 - theta_o + beta)
    Liow = 2 * b_tilde * sin(beta) / cos(theta_i + beta) * (pi / 2 - theta_i - beta)

    Gl = Al / ((Loow + Liow + 2 * (1 - b_tilde)) ** 2)

    a1, a2, a3 = get_tabulated_conductance_constants(beta)

    gl = math.e ** (a1 * log(Al**3 * Gl) ** 2 + a2 * log(Al**3 * Gl) + a3)

    gpl = bo**4 * gl / c.mio

    return gpl


@njit
def get_tabulated_conductance_constants(beta):
    if beta < 0.17453292519943295:  # 10 degrees
        a1, a2, a3 = -1.061e-02, 5.161e-01, -2.065
    elif beta < 0.3490658503988659:  # 20 degrees
        a1, a2, a3 = -2.681e-02, 1.867e-01, -3.598
    elif beta < 0.5235987755982988:  # 30 degrees
        a1, a2, a3 = -4.402e-02, -6.320e-02, -4.375
    elif beta < 0.6981317007977318:  # 40 degrees
        a1, a2, a3 = -3.152e-02, 1.695e-01, -3.360
    elif beta < 0.8726646259971648:  # 50 degrees
        a1, a2, a3 = -3.137e-02, 1.933e-01, -3.267
    elif beta < 1.0471975511965976:  # 60 degrees
        a1, a2, a3 = -2.320e-02, 3.118e-01, -2.925
    else:
        a1, a2, a3 = -3.576e-02, -6.537e-04, -4.702
    return a1, a2, a3
