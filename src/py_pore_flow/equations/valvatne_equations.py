from math import cos, pi, sin, sqrt

from numba import njit

from .common_equations import CIRCLE, SQUARE, TRIANGLE
from py_pore_flow.constants import mio, sigma


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
def calculate_element_multiphase_oil_conductance(A, G, shape, So):
    go = calculate_element_singlephase_conductance(A, G, shape, mio)
    return So * go


@njit
def calculate_pc(shape_factor, radius, theta, shape, C, beta, n_corners):
    G = shape_factor
    r = radius

    if shape == CIRCLE:
        Fd = 1
    else:
        S1 = C
        S2 = 0
        S3 = 0
        for i in range(n_corners):
            if beta[i] < pi / 2 - theta:
                S2 += cos(theta + beta[i]) / sin(beta[i])
                S3 += pi / 2 - theta - beta[i]
        S3 = 2 * S3
        D = S1 - 2 * S2 * cos(theta) + S3
        Fd = (1 + sqrt(1 + (4 * G * D / cos(theta) ** 2))) / (1 + 2 * sqrt(pi * G))

    Pc = sigma * cos(theta) / r * (1 + 2 * sqrt(pi * G)) * Fd
    return Pc
