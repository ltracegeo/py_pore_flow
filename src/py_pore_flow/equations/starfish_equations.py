from math import cos, pi, sqrt

from numba import njit

from .common_equations import CIRCLE, SQUARE
import py_pore_flow.constants as c


@njit
def calculate_element_singlephase_conductance(A, G, shape, mi):
    if shape == SQUARE:
        N = 4  # Number of sides
        g = 2 * A**2 * sqrt(G) / (14.18 + (0.5 / N) - (26.4 / N**2) + (102.18 / N**3))
    else:
        if shape == CIRCLE:
            v = 0.5
        else:  # shape == TRIANGLE
            v = 0.6
        g = v * (A**2 * G) / (mi)
    return g


@njit
def calculate_element_multiphase_oil_conductance(A, G, shape, So):
    if shape == CIRCLE:
        v = 0.5
    else:  # shape == TRIANGLE
        v = 0.6
    go = v * (So * A) ** 2 * G / c.mio
    return go


@njit
def calculate_pc(shape_factor, radius, theta, shape, C, beta, n_corners):
    G = shape_factor
    r = radius

    if shape == CIRCLE:
        Fd = 1
    else:
        Fd = (1 + sqrt(1 - (4 * G * C / cos(theta) ** 2))) / (1 + 2 * sqrt(pi * G))

    Pc = c.sigma * cos(theta) / r * (1 + 2 * sqrt(pi * G)) * Fd
    return Pc
