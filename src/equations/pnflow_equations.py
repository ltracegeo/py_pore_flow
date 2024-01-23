from numba import njit

from .common_equations import CIRCLE, SQUARE, TRIANGLE
from constants import mio


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
