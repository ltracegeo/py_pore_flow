from math import cos, pi, sin

from numba import njit


TRIANGLE = 0
SQUARE = 1
CIRCLE = 2

WATER = 0
OIL = 1

MAX_NUMBER_OF_CORNERS = 4


@njit
def calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o):
    Ao = (sin(beta) / cos(theta_o - beta)) ** 2 * (
        (cos(theta_o) * cos(theta_o - beta) / sin(beta)) - theta_o + beta + pi / 2
    )
    Ai = (b_tilde * sin(beta) / cos(theta_i + beta)) ** 2 * (
        (cos(theta_i) * cos(theta_i + beta) / sin(beta)) + theta_i + beta - pi / 2
    )

    Al = Ao - Ai

    return Al


@njit
def calculate_corner_C(theta, beta):
    return cos(theta) * (cos(theta + beta) / sin(beta)) - (pi / 2 - theta - beta)
