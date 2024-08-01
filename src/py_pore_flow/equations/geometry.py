from math import acos, asin, atan, cos, pi, sin, sqrt, tan

from numba import njit
import numpy as np

import py_pore_flow.constants as c

TRIANGLE = 0
SQUARE = 1
CIRCLE = 2

WATER = 0
OIL = 1

MAX_NUMBER_OF_CORNERS = 4


@njit
def calculate_drainage_network_saturation(shape, c, pc, a, oil_filled):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_drainage_element_saturation(shape[i], c[i], pc, a[i], oil_filled[i])
    return sw, so


@njit
def calculate_drainage_element_saturation(shape, C_r, current_Pc, A, oil_filled):
    if not oil_filled:
        return 1.0, 0.0

    Pc = current_Pc

    if shape == CIRCLE:
        Aw = 0
    elif Pc == 0:
        Aw = A
    else:
        R = calculate_r(Pc)
        Aw = R**2 * C_r

    if Aw < A:
        Sw = Aw / A
    else:
        Sw = 1
    So = 1 - Sw

    return Sw, So


@njit
def calculate_imbibition_network_saturation(
    shape, a, oil_filled, theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer
):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_imbibition_element_saturation(
            a[i],
            oil_filled[i],
            theta_a[i],
            theta_h[i],
            beta[i],
            bi[i],
            bo[i],
            n_corners[i],
            has_oil_layer[i],
        )
    return sw, so


@njit
def calculate_imbibition_element_saturation(A, oil_filled, theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer):
    if oil_filled:
        Aw = calculate_water_layer_area(bi, beta, theta_h, n_corners)
        if Aw < 0:
            Sw = 0
        if Aw < A:
            Sw = Aw / A
        else:
            Sw = 1
        So = 1 - Sw
    else:
        Ao = calculate_element_oil_layer_area(theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer)
        if Ao < 0:
            So = 0
        elif Ao < A:
            So = Ao / A
        else:
            So = 1.0
        Sw = 1.0 - So

    return Sw, So


@njit
def calculate_element_oil_layer_area(theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer):
    Ao = 0
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            Ao += calculate_corner_oil_layer_area(bi[i], bo[i], beta[i], theta_h[i], theta_a)
    return Ao


@njit
def calculate_corner_oil_layer_area(bi, bo, beta, theta_i, theta_o):
    Ao = (bo * sin(beta) / cos(theta_o - beta)) ** 2 * (
        (cos(theta_o) * cos(theta_o - beta) / sin(beta)) - theta_o + beta + pi / 2
    )
    Ai = (bi * sin(beta) / cos(theta_i + beta)) ** 2 * (
        (cos(theta_i) * cos(theta_i + beta) / sin(beta)) + theta_i + beta - pi / 2
    )

    Al = Ao - Ai

    return Al


@njit
def calculate_water_layer_area(b, beta, theta, n_corners):
    Aw = 0.0
    for i in range(int(n_corners)):
        Ac = ((b[i] * sin(beta[i])) / cos(theta[i] + beta[i])) ** 2 * (
            cos(theta[i]) * cos(theta[i] + beta[i]) / sin(beta[i]) + theta[i] + beta[i] - pi / 2
        )
        Aw += Ac
    return Aw


@njit
def calculate_sw_total(Sw, V, Vclay, total_volume):
    sum = 0
    for i in range(len(Sw)):
        sum += Sw[i] * V[i] + Vclay[i]
    Sw_total = sum / total_volume
    return Sw_total


@njit
def calculate_beta(G, shape):
    beta = np.full(MAX_NUMBER_OF_CORNERS, 0, dtype=np.float64)

    if shape == TRIANGLE:
        b2_min = atan(2 / sqrt(3) * cos(acos(-12 * sqrt(3) * G) / 3 + 4 * pi / 3))
        b2_max = atan(2 / sqrt(3) * cos(acos(-12 * sqrt(3) * G) / 3))
        b2 = np.random.uniform(b2_min, b2_max)
        b1 = -b2 / 2 + (1 / 2) * asin(((tan(b2) + 4 * G) / (tan(b2) - 4 * G)) * sin(b2))
        b3 = pi / 2 - b1 - b2

        beta[0] = b1
        beta[1] = b2
        beta[2] = b3
    elif shape == SQUARE:
        for i in range(4):
            beta[i] = pi / 4

    return beta


@njit
def calculate_sides(beta, radius, n_corners):
    sides = np.full(MAX_NUMBER_OF_CORNERS, 0, dtype=np.float64)
    if n_corners == 3:
        half_lengths = radius / np.tan(beta[:n_corners])
        sides[0] = half_lengths[0] + half_lengths[1]
        sides[1] = half_lengths[1] + half_lengths[2]
        sides[2] = half_lengths[2] + half_lengths[0]
    elif n_corners == 4:
        sides = np.full(MAX_NUMBER_OF_CORNERS, 2 * radius, dtype=np.float64)
    return sides


@njit
def calculate_max_b(sides, n_corners):
    if n_corners == 0:
        return 0.0
    else:
        return np.max(sides[:n_corners])


@njit
def calculate_shape(G):
    """
    Args:
        G: shape_factor

    Return:
        TRIANGLE, SQUARE or CIRCLE
    """
    if G > 0 and G <= sqrt(3) / 36:
        return TRIANGLE, G
    elif G <= 1 / 16:
        return SQUARE, 1 / 16
    # elif G == 1/(4*pi):
    else:
        return CIRCLE, 1 / (4 * pi)


@njit
def calculate_area(r, G):
    A = (r**2) / (4 * G)
    return A


@njit
def calculate_C(theta, beta, n_corners):
    """
    Args:
        theta: receding_contact_angle
        beta: corner_half_angles
    Return:
        C
    """
    C = 0
    for i in range(int(n_corners)):
        if beta[i] < pi / 2 - theta[i]:
            C += cos(theta[i]) * (cos(theta[i] + beta[i]) / sin(beta[i])) - (pi / 2 - theta[i] - beta[i])

    return C


@njit
def calculate_hinging_contact_angle(Pc, beta, b, n_corners):
    theta_h = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)

    if Pc == 0:
        for i in range(int(n_corners)):
            theta_h[i] = pi / 2 - beta[i]
    else:
        R = calculate_r(Pc)
        if R == 0:
            theta_h[i] = 0
        else:
            for i in range(int(n_corners)):
                x = b[i] * sin(beta[i]) / R
                if x > 1:
                    x = 1
                elif x < -1:
                    x = -1
                theta_h[i] = acos(x) - beta[i]
    return theta_h


@njit
def calculate_drainage_b(beta, theta, n_corners, Pc_max):
    Rmin = c.sigma / Pc_max

    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(int(n_corners)):
        if theta[i] + beta[i] > pi / 2:
            b[i] = 0.0
        else:
            b[i] = Rmin * cos(theta[i] + beta[i]) / sin(beta[i])
    return b


@njit
def calculate_imbibition_network_b(beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, pc, Rmin):
    number_of_elements = len(beta)

    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_b(
            beta[i], max_b[i], theta_r[i], theta_a[i], theta_h[i], has_water_layer[i], n_corners[i], Rmin, pc
        )
    return b


@njit
def calculate_imbibition_element_b(beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, Rmin, pc):
    b = np.full(MAX_NUMBER_OF_CORNERS, 0, dtype=np.float64)
    R = calculate_r(pc, max_b)
    for i in range(int(n_corners)):
        if has_water_layer[i]:
            b[i] = calculate_imbibition_corner_b(beta[i], theta_r, theta_a, theta_h[i], R, Rmin)
    return b


@njit
def calculate_imbibition_corner_b(beta, theta_r, theta_a, theta_h, R, Rmin):
    if theta_h - theta_a > -1e-10:
        return calculate_corner_bi(beta, theta_a, R)
    else:
        return calculate_corner_bi(beta, theta_r, Rmin)


@njit
def calculate_drainage_corner_b(beta, theta_r, Rmin):
    return calculate_corner_bi(beta, theta_r, Rmin)


@njit
def calculate_corner_bi(beta, theta, R):
    if R < 0:
        return 0
    b = R * cos(theta + beta) / sin(beta)
    return max(0, b)


@njit
def calculate_imbibition_network_bo(beta, max_b, theta_a, has_oil_layer, n_corners, pc):
    number_of_elements = len(beta)
    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_bo(beta[i], max_b[i], theta_a[i], has_oil_layer[i], n_corners[i], pc)
    return b


@njit
def calculate_imbibition_element_bo(beta, max_b, theta_a, has_oil_layer, n_corners, pc):
    R = calculate_r(pc, max_b)
    b = np.full(MAX_NUMBER_OF_CORNERS, 0, dtype=np.float64)
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            b[i] = calculate_corner_bo(beta[i], theta_a, R)
    return b


@njit
def calculate_corner_bo(beta, theta, R):
    b = R * cos(theta - beta) / sin(beta)
    return max(0, b)


@njit
def calculate_r(pc, limit=None):
    if limit is None:
        R = c.sigma / pc
    else:
        if pc == 0:
            R = limit
        else:
            R = c.sigma / pc
            R = min(R, limit)
    return R


@njit
def calculate_oil_layer(theta_a, beta, n_corners, pc):
    if pc > 0:
        return np.full(MAX_NUMBER_OF_CORNERS, False, dtype=np.bool_)

    corner_oil_layer = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.bool_)
    for i in range(int(n_corners)):
        corner_oil_layer[i] = theta_a > pi / 2 + beta[i]
    return corner_oil_layer


@njit
def check_water_layer(has_oil_layer, filled_with_oil):
    if filled_with_oil:
        return np.full(MAX_NUMBER_OF_CORNERS, True, dtype=np.bool_)
    else:
        return has_oil_layer.copy()


@njit
def check_water_layer_network(corner_oil_layer, filled_with_oil):
    number_of_elements = len(filled_with_oil)
    corner_water_layer = np.full((number_of_elements, MAX_NUMBER_OF_CORNERS), False, np.bool_)
    for i in range(number_of_elements):
        corner_water_layer[i] = check_water_layer(corner_oil_layer[i], filled_with_oil[i])
    return corner_water_layer


@njit
def calculate_oil_layer_collapse_pc(beta, theta_a, b):
    Pc = c.sigma * cos(acos(2 * sin(beta) + cos(theta_a)) + beta) / (b * sin(beta))
    return Pc


@njit
def calculate_oil_layer_element(corner_oil_layer, beta, theta_a, b, n_corners, pc):
    for i in range(int(n_corners)):
        if corner_oil_layer[i]:
            if b[i] == 0:
                corner_oil_layer[i] = False
            else:
                collapse_pc = calculate_oil_layer_collapse_pc(beta[i], theta_a, b[i])
                if pc < collapse_pc:
                    corner_oil_layer[i] = False
    return corner_oil_layer


@njit
def calculate_oil_layer_network(corner_oil_layer, beta, theta_a, b, n_corners, pc):
    number_of_elements = len(theta_a)
    for i in range(number_of_elements):
        corner_oil_layer[i] = calculate_oil_layer_element(
            corner_oil_layer[i],
            beta[i],
            theta_a[i],
            b[i],
            n_corners[i],
            pc,
        )
    return corner_oil_layer


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
