from math import acos, asin, atan, cos, pi, sin, sqrt, tan

from numba import njit
import numpy as np

from .common_equations import CIRCLE, OIL, SQUARE, TRIANGLE, WATER, MAX_NUMBER_OF_CORNERS
from .pnflow_equations import calculate_element_multiphase_oil_conductance, calculate_element_singlephase_conductance
from py_pore_flow.constants import g, mio, miw, po, pw, sigma


@njit
def cot(x):
    return 1 / tan(x)


# Drainage


@njit
def calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, C_r, height_above_datum):
    theta = receding_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, C_r, height_above_datum)


@njit
def calculate_pc(shape_factor, radius, theta, shape, C, height_above_datum):
    G = shape_factor
    r = radius
    h = height_above_datum

    if shape == CIRCLE:
        Fd = 1
    else:
        Fd = 1 + sqrt(1 - (4 * G * C / cos(theta) ** 2))

    Pc = sigma * cos(theta) / r * (1 + 2 * sqrt(pi * G) / Fd - (pw - po) * g * h)

    return Pc


@njit
def calculate_network_saturation(shape, c, pc, a, oil_filled):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_drainage_displacement(shape[i], c[i], pc, a[i], oil_filled[i])
    return sw, so


@njit
def calculate_drainage_displacement(shape, C_r, current_Pc, A, oil_filled):
    if not oil_filled:
        return 1.0, 0.0

    Pc = current_Pc

    if shape == CIRCLE:
        Aw = 0
    elif Pc == 0:
        Aw = A
    else:
        R = sigma / Pc
        Aw = R**2 * C_r

    if Aw < A:
        Sw = Aw / A
    else:
        Sw = 1
    So = 1 - Sw

    return Sw, So


@njit
def calculate_sw_total(Sw, V, Vclay):
    sum_a = 0
    sum_b = 0

    for i in range(len(Sw)):
        sum_a += Sw[i] * V[i] + Vclay[i]
        sum_b += V[i]
    Sw_total = sum_a / sum_b
    return Sw_total


@njit
def calculate_beta(G, shape):
    beta = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)

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
def calculate_shape(G):
    """
    Args:
        G: shape_factor

    Return:
        TRIANGLE, SQUARE or CIRCLE
    """
    if G > 0 and G < sqrt(3) / 36:
        return TRIANGLE
    elif G <= 1 / 16:
        return SQUARE
    # elif G == 1/(4*pi):
    else:
        return CIRCLE


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
    for i in range(n_corners):
        if beta[i] < pi / 2 - theta[i]:
            C += cos(theta[i]) * (cos(theta[i] + beta[i]) / sin(beta[i])) - (pi / 2 - theta[i] - beta[i])

    return C


def flow_rate(L, A, G, shape, phase, P, h):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = [None] * 3
    for n in [i, j, k]:
        if phase[n] == WATER:
            mi = miw
        else:
            mi = mio
        gp[n] = calculate_element_singlephase_conductance(A[n], G[n], shape[n], mi)

    fi = [None] * 2
    for n in [i, k]:
        if phase[n] == WATER:
            pp = pw
        else:
            pp = po
        fi[n] = P[n] - pp * g * h

    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    q = Gp * (fi[i] - fi[k])

    return q


@njit
def calculate_element_multiphase_water_conductance(theta_h, b, beta, n_corners):
    gw = 0
    for i in range(n_corners):
        Ac = ((b[i] * sin(beta[i])) / cos(theta_h[i] + beta[i])) ** 2 * (
            cos(theta_h[i] + beta[i]) / sin(beta[i]) + theta_h[i] + beta[i] - pi / 2
        )
        Gc = Ac / (
            4 * b[i] ** 2 * (1 - sin(beta[i]) / cos(theta_h[i] + beta[i]) * (theta_h[i] + beta[i] - pi / 2)) ** 2
        )
        G2 = sin(beta[i]) * cos(beta[i]) / 4 * (1 + sin(beta[i])) ** 2
        Cw = 0.364 + 0.28 * (G2 / Gc)
        gw += Cw * (Ac**2 * Gc / miw)
    return gw


@njit
def calculate_singlephase_conductance(L, A, G, shape, phase=WATER):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = np.empty(3, dtype=np.float64)
    mi = miw if phase == WATER else mio
    for n in [i, j, k]:
        gp[n] = calculate_element_singlephase_conductance(A[n], G[n], shape[n], mi)
    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    return Gp


@njit
def calculate_multiphase_network_oil_conductance(L, A, G, shape, So, number_of_elements):
    multiphase_conductance = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        multiphase_conductance[i] = calculate_multiphase_oil_conductance(L[i], A[i], G[i], shape[i], So[i])
    return multiphase_conductance


@njit
def calculate_multiphase_oil_conductance(L, A, G, shape, So):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = np.empty(3, dtype=np.float64)
    for n in [i, j, k]:
        if So[n] == 1:
            gp[n] = calculate_element_singlephase_conductance(A[n], G[n], shape[n], mio)
        else:
            gp[n] = calculate_element_multiphase_oil_conductance(A[n], G[n], shape[n], So[n])
    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    return Gp


def calculate_multiphase_network_water_conductance(L, A, G, shape, Sw, theta_h, b, beta, n_corners, number_of_elements):
    multiphase_conductance = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        multiphase_conductance[i] = calculate_multiphase_water_conductance(
            L[i], A[i], G[i], shape[i], Sw[i], theta_h[i], b[i], beta[i], n_corners[i]
        )
    return multiphase_conductance


def calculate_multiphase_water_conductance(L, A, G, shape, Sw, theta_h, b, beta, n_corners):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = np.empty(3, dtype=np.float64)
    for n in [i, j, k]:
        if Sw[n] == 1:
            gp[n] = calculate_element_singlephase_conductance(A[n], G[n], shape[n], miw)
        else:
            gp[n] = calculate_element_multiphase_water_conductance(theta_h[n], b[n], beta[n], n_corners[n])
    return calculate_conductance(L, gp)


@njit
def calculate_conductance(L, gp):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    return Gp


@njit
def calculate_hinging_contact_angle(Pc, beta, b, n_corners):
    theta_h = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)

    if Pc == 0:
        return theta_h

    R = sigma / Pc
    for i in range(n_corners):
        theta_h[i] = acos(b[i] * sin(beta[i]) / R) - beta[i]
    return theta_h


@njit
def calculate_drainage_b(beta, theta, n_corners, Pc_max):
    Rmin = sigma / Pc_max

    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(n_corners):
        b[i] = Rmin * cos(theta[i] + beta[i]) / sin(beta[i])
    return b


@njit
def calculate_imbibition_network_b(beta, theta_r, theta_a, theta_h, n_corners, pc, Rmin):
    number_of_elements = len(beta)
    R = sigma / pc

    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_b(beta[i], theta_r[i], theta_a[i], theta_h[i], n_corners[i], R, Rmin)


@njit
def calculate_imbibition_element_b(beta, theta_r, theta_a, theta_h, n_corners, R, Rmin):
    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(n_corners):
        b[i] = calculate_imbibition_corner_b(beta[i], theta_r, theta_a, theta_h[i], R, Rmin)
    return b


@njit
def calculate_imbibition_corner_b(beta, theta_r, theta_a, theta_h, R, Rmin):
    if theta_h >= theta_a:
        return calculate_corner_b(beta, theta_a, R)
    else:
        return calculate_corner_b(beta, theta_r, Rmin)


@njit
def calculate_drainage_corner_b(beta, theta_r, Rmin):
    return calculate_corner_b(beta, theta_r, Rmin)


@njit
def calculate_corner_b(beta, theta, R):
    b = R * cos(theta + beta) / sin(beta)
    return b


# Imbibition


@njit
def calculate_imbibition_pc(
    shape_factor, radius, advancing_contact_angle, shape, C_a, pc, height_above_datum, number_of_oil_filled_neighbors, K
):
    if number_of_oil_filled_neighbors > 1 and pc > 0:
        return calculate_pore_body_filling_pc(radius, advancing_contact_angle, K, number_of_oil_filled_neighbors, 0)
    else:
        return calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, C_a, pc, height_above_datum
        )


@njit
def calculate_imbibition_piston_like_displacement_pc(
    shape_factor, radius, advancing_contact_angle, shape, C_a, pc, height_above_datum
):
    if pc < 0:
        theta = pi - advancing_contact_angle
    else:
        theta = advancing_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, C_a, height_above_datum)


@njit
def calculate_absolute_permeability(phase, qts, A, L, fi_inlet, fi_outlet):
    if phase == WATER:
        mi = miw
    else:
        mi = mio
    K = (mi * qts * L) / (A * (fi_inlet - fi_outlet))
    return K


@njit
def calculate_pore_body_filling_pc(r, theta_a, K, oil_filled_throats_n, h):
    x = [0] * oil_filled_throats_n
    for i in range(oil_filled_throats_n):
        x[i] = np.random.uniform(0, 1)

    lamb = [0] * oil_filled_throats_n
    for i in range(oil_filled_throats_n):
        if i == 0:
            lamb[i] = 0
        else:
            lamb[i] = 0.03 / sqrt(K / (1.01325 * 10**15))

    for i in range(oil_filled_throats_n):
        lambda_sum = lamb[i] * x[i] - (pw - po) * g * h

    Pc = 2 * sigma * cos(theta_a) / r - sigma * lambda_sum

    return Pc


@njit
def calculate_snap_off_pc(r, beta, theta_a, theta_r, theta_h, Rmin, pc, h):
    if pc > 0:
        Pc_0 = (
            sigma
            / r
            * (cos(theta_a) * cot(beta[0]) - sin(theta_a) + cos(theta_h[2]) * cot(beta[2]) - sin(theta_h[1]))
            / (cot(beta[0]) + cot(beta[1]))
            - (pw - po) * g * h
        )
        Pc_1 = sigma / r * (cos(theta_a) - (2 * sin(theta_a) / (cot(beta[0]) + cot(beta[1])))) - (pw - po) * g * h
        Pc = max(Pc_0, Pc_1)
    else:
        new_Pc = (sigma * cos(theta_a + beta[0]) / Rmin * cos(theta_r + beta[0])) - (pw - po) * g * h
        if new_Pc <= (pi - beta[0]):
            Pc = new_Pc
        else:
            Pc = (-sigma / Rmin * cos(theta_r + beta[0])) - (pw - po) * g * h

    return Pc


@njit
def calculate_r(pc):
    R = sigma / pc
    return R


@njit
def calculate_oil_layer(theta_a, beta, n_corners):
    corner_oil_layer = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.bool_)
    for i in range(n_corners):
        corner_oil_layer[i] = theta_a > pi / 2 + beta[i]
    return corner_oil_layer


@njit
def calculate_oil_layer_collapse_pc(beta, theta_a, b):
    Pc = sigma * cos(acos(2 * sin(beta) + cos(theta_a)) + beta) / (b * sin(beta))
    return Pc


@njit
def calculate_oil_layer_element(corner_oil_layer, beta, theta_a, b, n_corners, pc):
    for i in range(n_corners):
        if corner_oil_layer[i]:
            collapse_pc = calculate_oil_layer_collapse_pc(beta[i], theta_a, b[i])
            if pc > collapse_pc:
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
