import math
from math import acos, asin, atan, cos, log, pi, sin, sqrt, tan
import sys

from numba import njit
import numpy as np

from .common_equations import (
    CIRCLE,
    OIL,
    SQUARE,
    TRIANGLE,
    WATER,
    MAX_NUMBER_OF_CORNERS,
    calculate_dimensionless_layer_area,
)

# from .starfish_equations import calculate_pc
from .valvatne_equations import (
    calculate_element_multiphase_oil_conductance,
    calculate_element_singlephase_conductance,
    calculate_pc,
)
from py_pore_flow.constants import g, mio, miw, po, pw, sigma


@njit
def cot(x):
    return 1 / tan(x)


# Drainage


@njit
def calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, C_r, beta, n_corners):
    theta = receding_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, C_r, beta, n_corners)


@njit
def calculate_drainage_network_saturation(shape, r, c, pc, a, oil_filled):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_drainage_displacement(shape[i], r[i], c[i], pc, a[i], oil_filled[i])
    return sw, so


@njit
def calculate_drainage_displacement(shape, r, C_r, current_Pc, A, oil_filled):
    if not oil_filled:
        return 1.0, 0.0

    Pc = current_Pc

    if shape == CIRCLE:
        Aw = 0
    elif Pc == 0:
        Aw = A
    else:
        R = calculate_r(Pc, r)
        Aw = R**2 * C_r

    if Aw < A:
        Sw = Aw / A
    else:
        Sw = 1
    So = 1 - Sw

    return Sw, So


@njit
def calculate_imbibition_network_saturation(
    shape, r, c, pc, a, oil_filled, theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer
):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_imbibition_displacement(
            shape[i],
            c[i],
            pc,
            a[i],
            oil_filled[i],
            theta_a[i],
            theta_h[i],
            beta[i],
            bi[i],
            bo[i],
            n_corners[i],
            r[i],
            has_oil_layer[i],
        )
    return sw, so


@njit
def calculate_imbibition_displacement(
    shape, C_h, pc, A, oil_filled, theta_a, theta_h, beta, bi, bo, n_corners, r, has_oil_layer
):
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
        R = calculate_r(pc, r)
        Ao = calculate_oil_layer_area(theta_a, theta_h, beta, bi, bo, n_corners, R, has_oil_layer)
        if Ao < 0:
            So = 0
        elif Ao < A:
            So = Ao / A
        else:
            So = 1.0
        Sw = 1.0 - So

    return Sw, So


@njit
def calculate_oil_layer_area(theta_a, theta_h, beta, bi, bo, n_corners, R, has_oil_layer):
    Ao = 0
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            b_tilde = bi[i] / bo[i]
            Ao += calculate_dimensionless_layer_area(b_tilde, beta[i], theta_h[i], theta_a)
    return Ao


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
def calculate_corner_C(theta, beta):
    return cos(theta) * (cos(theta + beta) / sin(beta)) - (pi / 2 - theta - beta)


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
    for i in range(int(n_corners)):
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
    for i in range(int(n_corners)):
        if b[i] == 0.0:
            continue

        Ac = ((b[i] * sin(beta[i])) / cos(theta_h[i] + beta[i])) ** 2 * (
            cos(theta_h[i]) * cos(theta_h[i] + beta[i]) / sin(beta[i]) + theta_h[i] + beta[i] - pi / 2
        )
        Gc = Ac / (
            4 * b[i] ** 2 * (1 - sin(beta[i]) / cos(theta_h[i] + beta[i]) * (theta_h[i] + beta[i] - pi / 2)) ** 2
        )
        G2 = sin(beta[i]) * cos(beta[i]) / (4 * (1 + sin(beta[i])) ** 2)
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
def calculate_multiphase_network_oil_conductance(
    L, A, G, shape, So, b, bo, beta, theta_h, theta_a, n_corners, has_oil_in_center, number_of_elements
):
    multiphase_conductance = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        multiphase_conductance[i] = calculate_multiphase_oil_conductance(
            L[i],
            A[i],
            G[i],
            shape[i],
            So[i],
            b[i],
            bo[i],
            beta[i],
            theta_h[i],
            theta_a[i],
            n_corners[i],
            has_oil_in_center[i],
        )
    return multiphase_conductance


@njit
def calculate_multiphase_oil_conductance(
    L, A, G, shape, So, b, bo, beta, theta_h, theta_a, n_corners, has_oil_in_center
):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = np.empty(3, dtype=np.float64)
    for n in [i, j, k]:
        if has_oil_in_center[n]:
            gp[n] = So[n] * calculate_element_singlephase_conductance(A[n], G[n], shape[n], mio)
        else:
            gp[n] = calculate_element_multiphase_oil_conductance(
                b[n], bo[n], beta[n], theta_h[n], theta_a[n], n_corners[n]
            )
    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    return Gp


@njit
def calculate_multiphase_network_water_conductance(
    L, A, G, shape, Sw, theta_h, b, beta, n_corners, has_water_in_center, number_of_elements, throat_id_list
):
    multiphase_conductance = np.empty(number_of_elements, np.float64)
    for i in throat_id_list:
        multiphase_conductance[i] = calculate_multiphase_water_conductance(
            L[i], A[i], G[i], shape[i], Sw[i], theta_h[i], b[i], beta[i], n_corners[i], has_water_in_center[i]
        )
    return multiphase_conductance


@njit
def calculate_multiphase_water_conductance(L, A, G, shape, Sw, theta_h, b, beta, n_corners, has_water_in_center):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    gp = np.empty(3, dtype=np.float64)
    for n in [i, j, k]:
        if has_water_in_center[n]:
            gp[n] = Sw[n] * calculate_element_singlephase_conductance(A[n], G[n], shape[n], miw)
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
def calculate_hinging_contact_angle(Pc, r, beta, b, n_corners):
    theta_h = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)

    R = calculate_r(Pc, r)
    for i in range(int(n_corners)):
        theta_h[i] = acos(b[i] * sin(beta[i]) / R) - beta[i]
    return theta_h


@njit
def calculate_drainage_b(beta, theta, n_corners, Pc_max):
    Rmin = sigma / Pc_max

    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(int(n_corners)):
        if theta[i] + beta[i] > pi / 2:
            b[i] = 0.0
        else:
            b[i] = Rmin * cos(theta[i] + beta[i]) / sin(beta[i])
    return b


@njit
def calculate_imbibition_network_b(r, beta, theta_r, theta_a, theta_h, n_corners, pc, Rmin):
    number_of_elements = len(beta)

    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_b(r[i], beta[i], theta_r[i], theta_a[i], theta_h[i], n_corners[i], Rmin, pc)
    return b


@njit
def calculate_imbibition_element_b(r, beta, theta_r, theta_a, theta_h, n_corners, Rmin, pc):
    R = calculate_r(pc, r)
    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(int(n_corners)):
        b[i] = calculate_imbibition_corner_b(beta[i], theta_r, theta_a, theta_h[i], R, Rmin)
    return b


@njit
def calculate_imbibition_corner_b(beta, theta_r, theta_a, theta_h, R, Rmin):
    if theta_h >= theta_a:
        return calculate_corner_bi(beta, theta_a, R)
    else:
        return calculate_corner_bi(beta, theta_r, Rmin)


@njit
def calculate_drainage_corner_b(beta, theta_r, Rmin):
    return calculate_corner_bi(beta, theta_r, Rmin)


@njit
def calculate_corner_bi(beta, theta, R):
    b = R * cos(theta + beta) / sin(beta)
    if b < 0:
        b = 0.0
    return b


@njit
def calculate_imbibition_network_bo(r, beta, theta_a, has_oil_layer, n_corners, pc):
    number_of_elements = len(beta)
    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_bo(r[i], beta[i], theta_a[i], has_oil_layer[i], n_corners[i], pc)
    return b


@njit
def calculate_imbibition_element_bo(r, beta, theta_a, has_oil_layer, n_corners, pc):
    R = calculate_r(pc, r)
    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            b[i] = calculate_corner_bo(beta[i], theta_a, R)
    return b


@njit
def calculate_corner_bo(beta, theta, R):
    b = R * cos(theta - beta) / sin(beta)
    if b < 0:
        b = 0.0
    return b


# Imbibition


@njit
def calculate_imbibition_pc(
    shape_factor, radius, advancing_contact_angle, shape, C_a, pc, beta, n_corners, number_of_oil_filled_neighbors, K
):
    if number_of_oil_filled_neighbors > 1 and pc > 0:
        return calculate_pore_body_filling_pc(radius, advancing_contact_angle, K, number_of_oil_filled_neighbors, 0)
    else:
        return calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, C_a, pc, beta, n_corners
        )


@njit
def calculate_imbibition_piston_like_displacement_pc(
    shape_factor, radius, advancing_contact_angle, shape, C_a, pc, beta, n_corners
):
    if pc < 0:
        theta = pi - advancing_contact_angle
    else:
        theta = advancing_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, C_a, beta, n_corners)


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
def calculate_r(pc, r):
    if pc == 0:
        R = r
    else:
        R = sigma / pc

    if R > r:
        R = r
    elif R < -r:
        R = -r
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
def calculate_oil_layer_collapse_pc(beta, theta_a, b):
    Pc = sigma * cos(acos(2 * sin(beta) + cos(theta_a)) + beta) / (b * sin(beta))
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
def calculate_spontaneous_imbibition_viability(theta_a, theta_r, beta, n_corners, r, G, Rmin):
    angle_sum = 0.0
    for i in range(int(n_corners)):
        angle_sum += cos(theta_r + beta[i])

    return theta_a <= acos((-4 * G * angle_sum) / (r / Rmin - cos(theta_r) + 12 * G * sin(theta_r)))
