import math
from math import acos, asin, atan, cos, log, pi, sin, sqrt, tan
import sys

from numba import njit
import numpy as np

from .common_equations import CIRCLE, OIL, SQUARE, TRIANGLE, WATER, MAX_NUMBER_OF_CORNERS

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
def calculate_drainage_network_saturation(shape, c, pc, a, oil_filled):
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
def calculate_imbibition_network_saturation(
    shape, c, pc, a, oil_filled, theta_a, theta_h, beta, n_corners, has_oil_layer
):
    R = calculate_r(pc)

    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_imbibition_displacement(
            shape[i], c[i], pc, a[i], oil_filled[i], theta_a[i], theta_h[i], beta[i], n_corners[i], R, has_oil_layer[i]
        )
    return sw, so


@njit
def calculate_imbibition_displacement(
    shape, C_h, current_Pc, A, oil_filled, theta_a, theta_h, beta, n_corners, R, has_oil_layer
):
    if oil_filled:
        Sw, So = calculate_drainage_displacement(shape, C_h, current_Pc, A, oil_filled)
    else:
        Ao = calculate_oil_layer_area(theta_a, theta_h, beta, n_corners, R, has_oil_layer)
        if Ao < 0:
            So = 0
        elif Ao < A:
            So = Ao / A
        else:
            So = 1.0
        Sw = 1.0 - So

    return Sw, So


@njit
def calculate_oil_layer_area(theta_a, theta_h, beta, n_corners, R, has_oil_layer):
    Ao = 0
    for i in range(n_corners):
        if has_oil_layer[i]:
            Ao += calculate_dimensionless_layer_area(0.01, beta[i], theta_h[i], theta_a)
    return Ao


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


@njit
def calculate_multiphase_network_water_conductance(
    L, A, G, shape, Sw, theta_h, b, beta, n_corners, number_of_elements, throat_id_list
):
    multiphase_conductance = np.empty(number_of_elements, np.float64)
    for i in throat_id_list:
        multiphase_conductance[i] = calculate_multiphase_water_conductance(
            L[i], A[i], G[i], shape[i], Sw[i], theta_h[i], b[i], beta[i], n_corners[i]
        )
    return multiphase_conductance


@njit
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
    R = calculate_r(pc)

    b = np.empty((number_of_elements, MAX_NUMBER_OF_CORNERS), np.float64)
    for i in range(number_of_elements):
        b[i] = calculate_imbibition_element_b(beta[i], theta_r[i], theta_a[i], theta_h[i], n_corners[i], R, Rmin)
    return b


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
def calculate_r(pc):
    if pc == 0:
        R = np.finfo(np.float64).max
    else:
        R = sigma / pc
    return R


@njit
def calculate_oil_layer(theta_a, beta, n_corners, pc):
    if pc > 0:
        return np.full(MAX_NUMBER_OF_CORNERS, False, dtype=np.bool_)

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
    for i in range(n_corners):
        angle_sum += cos(theta_r + beta[i])

    return theta_a <= acos((-4 * G * angle_sum) / (r / Rmin - cos(theta_r) + 12 * G * sin(theta_r)))


@njit
def calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o):
    Ao = (
        (sin(beta) / cos(theta_o - beta)) ** 2 * (cos(theta_o) * cos(theta_o - beta) / sin(beta))
        - theta_o
        + beta
        + pi / 2
    )
    Ai = (
        (b_tilde * sin(beta) / cos(theta_i + beta)) ** 2 * (cos(theta_i) * cos(theta_i + beta) / sin(beta))
        + theta_i
        + beta
        - pi / 2
    )

    Al = Ao - Ai

    return Al


@njit
def calculate_dimensionless_layer_conductance(bi, bo, beta, theta_i, theta_o):
    b_tilde = bi / bo

    Al = calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o)

    Loow = 2 * sin(beta) / cos(theta_o - beta) * (pi / 2 - theta_o + beta)
    Liow = 2 * b_tilde * sin(beta) / cos(theta_i + beta) * (pi / 2 - theta_i - beta)

    Gl = Al / ((Loow + Liow + 2 * (1 - b_tilde)) ** 2)

    a1, a2, a3 = get_tabulated_conductance_constants(beta)

    gl = math.e ** (a1 * log(Al**3 * Gl) ** 2 + a2 * log(Al**3 * Gl) + a3)

    gpl = bo**4 * gl / mio

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
