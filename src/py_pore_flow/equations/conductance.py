import math
from math import cos, log, pi, sin

from numba import njit
import numpy as np

import py_pore_flow.constants as c
from py_pore_flow.equations.geometry import CIRCLE, TRIANGLE, WATER, calculate_dimensionless_layer_area


@njit(cache=True)
def calculate_element_singlephase_conductivity(A, G, shape, mi):
    if shape == CIRCLE:
        v = 0.5
    elif shape == TRIANGLE:
        v = 0.6
    else:
        v = 0.5623
    g = v * (A**2 * G) / (mi)
    return g


@njit(cache=True)
def calculate_element_multiphase_oil_conductivity(bi, bo, beta, theta_i, theta_o, corner_oil_layer, n_corners):
    go = 0.0
    for i in range(n_corners):
        if corner_oil_layer[i]:
            go += calculate_dimensionless_layer_conductivity(bi[i], bo[i], beta[i], theta_i[i], theta_o[i])
    return go


@njit(cache=True)
def calculate_dimensionless_layer_conductivity(bi, bo, beta, theta_i, theta_o):
    if bo == 0:
        return 0

    b_tilde = bi / bo

    Al = calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o)

    Loow = 2 * sin(beta) / cos(theta_o - beta) * (pi / 2 - theta_o + beta)
    Liow = 2 * b_tilde * sin(beta) / cos(theta_i + beta) * (pi / 2 - theta_i - beta)

    Gl = Al / ((Loow + Liow + 2 * (1 - b_tilde)) ** 2)

    a1, a2, a3 = get_tabulated_conductance_constants(beta)

    gl = math.e ** (a1 * log(Al**3 * Gl) ** 2 + a2 * log(Al**3 * Gl) + a3)

    gpl = bo**4 * gl / c.mio

    return gpl


@njit(cache=True)
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


@njit(cache=True)
def calculate_element_multiphase_water_conductivity(theta_h, b, beta, n_corners):
    gw = 0
    for i in range(int(n_corners)):
        if b[i] == 0.0:
            continue

        Ac = ((b[i] * sin(beta[i])) / cos(theta_h[i] + beta[i])) ** 2 * (
            cos(theta_h[i]) * cos(theta_h[i] + beta[i]) / sin(beta[i]) + theta_h[i] + beta[i] - pi / 2
        )
        if Ac <= 0.0:
            continue
        Gc = Ac / (
            4 * b[i] ** 2 * (1 - sin(beta[i]) / cos(theta_h[i] + beta[i]) * (theta_h[i] + beta[i] - pi / 2)) ** 2
        )
        G2 = sin(beta[i]) * cos(beta[i]) / (4 * (1 + sin(beta[i])) ** 2)
        Cw = 0.364 + 0.28 * (G2 / Gc)
        gw += Cw * (Ac**2 * Gc / c.miw)
    return gw


@njit(cache=True)
def calculate_conductance(L, gp):
    i = 0  # Pore 1
    j = 2  # Throat connecting pore 0 and 1
    k = 1  # Pore 2

    Gp = 1 / ((L[i] / gp[i]) + (L[j] / gp[j]) + (L[k] / gp[k]))
    return Gp


@njit(cache=True)
def calculate_absolute_permeability(phase, qts, A, L, fi_inlet, fi_outlet):
    if phase == WATER:
        mi = c.miw
    else:
        mi = c.mio
    K = (mi * qts * L) / (A * (fi_inlet - fi_outlet))
    return K


@njit(cache=True)
def calculate_network_singlephase_conductivity(gp, A, G, shape, phase=WATER):
    number_of_elements = len(gp)

    mi = c.miw if phase == WATER else c.mio
    for n in range(number_of_elements):
        gp[n] = calculate_element_singlephase_conductivity(A[n], G[n], shape[n], mi)


@njit(cache=True)
def calculate_network_multiphase_conductivity(
    gp,
    A,
    G,
    shape,
    Sw,
    So,
    b,
    bo,
    beta,
    theta_h,
    theta_a,
    corner_oil_layer,
    n_corners,
    has_water_in_center,
    has_oil_in_center,
    phase,
):
    if phase == WATER:
        calculate_network_multiphase_water_conductivity(
            gp, A, G, shape, Sw, theta_h, b, beta, n_corners, has_water_in_center
        )
    else:
        calculate_network_multiphase_oil_conductivity(
            gp, A, G, shape, So, b, bo, beta, theta_h, theta_a, corner_oil_layer, n_corners, has_oil_in_center
        )


@njit(cache=True)
def calculate_network_multiphase_water_conductivity(
    gp, A, G, shape, Sw, theta_h, b, beta, n_corners, has_water_in_center
):
    number_of_elements = len(gp)

    for n in range(number_of_elements):
        if Sw[n] == 0.0:
            gp[n] = 0.0
            continue

        if has_water_in_center[n]:
            gp[n] = Sw[n] * calculate_element_singlephase_conductivity(A[n], G[n], shape[n], c.miw)
        else:
            gp[n] = calculate_element_multiphase_water_conductivity(theta_h[n], b[n], beta[n], n_corners[n])


@njit(cache=True)
def calculate_network_multiphase_oil_conductivity(
    gp, A, G, shape, So, b, bo, beta, theta_h, theta_a, corner_oil_layer, n_corners, has_oil_in_center
):
    number_of_elements = len(gp)

    for n in range(number_of_elements):
        if So[n] == 0.0:
            gp[n] = 0.0
            continue

        if has_oil_in_center[n]:
            gp[n] = So[n] * calculate_element_singlephase_conductivity(A[n], G[n], shape[n], c.mio)
        else:
            gp[n] = calculate_element_multiphase_oil_conductivity(
                b[n], bo[n], beta[n], theta_h[n], theta_a[n], corner_oil_layer[n], n_corners[n]
            )


@njit(cache=True)
def calculate_network_conductance(Gp, throat_gp, throat_L, throat_conn, pore_gp, pore_L):
    number_of_elements = len(Gp)
    for i in range(number_of_elements):
        throat_id = i
        pore_0_id = throat_conn[throat_id][0]
        pore_1_id = throat_conn[throat_id][1]
        gp = np.array([pore_gp[pore_0_id], throat_gp[throat_id], pore_gp[pore_1_id]])
        L = np.array([pore_L[pore_0_id], throat_L[throat_id], pore_L[pore_1_id]])

        if np.any(gp == 0.0):
            Gp[i] = 0
        else:
            Gp[i] = calculate_conductance(L, gp)
