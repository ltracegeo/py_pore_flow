from math import acos, cos, pi, sin, sqrt, tan

from numba import njit
import numpy as np

import py_pore_flow.constants as c
from py_pore_flow.equations.geometry import CIRCLE, calculate_corner_C, MAX_NUMBER_OF_CORNERS


@njit
def cot(x):
    return 1 / tan(x)


@njit
def calculate_pc(shape_factor, radius, theta, shape, beta, n_corners):
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
def calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, beta, n_corners):
    theta = receding_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, beta, n_corners)


@njit
def calculate_imbibition_pc(
    shape_factor, radius, advancing_contact_angle, shape, pc, beta, n_corners, number_of_oil_filled_neighbors, K
):
    if number_of_oil_filled_neighbors > 1 and pc > 0:
        return calculate_pore_body_filling_pc(radius, advancing_contact_angle, K, number_of_oil_filled_neighbors, 0)
    else:
        return calculate_imbibition_piston_like_displacement_pc(
            shape_factor, radius, advancing_contact_angle, shape, pc, beta, n_corners
        )


@njit
def calculate_imbibition_piston_like_displacement_pc(
    shape_factor, radius, advancing_contact_angle, shape, pc, beta, n_corners
):
    if pc < 0:
        theta = pi - advancing_contact_angle
        return -calculate_pc(shape_factor, radius, theta, shape, beta, n_corners)
    else:
        theta = advancing_contact_angle
        return calculate_pc(shape_factor, radius, theta, shape, beta, n_corners)


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
        lambda_sum = lamb[i] * x[i] - (c.pw - c.po) * c.g * h

    Pc = 2 * c.sigma * cos(theta_a) / r - c.sigma * lambda_sum

    return Pc


@njit
def calculate_snap_off_pc_network(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    number_of_elements = len(r)

    Pc = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        if n_corners[i] != 0:
            Pc[i] = calculate_snap_off_pc_element(r[i], beta[i], theta_a[i], theta_r[i], n_corners[i], Rmin, h[i])
        else:
            Pc[i] = 0.0
    return Pc


@njit
def calculate_snap_off_pc_element(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    if theta_a < (pi / 2 - np.min(beta[:n_corners])):
        Pc = calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmin, h)
    else:
        Pc = calculate_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmin, h)
    return Pc


@njit
def calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    i0 = np.argmin(beta[:n_corners])
    i1 = (i0 + 1) % n_corners
    i2 = (i0 - 1) % n_corners
    if beta[i1] > beta[i2]:
        i1, i2 = i2, i1
    Pc_0 = (
        c.sigma
        * (cos(theta_a) * cot(beta[i0]) - sin(theta_a))
        / (r * (cot(beta[i0]) + cot(beta[i1])) - Rmin * (cos(theta_r) * cot(beta[i1]) - sin(theta_r)))
        - (c.pw - c.po) * c.g * h
    )
    Pc_1 = c.sigma / r * (cos(theta_a) - (2 * sin(theta_a) / (cot(beta[i0]) + cot(beta[i1])))) - (c.pw - c.po) * c.g * h
    Pc = max(Pc_0, Pc_1)
    return Pc


@njit
def calculate_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmin, h):
    i = np.argmin(beta[:n_corners])
    if theta_a <= (pi - beta[i]):
        Pc = (c.sigma * cos(theta_a + beta[i]) / (Rmin * cos(theta_r + beta[i]))) - (c.pw - c.po) * c.g * h
    else:
        Pc = (-c.sigma / (Rmin * cos(theta_r + beta[i]))) - (c.pw - c.po) * c.g * h
    return Pc


@njit
def calculate_spontaneous_imbibition_viability(theta_a, theta_r, beta, n_corners, r, G, Rmin):
    angle_sum = 0.0
    for i in range(int(n_corners)):
        angle_sum += cos(theta_r + beta[i])

    return theta_a <= acos((-4 * G * angle_sum) / (r / Rmin - cos(theta_r) + 12 * G * sin(theta_r)))


@njit
def calculate_snap_off_viability(theta_r, beta, n_corners):
    if n_corners > 0:
        beta_min = np.min(beta[:n_corners])
        return theta_r + beta_min < pi / 2
    else:
        return False
