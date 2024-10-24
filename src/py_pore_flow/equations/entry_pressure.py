from math import acos, cos, pi, sin, sqrt, tan

from numba import njit
import numpy as np

import py_pore_flow.constants as c
from py_pore_flow.equations.geometry import CIRCLE, calculate_corner_C


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
def calculate_snap_off_pc(r, beta, theta_a, theta_r, theta_h, n_corners, Rmin, pc, h):
    if theta_a < (pi / 2 - np.min(beta[:n_corners])):
        Pc = calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_h, n_corners, h)
    else:
        Pc = calculate_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmin, h)
    return Pc


@njit
def calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_h, n_corners, h):
    i0 = np.argmin(beta[:n_corners])
    i1 = (i0 + 1) % n_corners
    i2 = (i0 - 1) % n_corners
    if beta[i1] > beta[i2]:
        i1, i2 = i2, i1

    Pc_0 = calculate_two_moving_menisci_pc(r, beta[i0], beta[i1], theta_a)
    Pc_1 = calculate_single_moving_meniscus_pc(r, beta[i0], beta[i2], beta[i1], theta_a, theta_h[i1])
    Pc = max(Pc_0, Pc_1)
    return Pc


@njit
def calculate_two_moving_menisci_pc(r, beta1, beta2, theta_a):
    Pc = c.sigma / r * (cos(theta_a) - (2 * sin(theta_a) / (cot(beta1) + cot(beta2))))
    return Pc


@njit
def calculate_single_moving_meniscus_pc(r, beta1, beta2, beta3, theta_a, theta_h3):
    Pc = (
        c.sigma
        / r
        * (cos(theta_a) * cot(beta1) - sin(theta_a) + cos(theta_h3) * cot(beta3) - sin(theta_h3))
        / (cot(beta1) + cot(beta2))
    )
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
