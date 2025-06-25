from math import acos, cos, pi, asin, sin, sqrt, tan

from numba import njit
import numpy as np

import py_pore_flow.constants as c
from py_pore_flow.equations.geometry import CIRCLE, calculate_corner_C, MAX_NUMBER_OF_CORNERS


@njit(cache=True)
def cot(x):
    return 1 / tan(x)


@njit(cache=True)
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


@njit(cache=True)
def calculate_drainage_pc(shape_factor, radius, receding_contact_angle, shape, beta, n_corners):
    theta = receding_contact_angle
    return calculate_pc(shape_factor, radius, theta, shape, beta, n_corners)


@njit(cache=True)
def calculate_imbibition_piston_like_displacement_pc_network(
    shape_factor, r, theta_a, theta_r, shape, beta, n_corners, Rmin
):
    number_of_elements = len(r)

    Pc = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        if n_corners[i] != 0:
            Pc[i] = calculate_imbibition_piston_like_displacement_pc_element(
                shape_factor[i],
                r[i],
                theta_a[i],
                theta_r[i],
                beta[i],
                n_corners[i],
                Rmin,
            )
            if Pc[i] < 0:
                Pc[i] = -calculate_pc(shape_factor[i], r[i], pi - theta_a[i], shape[i], beta[i], n_corners[i])
        else:
            Pc[i] = calculate_pc(shape_factor[i], r[i], theta_a[i], shape[i], beta[i], n_corners[i])

    return Pc


@njit(cache=True)
def calculate_imbibition_piston_like_displacement_pc_element(
    shape_factor, radius, advancing_contact_angle, receding_contact_angle, beta, n_corners, Rmin
):
    b = np.empty(n_corners, np.float64)
    theta_h = np.empty(n_corners, np.float64)
    arc_meniscus = [i for i in range(n_corners) if beta[i] < pi / 2 - receding_contact_angle]
    a = 0
    A = radius**2 / (4 * shape_factor)
    R = Rmin
    Rold = R + 1000
    while abs((R - Rold) / R) > 0.001 and a < 100:
        for i in range(n_corners):
            b[i] = Rmin * cos(receding_contact_angle + beta[i]) / sin(beta[i])
            arc = b[i] * np.sin(beta[i]) / R
            arc = np.clip(np.array(arc), -1, 1)
            theta_h[i] = np.arccos(arc) - beta[i]

            if theta_h[i] > advancing_contact_angle:
                b[i] = R * cos(advancing_contact_angle + beta[i]) / sin(beta[i])
                theta_h[i] = advancing_contact_angle

        Aeff = A - R**2 * sum(
            [
                cos(theta_h[i]) * cos(theta_h[i] + beta[i]) / sin(beta[i]) + theta_h[i] + beta[i] - pi / 2.0
                for i in arc_meniscus
            ]
        )
        Los = radius / (2 * shape_factor) - 2 * sum([b[i] for i in arc_meniscus])
        arc = np.clip(np.array([b[i] * sin(beta[i]) / R for i in arc_meniscus]), -1, 1)
        Low = 2 * R * sum([asin(arc[i]) for i in arc_meniscus])
        Rold = R
        R = Aeff / (Low + Los * cos(advancing_contact_angle))
        a += 1

    return c.sigma / R


@njit(cache=True)
def calculate_pore_body_filling_blunt1_pc(r, theta_a, oil_filled_throats_n, throats_radii, G):
    x = np.random.uniform(0, 1, oil_filled_throats_n)

    lambda_sum = 0
    for i in range(oil_filled_throats_n):
        if i < 6:
            lambda_sum += c.pore_fill_weights[i] * x[i]
        else:
            lambda_sum += c.pore_fill_weights[-1] * x[i]

    Pc = 2 * c.sigma * cos(theta_a) / (r + lambda_sum)

    return Pc


@njit(cache=True)
def calculate_pore_body_filling_blunt2_pc(r, theta_a, oil_filled_throats_n, throats_radii, G):
    x = np.random.uniform(0, 1, oil_filled_throats_n)

    lambda_sum = 0
    for i in range(oil_filled_throats_n):
        if i < 6:
            lambda_sum += c.pore_fill_weights[i] * x[i]
        else:
            lambda_sum += c.pore_fill_weights[-1] * x[i]

    Pc = 2 * c.sigma * cos(theta_a) / r - c.sigma * lambda_sum

    return Pc


@njit(cache=True)
def calculate_pore_body_filling_oren1_pc(r, theta_a, oil_filled_throats_n, throats_radii, G):
    x = np.random.uniform(0, 1, oil_filled_throats_n)

    lambda_sum = 0
    for i in range(oil_filled_throats_n):
        if i < 6:
            lambda_sum += c.pore_fill_weights[i] * x[i] * throats_radii[i]
        else:
            lambda_sum += c.pore_fill_weights[-1] * x[i] * throats_radii[i]

    Pc = 2 * c.sigma * cos(theta_a) / (r + lambda_sum)

    return Pc


@njit(cache=True)
def calculate_pore_body_filling_oren2_pc(r, theta_a, oil_filled_throats_n, throats_radii, G):
    x = np.random.uniform(0, 1, oil_filled_throats_n)

    lambda_sum = 0
    for i in range(oil_filled_throats_n):
        if i < 6:
            lambda_sum += c.pore_fill_weights[i] * x[i] * throats_radii[i]
        else:
            lambda_sum += c.pore_fill_weights[-1] * x[i] * throats_radii[i]

    Pc = c.sigma * cos(theta_a) * (1 + 2 * np.sqrt(pi * G)) / (r + lambda_sum)

    return Pc


BLUNT1 = 0
BLUNT2 = 1
OREN1 = 2
OREN2 = 3


PORE_BODY_FILLING_STRATEGY = {
    "blunt1": BLUNT1,
    "blunt2": BLUNT2,
    "oren1": OREN1,
    "oren2": OREN2,
}


@njit(cache=True)
def calculate_pore_body_filling_pc(strategy, r, theta_a, oil_filled_throats_n, throats_radii, G):
    if strategy == BLUNT1:
        return calculate_pore_body_filling_blunt1_pc(r, theta_a, oil_filled_throats_n, throats_radii, G)
    elif strategy == BLUNT2:
        return calculate_pore_body_filling_blunt2_pc(r, theta_a, oil_filled_throats_n, throats_radii, G)
    elif strategy == OREN1:
        return calculate_pore_body_filling_oren1_pc(r, theta_a, oil_filled_throats_n, throats_radii, G)
    else:
        return calculate_pore_body_filling_oren2_pc(r, theta_a, oil_filled_throats_n, throats_radii, G)


@njit(cache=True)
def calculate_snap_off_pc_network(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    number_of_elements = len(r)

    Pc = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        if n_corners[i] != 0:
            Pc[i] = calculate_snap_off_pc_element(r[i], beta[i], theta_a[i], theta_r[i], n_corners[i], Rmin, h[i])
        else:
            Pc[i] = -c.sigma / Rmin
    return Pc


@njit(cache=True)
def calculate_snap_off_pc_element(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    if theta_a < (pi / 2 - np.min(beta[:n_corners])):
        Pc = calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmin, h)
    else:
        Pc = calculate_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmin, h)
    return Pc


@njit(cache=True)
def calculate_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmin, h):
    i0 = np.argmin(beta[:n_corners])
    i1 = (i0 + 1) % n_corners
    i2 = (i0 - 1) % n_corners
    if beta[i1] > beta[i2]:
        i1, i2 = i2, i1
    Pc_0 = (
        c.sigma
        * (cos(theta_a) * cot(beta[i0]) - sin(theta_a))
        / (r * (cot(beta[i0]) + cot(beta[i2])) - Rmin * (cos(theta_r) * cot(beta[i2]) - sin(theta_r)))
        - (c.pw - c.po) * c.g * h
    )
    Pc_1 = c.sigma / r * (cos(theta_a) - (2 * sin(theta_a) / (cot(beta[i0]) + cot(beta[i1])))) - (c.pw - c.po) * c.g * h
    Pc = max(Pc_0, Pc_1)
    return Pc


@njit(cache=True)
def calculate_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmin, h):
    i = np.argmin(beta[:n_corners])
    if theta_a <= (pi - beta[i]):
        Pc = (c.sigma * cos(theta_a + beta[i]) / (Rmin * cos(theta_r + beta[i]))) - (c.pw - c.po) * c.g * h
    else:
        Pc = (-c.sigma / (Rmin * cos(theta_r + beta[i]))) - (c.pw - c.po) * c.g * h
    return Pc


@njit(cache=True)
def calculate_imbibition_snap_off_viability(theta_r, beta, n_corners):
    if n_corners > 0:
        beta_min = np.min(beta[:n_corners])
        return theta_r + beta_min < pi / 2
    else:
        return False


@njit(cache=True)
def calculate_2nd_drainage_snap_off_viability(snap_off_viability, corner_oil_layer, n_conns):
    for i in range(len(snap_off_viability)):
        snap_off_viability[i] = corner_oil_layer[i][: n_conns[i]].any()


@njit(cache=True)
def calculate_2nd_drainage_piston_like_displacement_pc_network(
    shape_factor, r, theta_r, theta_a, corner_oil_layer, n_corners, bo, shape, beta, Rmax
):
    number_of_elements = len(r)

    Pc = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        Pc[i] = calculate_2nd_drainage_piston_like_displacement_pc_element(
            shape_factor[i],
            r[i],
            theta_r[i],
            theta_a[i],
            corner_oil_layer[i],
            bo[i],
            shape[i],
            beta[i],
            n_corners[i],
            Rmax,
        )

    return Pc


@njit(cache=True)
def calculate_2nd_drainage_piston_like_displacement_pc_element(
    G, r, theta_r, theta_a, corner_oil_layer, bo, shape, beta, n_corners, Rmax
):
    corner_has_oil_layer = corner_oil_layer[:n_corners]
    if len(corner_has_oil_layer) > 0 and corner_has_oil_layer.any():
        Pc = calculate_pc(G, r, theta_r, shape, beta, n_corners)
    else:
        Fd = 1
        theta = theta_r
        Pc = c.sigma * cos(theta) / r * (1 + 2 * sqrt(pi * G)) * Fd
    return Pc


@njit(cache=True)
def calculate_2nd_drainage_snap_off_pc_network(r, beta, theta_a, theta_r, n_corners, Rmax, h):
    number_of_elements = len(r)

    Pc = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        if n_corners[i] != 0:
            Pc[i] = calculate_2nd_drainage_snap_off_pc_element(
                r[i], beta[i], theta_a[i], theta_r[i], n_corners[i], Rmax, h[i]
            )
        else:
            Pc[i] = c.sigma / Rmax
    return Pc


@njit(cache=True)
def calculate_2nd_drainage_snap_off_pc_element(r, beta, theta_a, theta_r, n_corners, Rmax, h):
    if pi - theta_r < (pi / 2 - np.min(beta[:n_corners])):
        Pc = calculate_2nd_drainage_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmax)
    else:
        Pc = calculate_2nd_drainage_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmax)
    return Pc


@njit(cache=True)
def calculate_2nd_drainage_waterwet_snapoff_pc(r, beta, theta_a, theta_r, n_corners, Rmax):
    i0 = np.argmin(beta[:n_corners])
    i1 = (i0 + 1) % n_corners
    i2 = (i0 - 1) % n_corners
    if beta[i1] > beta[i2]:
        i1, i2 = i2, i1
    Pc_0 = c.sigma / (
        (sin(beta[i0]) / cos(theta_r - beta[i0]))
        * (r * (cot(beta[i0]) + cot(beta[i1])) - (Rmax * (cos(theta_a - beta[i1]) / sin(beta[i1]))))
    )
    Pc_1 = (
        (c.sigma / r)
        * ((cos(theta_r - beta[i0]) / sin(beta[i0])) + (cos(theta_r - beta[i1]) / sin(beta[i1])))
        / (cot(beta[i0]) + cot(beta[i1]))
    )
    Pc = min(Pc_0, Pc_1)
    return Pc


@njit(cache=True)
def calculate_2nd_drainage_oilwet_snapoff_pc(beta, theta_a, theta_r, n_corners, Rmax):
    i = np.argmin(beta[:n_corners])
    if theta_r > beta[i]:
        Pc = c.sigma * cos(theta_r - beta[i]) / (Rmax * cos(theta_a - beta[i]))
    else:
        Pc = c.sigma / (Rmax * cos(theta_a - beta[i]))
    return Pc
