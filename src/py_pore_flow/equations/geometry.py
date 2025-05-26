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


@njit(cache=True)
def calculate_drainage_network_saturation(shape, c, pc, a, oil_filled):
    number_of_elements = len(shape)
    sw = np.empty(number_of_elements, np.float64)
    so = np.empty(number_of_elements, np.float64)
    for i in range(number_of_elements):
        sw[i], so[i] = calculate_drainage_element_saturation(shape[i], c[i], pc, a[i], oil_filled[i])
    return sw, so


@njit(cache=True)
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


@njit(cache=True)
def calculate_imbibition_network_saturation(a, oil_filled, theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer):
    number_of_elements = len(a)
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


@njit(cache=True)
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
            if has_oil_layer[:n_corners].any():
                So = 1e-5
            else:
                So = 0
        elif Ao < A:
            So = Ao / A
        else:
            So = 1.0
        Sw = 1.0 - So

    return Sw, So


@njit(cache=True)
def calculate_element_oil_layer_area(theta_a, theta_h, beta, bi, bo, n_corners, has_oil_layer):
    Ao = 0
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            Ao += calculate_corner_oil_layer_area(bi[i], bo[i], beta[i], theta_h[i], theta_a)
    return Ao


@njit(cache=True)
def calculate_corner_oil_layer_area(bi, bo, beta, theta_i, theta_o):
    Ao = (bo * sin(beta) / cos(theta_o - beta)) ** 2 * (
        (cos(theta_o) * cos(theta_o - beta) / sin(beta)) - theta_o + beta + pi / 2
    )
    Ai = (bi * sin(beta) / cos(theta_i + beta)) ** 2 * (
        (cos(theta_i) * cos(theta_i + beta) / sin(beta)) + theta_i + beta - pi / 2
    )

    Al = Ao - Ai

    return Al


@njit(cache=True)
def calculate_water_layer_area(b, beta, theta, n_corners):
    Aw = 0.0
    for i in range(int(n_corners)):
        Ac = ((b[i] * sin(beta[i])) / cos(theta[i] + beta[i])) ** 2 * (
            cos(theta[i]) * cos(theta[i] + beta[i]) / sin(beta[i]) + theta[i] + beta[i] - pi / 2
        )
        Aw += Ac
    return Aw


@njit(cache=True)
def calculate_sw_total(Sw, V, Vclay, N, total_volume):
    sum = 0
    for i in range(len(Sw)):
        sum += (Sw[i] * V[i] + Vclay[i]) * N[i]
    Sw_total = sum / total_volume
    return Sw_total


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def calculate_max_b(sides, n_corners):
    if n_corners == 0:
        return 0.0
    else:
        return np.max(sides[:n_corners])


@njit(cache=True)
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


@njit(cache=True)
def calculate_area(r, G):
    A = (r**2) / (4 * G)
    return A


@njit(cache=True)
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


@njit(cache=True)
def calculate_network_hinging_contact_angle(
    Pc,
    beta,
    b,
    n_corners,
    theta_r,
    theta_a,
    theta_h,
    trapped,
):
    number_of_elements = len(beta)

    for i in range(number_of_elements):
        if not trapped[i]:
            calculate_hinging_contact_angle(Pc, beta[i], b[i], n_corners[i], theta_r[i], theta_a[i], theta_h[i])
    return theta_h


@njit(cache=True)
def calculate_hinging_contact_angle(Pc, beta, b, n_corners, theta_r, theta_a, theta_hi):
    if Pc == 0:
        for i in range(int(n_corners)):
            theta_hi[i] = pi / 2 - beta[i]
    else:
        R = calculate_r(Pc)
        if R == 0:
            theta_hi[:] = 0
        else:
            for i in range(int(n_corners)):
                x = b[i] * sin(beta[i]) / R
                if x > 1:
                    x = 1
                elif x < -1:
                    x = -1
                theta_hi[i] = acos(x) - beta[i]
    theta_hi[:] = np.clip(theta_hi, theta_r, theta_a)


@njit(cache=True)
def calculate_drainage_b(beta, theta, n_corners, max_b, Pc_max):
    Rmin = calculate_r(Pc_max, max_b)

    b = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.float64)
    for i in range(int(n_corners)):
        if theta[i] + beta[i] > pi / 2:
            b[i] = 0.0
        else:
            b[i] = Rmin * cos(theta[i] + beta[i]) / sin(beta[i])
    return b


@njit(cache=True)
def calculate_imbibition_network_b(
    b, trapped, beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, pc, Rmin
):
    number_of_elements = len(beta)

    for i in range(number_of_elements):
        if not trapped[i]:
            b[i] = calculate_imbibition_element_b(
                b[i], beta[i], max_b[i], theta_r[i], theta_a[i], theta_h[i], has_water_layer[i], n_corners[i], Rmin, pc
            )


@njit(cache=True)
def calculate_imbibition_element_b(b, beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, Rmin, pc):
    R = calculate_r(pc, max_b)
    for i in range(int(n_corners)):
        if has_water_layer[i]:
            b[i] = calculate_imbibition_corner_b(b[i], beta[i], theta_r, theta_a, theta_h[i], R, Rmin)
    return b


@njit(cache=True)
def calculate_imbibition_corner_b(b, beta, theta_r, theta_a, theta_h, R, Rmin):
    if theta_h - theta_a > -1e-10:
        return calculate_corner_bi(beta, theta_a, R)
    else:
        return b


@njit(cache=True)
def calculate_2nd_drainage_network_b(
    b, trapped, beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, pc, Rmin
):
    number_of_elements = len(beta)

    for i in range(number_of_elements):
        if not trapped[i]:
            b[i] = calculate_2nd_drainage_element_b(
                b[i], beta[i], max_b[i], theta_r[i], theta_a[i], theta_h[i], has_water_layer[i], n_corners[i], Rmin, pc
            )


@njit(cache=True)
def calculate_2nd_drainage_element_b(b, beta, max_b, theta_r, theta_a, theta_h, has_water_layer, n_corners, Rmin, pc):
    R = calculate_r(pc, max_b)
    for i in range(int(n_corners)):
        if has_water_layer[i]:
            b[i] = calculate_2nd_drainage_corner_b(b[i], beta[i], theta_r, theta_a, theta_h[i], R, Rmin)
        else:
            b[i] = 0
    return b


@njit(cache=True)
def calculate_2nd_drainage_corner_b(b, beta, theta_r, theta_a, theta_h, R, Rmin):
    if theta_r - theta_h > -1e-10:
        return calculate_corner_bi(beta, theta_h, R)
    else:
        return b


@njit(cache=True)
def calculate_drainage_corner_b(beta, theta_r, Rmin):
    return calculate_corner_bi(beta, theta_r, Rmin)


@njit(cache=True)
def calculate_corner_bi(beta, theta, R):
    if R < 0:
        return 0
    b = R * cos(theta + beta) / sin(beta)
    return max(0, b)


@njit(cache=True)
def calculate_imbibition_network_bo(bo, trapped, beta, max_b, theta_a, has_oil_layer, n_corners, pc):
    number_of_elements = len(beta)
    for i in range(number_of_elements):
        if not trapped[i]:
            bo[i] = calculate_imbibition_element_bo(beta[i], max_b[i], theta_a[i], has_oil_layer[i], n_corners[i], pc)


@njit(cache=True)
def calculate_imbibition_element_bo(beta, max_b, theta_a, has_oil_layer, n_corners, pc):
    R = calculate_r(pc, max_b)
    b = np.full(MAX_NUMBER_OF_CORNERS, 0, dtype=np.float64)
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            b[i] = calculate_corner_bo(beta[i], theta_a, R)
    return b


@njit(cache=True)
def calculate_corner_bo(beta, theta, R):
    b = R * cos(theta - beta) / sin(beta)
    return max(0, b)


@njit(cache=True)
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


@njit(cache=True)
def calculate_oil_layer(theta_a, beta, n_corners, pc):
    if pc > 0:
        return np.full(MAX_NUMBER_OF_CORNERS, False, dtype=np.bool_)

    corner_oil_layer = np.empty(MAX_NUMBER_OF_CORNERS, dtype=np.bool_)
    for i in range(int(n_corners)):
        corner_oil_layer[i] = theta_a > pi / 2 + beta[i]
    return corner_oil_layer


@njit(cache=True)
def check_water_layer(has_oil_layer, filled_with_oil):
    if filled_with_oil:
        return np.full(MAX_NUMBER_OF_CORNERS, True, dtype=np.bool_)
    else:
        return has_oil_layer.copy()


@njit(cache=True)
def check_water_layer_network(corner_water_layer, trapped, corner_oil_layer, filled_with_oil):
    number_of_elements = len(filled_with_oil)
    for i in range(number_of_elements):
        if not trapped[i]:
            corner_water_layer[i] = check_water_layer(corner_oil_layer[i], filled_with_oil[i])


@njit(cache=True)
def calculate_oil_layer_collapse_pc(beta, theta_a, b):
    Pc = c.sigma * cos(acos(2 * sin(beta) + cos(theta_a)) + beta) / (b * sin(beta))
    return Pc


@njit(cache=True)
def calculate_oil_layer_element(corner_oil_layer, oil_layer_collapse_pc, b, n_corners, pc):
    for i in range(int(n_corners)):
        if corner_oil_layer[i]:
            if b[i] == 0:
                corner_oil_layer[i] = False
            else:
                if pc < oil_layer_collapse_pc[i]:
                    corner_oil_layer[i] = False
    return corner_oil_layer


@njit(cache=True)
def calculate_oil_layer_network(corner_oil_layer, oil_layer_collapse_pc, trapped, b, n_corners, pc):
    number_of_elements = len(trapped)
    for i in range(number_of_elements):
        if not trapped[i]:
            corner_oil_layer[i] = calculate_oil_layer_element(
                corner_oil_layer[i],
                oil_layer_collapse_pc[i],
                b[i],
                n_corners[i],
                pc,
            )

@njit(cache=True)
def calculate_2nd_drainage_oil_layer_existance_element(corner_oil_layer, bo, n_corners):
    for i in range(int(n_corners)):
        if corner_oil_layer[i]:
            if bo[i] == 0:
                corner_oil_layer[i] = False
    return corner_oil_layer


@njit(cache=True)
def calculate_2nd_drainage_oil_layer_existance_network(corner_oil_layer, bo, n_corners):
    number_of_elements = len(n_corners)
    for i in range(number_of_elements):
        corner_oil_layer[i] = calculate_2nd_drainage_oil_layer_existance_element(
            corner_oil_layer[i],
            bo[i],
            n_corners[i],
        )


@njit(cache=True)
def calculate_dimensionless_layer_area(b_tilde, beta, theta_i, theta_o):
    Ao = (sin(beta) / cos(theta_o - beta)) ** 2 * (
        (cos(theta_o) * cos(theta_o - beta) / sin(beta)) - theta_o + beta + pi / 2
    )
    Ai = (b_tilde * sin(beta) / cos(theta_i + beta)) ** 2 * (
        (cos(theta_i) * cos(theta_i + beta) / sin(beta)) + theta_i + beta - pi / 2
    )

    Al = Ao - Ai

    return Al


@njit(cache=True)
def calculate_corner_C(theta, beta):
    return cos(theta) * (cos(theta + beta) / sin(beta)) - (pi / 2 - theta - beta)


@njit(cache=True)
def calculate_oil_layer_collapse_pc_network(oil_layer_collapse_pc, beta, theta_a, theta_r, n_corners, Rmin):
    number_of_elements = len(theta_a)
    for i in range(number_of_elements):
        calculate_oil_layer_collapse_pc_element(
            oil_layer_collapse_pc[i], beta[i], theta_a[i], theta_r[i], n_corners[i], Rmin
        )


@njit(cache=True)
def calculate_oil_layer_collapse_pc_element(oil_layer_collapse_pc, beta, theta_a, theta_r, n_corners, Rmin):
    for i in range(int(n_corners)):
        oil_layer_collapse_pc[i] = calculate_oil_layer_collapse_pc_corner(beta[i], theta_a, theta_r, Rmin)


@njit(cache=True)
def calculate_oil_layer_collapse_pc_corner(beta, theta_a, theta_r, Rmin):
    if theta_a > pi / 2 + beta:
        if 2 * sin(beta) + cos(theta_a) > 1:
            Pc = c.sigma * cos(acos(1) + beta) / (Rmin * cos(theta_r + beta))
        else:
            Pc = c.sigma * cos(acos(2 * sin(beta) + cos(theta_a)) + beta) / (Rmin * cos(theta_r + beta))
    else:
        Pc = 0
    return Pc


@njit(cache=True)
def calculate_network_outter_hinging_contact_angle(
    Pc,
    beta,
    b,
    n_corners,
    theta_r,
    theta_a,
    theta_ho,
    trapped,
    has_oil_layer,
    max_b,
):
    number_of_elements = len(beta)

    for i in range(number_of_elements):
        if not trapped[i]:
            calculate_outter_hinging_contact_angle(
                Pc, beta[i], b[i], n_corners[i], theta_r[i], theta_a[i], theta_ho[i], has_oil_layer[i], max_b[i]
            )
    return theta_ho


@njit(cache=True)
def calculate_outter_hinging_contact_angle(Pc, beta, b, n_corners, theta_r, theta_a, theta_ho, has_oil_layer, max_b):
    R = calculate_r(Pc, max_b)
    for i in range(int(n_corners)):
        if has_oil_layer[i]:
            x = b[i] * sin(beta[i]) / R
            if x > 1:
                x = 1
            elif x < -1:
                x = -1
            theta_ho[i] = acos(x) + beta[i]
    theta_ho[:] = np.clip(theta_ho, theta_r, theta_a)


@njit(cache=True)
def calculate_2nd_drainage_network_bo(
    b, trapped, beta, max_b, theta_r, theta_a, theta_ho, has_water_layer, n_corners, pc
):
    number_of_elements = len(beta)

    for i in range(number_of_elements):
        if not trapped[i]:
            b[i] = calculate_2nd_drainage_element_bo(
                b[i], beta[i], max_b[i], theta_r[i], theta_a[i], theta_ho[i], has_water_layer[i], n_corners[i], pc
            )


@njit(cache=True)
def calculate_2nd_drainage_element_bo(b, beta, max_b, theta_r, theta_a, theta_ho, has_water_layer, n_corners, pc):
    R = calculate_r(pc, max_b)
    for i in range(int(n_corners)):
        if has_water_layer[i]:
            b[i] = calculate_2nd_drainage_corner_bo(b[i], beta[i], theta_r, theta_a, theta_ho[i], R)
        else:
            b[i] = 0
    return b


@njit(cache=True)
def calculate_2nd_drainage_corner_bo(b, beta, theta_r, theta_a, theta_ho, R):
    if theta_r - theta_ho > -1e-10:
        return calculate_corner_bo(beta, theta_r, R)
    else:
        return b


@njit(cache=True)
def calculate_multiphase_mask_drainage(is_multiphase, n_corners, b, oil_filled):
    for i in range(len(is_multiphase)):
        if oil_filled[i] and np.any(b[i][: n_corners[i]] > 0):
            is_multiphase[i] = True


@njit(cache=True)
def calculate_multiphase_mask_imbibition(
    is_multiphase,
    n_corners,
    oil_collapse_pc,
    water_filled,
    oil_filled,
    trapped,
    trapped_evacuation,
    corner_oil_layer,
    oil_filled_multiphase,
    pc,
):
    for i in range(len(is_multiphase)):
        if (n_corners[i] == 0) or trapped[i] or (is_multiphase[i] and trapped_evacuation[i]):
            continue
        if water_filled[i]:
            if np.all(pc < oil_collapse_pc[i][: n_corners[i]]):
                is_multiphase[i] = False
            elif np.any(corner_oil_layer[i][: n_corners[i]]):
                is_multiphase[i] = True
            else:
                is_multiphase[i] = False
        elif oil_filled[i] and not oil_filled_multiphase[i]:
            is_multiphase[i] = False
        else:
            is_multiphase[i] = True


@njit(cache=True)
def calculate_multiphase_mask_2nd_drainage(
    is_multiphase,
    n_corners,
    water_filled,
    oil_filled,
    trapped,
    trapped_evacuation,
    water_filled_multiphase,
    oil_filled_multiphase,
    water_corner_collapse_pressure,
    b_pinned,
    pc,
):
    for i in range(len(is_multiphase)):
        if (n_corners[i] == 0) or trapped[i] or (is_multiphase[i] and trapped_evacuation[i]):
            continue
        if oil_filled[i]:
            if oil_filled_multiphase[i]:
                is_multiphase[i] = True
            elif (b_pinned[i][: n_corners[i]] > 0).any() and pc < np.max(
                water_corner_collapse_pressure[i][: n_corners[i]]
            ):
                is_multiphase[i] = True
            else:
                is_multiphase[i] = False
        elif water_filled[i] and water_filled_multiphase[i]:
            is_multiphase[i] = True
        else:
            is_multiphase[i] = False


@njit(cache=True)
def calculate_water_filled_multiphase(filled_multiphase, n_corners, corner_oil_layer, bo):
    for i in range(filled_multiphase.size):
        if n_corners[i] == 0:
            filled_multiphase[i] = False
        else:
            # filled_multiphase[i] = np.any(corner_oil_layer[i][: n_corners[i]]) and np.any(bo[i][: n_corners[i]]) > 0
            filled_multiphase[i] = False
            for corner in range(n_corners[i]):
                if corner_oil_layer[i][corner] and (bo[i][corner] > 0):
                    filled_multiphase[i] = True
                    break



@njit(cache=True)
def calculate_imbibion_oil_filled_multiphase(filled_multiphase, n_corners, b_pinned):
    for i in range(filled_multiphase.size):
        if n_corners[i] == 0:
            filled_multiphase[i] = False
        else:
            filled_multiphase[i] = np.any(b_pinned[i][: n_corners[i]] > 0)


@njit(cache=True)
def calculate_2nd_drainage_oil_filled_multiphase(filled_multiphase, n_corners, theta_r, beta, b_pinned, oil_filled):
    for i in range(filled_multiphase.size):
        if n_corners[i] == 0:
            filled_multiphase[i] = False
        else:
            # if oil_filled[i]:
            #     filled_multiphase[i] = (theta_r[i] + np.min(beta[i][: n_corners[i]]) < pi / 2) or np.any(
            #         b_pinned[i][: n_corners[i]] > 0
            #     )
            # else:
            filled_multiphase[i] = theta_r[i] + np.min(beta[i][: n_corners[i]]) < pi / 2


@njit(cache=True)
def calculate_water_corner_collapse_pc(water_corner_collapse_pressure, theta_r, n_corners, b_pinned, beta):
    number_of_elements = len(water_corner_collapse_pressure)
    for i in range(number_of_elements):
        for j in range(n_corners[i]):
            Pc = (cos(theta_r[i] + beta[i][j]) * c.sigma) / (b_pinned[i][j] * sin(beta[i][j]))
            water_corner_collapse_pressure[i][j] = Pc
