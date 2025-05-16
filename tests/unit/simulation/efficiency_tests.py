# pytest .\tests\unit\simulation\efficiency_tests.py -s
import pandas as pd
import numpy as np
from scipy.stats import truncnorm

from py_pore_flow.equations.geometry import(
    calculate_beta_inplace,
)
from py_pore_flow.equations.entry_pressure import(
    calculate_pc_inplace,
    calculate_pc,
)
from py_pore_flow.simulation.minimum_root_tree import(
    _get_pressure,
    _add_node,
    _pop_root,
    _reorder_tree,
    _check_tree,
)
import py_pore_flow.constants as c

def test_drainage():
    
    pore_df = pd.read_csv('tests\\resources\\1_porosity_map_pore_table.csv')
    throat_df = pd.read_csv('tests\\resources\\1_porosity_map_throat_table.csv')
    #print(pore_df.columns)
    #print(throat_df.columns)
    pore_number = pore_df["pore.all"].size
    throat_number = throat_df["throat.all"].size
    SIGMA = np.float64(30)

    mean = 0.001
    std_dev = 0.0005
    lower = 0.0002
    upper = 0.02 
    a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
    pore_capillary_radius = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=pore_number)
    throat_capillary_radius = truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=throat_number)

    pore_radius = np.where(
        pore_df["pore.phase"] == 1,
        pore_df["pore.inscribed_diameter"] / 2,
        pore_capillary_radius,
    )
    throat_radius = np.where(
        (throat_df["throat.phases_0"] + throat_df["throat.phases_1"]) == 2,
        throat_df["throat.inscribed_diameter"] / 2,
        throat_capillary_radius,
    )

    min_contact_angle = 35.0
    max_contact_angle = 45.0
    pore_intrinsic_contact_angle = np.random.uniform(min_contact_angle, max_contact_angle, pore_number)
    pore_advancing_contact_angle = pore_intrinsic_contact_angle + 5
    pore_receeding_contact_angle = pore_intrinsic_contact_angle - 5
    throat_intrinsic_contact_angle = np.random.uniform(min_contact_angle, max_contact_angle, throat_number)
    throat_advancing_contact_angle = throat_intrinsic_contact_angle + 5
    throat_receeding_contact_angle = throat_intrinsic_contact_angle - 5

    pore_intrinsic_contact_angle = np.deg2rad(pore_intrinsic_contact_angle)
    pore_advancing_contact_angle = np.deg2rad(pore_advancing_contact_angle)
    pore_receeding_contact_angle = np.deg2rad(pore_receeding_contact_angle)
    throat_intrinsic_contact_angle = np.deg2rad(throat_intrinsic_contact_angle)
    throat_advancing_contact_angle = np.deg2rad(throat_advancing_contact_angle)
    throat_receeding_contact_angle = np.deg2rad(throat_receeding_contact_angle)


    pore_shape_factor = pore_df["pore.shape_factor"].to_numpy()
    pore_beta = np.zeros((pore_number, 3), dtype=np.float64)
    throat_shape_factor = throat_df["throat.shape_factor"].to_numpy()
    throat_beta = np.zeros((throat_number, 3), dtype=np.float64)

    calculate_beta_inplace(pore_shape_factor, pore_beta)
    calculate_beta_inplace(throat_shape_factor, throat_beta)

    pore_entry_pc = np.zeros(pore_number, dtype=np.float64)
    calculate_pc_inplace(
        pore_shape_factor, 
        pore_radius, 
        theta=pore_receeding_contact_angle,
        beta=pore_beta,
        pc_array=pore_entry_pc,
        sigma=SIGMA,
        )
    throat_entry_pc = np.zeros(throat_number, dtype=np.float64)
    calculate_pc_inplace(
        throat_shape_factor, 
        throat_radius, 
        theta=throat_receeding_contact_angle,
        beta=throat_beta,
        pc_array=throat_entry_pc,
        sigma=SIGMA,
        )
    
    #calculate_pc(shape_factor, radius, theta, shape, beta, n_corners)
    # for i in range(1000):
    #     shape_factor = pore_shape_factor[i]
    #     if shape_factor <= c.TRIANGLE_THRESHOLD:
    #         shape = c.TRIANGLE
    #         n_corners = 3
    #     elif shape_factor <= c.SQUARE_THRESHOLD:
    #         shape = c.SQUARE
    #         n_corners = 3
    #     else:
    #         shape = c.CIRCLE
    #         n_corners = 0
        # print(calculate_pc(
        #     pore_shape_factor[i], 
        #     pore_radius[i], 
        #     pore_receeding_contact_angle[i], 
        #     shape,
        #     pore_beta[i, :], 
        #     n_corners)
        # )
    
    # Initialize MRT
    pore_n = pore_number
    throat_n = throat_number
    pore_entry_pressure = pore_entry_pc # Index based
    throat_entry_pressure = throat_entry_pc # Index based
    pore_conns_n = np.zeros(pore_n, dtype=np.int16) # Index based
    for i in throat_df["throat.conns_0"]: # Conns is Index based
        pore_conns_n[i] += 1
    for i in throat_df["throat.conns_1"]:
        pore_conns_n[i] += 1
    pore_conns_labels = np.zeros((pore_n, pore_conns_n.max()), dtype=np.int32) # Index based
    pore_conns_n = np.zeros(pore_n, dtype=np.int16) # Index based
    for i_throat, i_pore, j_pore in (
        (i, 
        throat_df["throat.conns_0"][i], 
        throat_df["throat.conns_1"][i],) 
            for i in range(throat_df["throat.conns_0"].size)
    ):
        throat_unified_label = i_throat + pore_n + 1
        i_index = pore_conns_n[i_pore]
        j_index = pore_conns_n[j_pore]
        pore_conns_labels[i_pore, i_index] = throat_unified_label
        pore_conns_labels[j_pore, j_index] = throat_unified_label
        pore_conns_n[i_pore] += 1
        pore_conns_n[j_pore] += 1
    throat_conns_labels = np.stack((throat_df["throat.conns_0"], throat_df["throat.conns_1"]), axis=1)
    throat_conns_labels += 1 # Index based
    tree_invasion_status = np.zeros(pore_n + throat_n + 1, dtype=np.uint8) # Label based
    # 0 : uninvaded
    # 1 : uninvaded, adjacent to invaded
    # 2 : invaded
    tree_element_labels = np.zeros(pore_n + throat_n, dtype=np.uint32) # Index based
    tree_elements_n = np.zeros(1, dtype=np.uint32)

    for i in range(pore_df["pore.xmin"].size):
        if pore_df["pore.xmin"][i] == 1:
            _add_node(
                _get_pressure(
                    i+1,
                    throat_entry_pressure,
                    pore_entry_pressure,
                ),
                i+1,
                tree_element_labels,
                tree_elements_n,
                throat_entry_pressure,
                pore_entry_pressure,
            )
            tree_invasion_status[i+1] = 1
            P = _get_pressure(
                    i+1,
                    throat_entry_pressure,
                    pore_entry_pressure,
                )
            #print(P, i+1, tree_elements_n, tree_element_labels[:10])
    #print(tree_invasion_status.sum())
    while True:
        # print(_check_tree(
        #     tree_element_labels,
        #     tree_elements_n,
        #     throat_entry_pressure,
        #     pore_entry_pressure,
        # ), tree_element_labels[:7], 
        # tree_elements_n, 
        # [_get_pressure(
        #                     i,
        #                     throat_entry_pressure,
        #                     pore_entry_pressure,
        #                 ) for i in tree_element_labels[:7]],
        # )
        invasion_label = _pop_root(
            tree_element_labels,
            tree_elements_n,
            throat_entry_pressure,
            pore_entry_pressure,
            )
        # print(_check_tree(
        #     tree_element_labels,
        #     tree_elements_n,
        #     throat_entry_pressure,
        #     pore_entry_pressure,
        # ), tree_element_labels[:7], 
        # tree_elements_n, 
        # [_get_pressure(
        #                     i,
        #                     throat_entry_pressure,
        #                     pore_entry_pressure,
        #                 ) for i in tree_element_labels[:7]],
        # )
        # print(invasion_label, tree_element_labels.sum(), tree_elements_n)

        if invasion_label == 0:
            break
        tree_invasion_status[invasion_label] = 2
        if invasion_label <= pore_n:
            # print(f"invading pore {invasion_label}", 
            #       pore_conns_n[invasion_label-1],
            #       pore_conns_labels[invasion_label-1, :],
            #     )
            for conn_index in range(pore_conns_n[invasion_label-1]):
                element_label = pore_conns_labels[invasion_label-1, conn_index]
                if tree_invasion_status[element_label] == 0:
                    _add_node(
                        _get_pressure(
                            element_label,
                            throat_entry_pressure,
                            pore_entry_pressure,
                        ),
                        element_label,
                        tree_element_labels,
                        tree_elements_n,
                        throat_entry_pressure,
                        pore_entry_pressure,
                    )
                    tree_invasion_status[element_label] = 1
                    # print(f"Added throat {element_label}, {tree_elements_n}")
        else:
            for conn_index in range(2):
                element_label = throat_conns_labels[invasion_label-pore_n-1, conn_index]
                if tree_invasion_status[element_label] == 0:
                    _add_node(
                        _get_pressure(
                            element_label,
                            throat_entry_pressure,
                            pore_entry_pressure,
                        ),
                        element_label,
                        tree_element_labels,
                        tree_elements_n,
                        throat_entry_pressure,
                        pore_entry_pressure,
                    )
                    tree_invasion_status[element_label] = 1
                    # print(f"Added pore {element_label}, {tree_elements_n}")
        # _reorder_tree(
        #     tree_element_labels,
        #     tree_elements_n,
        #     throat_entry_pressure,
        #     pore_entry_pressure,
        # )
        # print(tree_invasion_status.sum())

    if not _check_tree(
        tree_element_labels,
        tree_elements_n,
        throat_entry_pressure,
        pore_entry_pressure,
    ):
            #pass
            # print(tree_element_labels[:7], tree_elements_n, throat_entry_pressure[:7], pore_entry_pressure[:7])
            raise
    #assert((tree_invasion_status[1:] == 2).all())
    # print(tree_invasion_status.sum())


