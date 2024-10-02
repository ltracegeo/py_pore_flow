from abc import abstractmethod

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    MAX_NUMBER_OF_CORNERS,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
    calculate_hinging_contact_angle,
    calculate_network_hinging_contact_angle,
    calculate_imbibition_element_saturation,
    calculate_imbibition_network_saturation,
    calculate_sw_total,
)
from py_pore_flow.equations.conductance import (
    calculate_absolute_permeability,
    calculate_multiphase_network_oil_conductance,
    calculate_multiphase_network_water_conductance,
    calculate_singlephase_conductance,
)
from py_pore_flow.network import Network

PORE = 0
THROAT = 1


class ConductanceCalculator:
    def __init__(self, network):
        self.network = network

    def calculate_singlephase_conductance(self, phase):
        singlephase_conductance = [0.0] * len(self.network.throat_list)
        for throat_id, throat in enumerate(self.network.throat_list):
            L, A, G, shape = self.__get_singlephase_parameters(throat)
            singlephase_conductance[throat_id] = calculate_singlephase_conductance(L, A, G, shape, phase)
        return singlephase_conductance

    def calculate_multiphase_conductance(self, phase, throat_id_list):
        number_of_throats = len(self.network.pn_core.throat.all)

        if phase == WATER:
            return self.multiphase_water_conductance(
                number_of_throats,
                throat_id_list,
                self.network.pn_core.throat.conns,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                self.network.pn_core.throat.hinging_contact_angle,
                self.network.pn_core.throat.b,
                self.network.pn_core.throat.corner_half_angle,
                self.network.pn_core.throat.n_corners,
                self.network.pn_core.throat.water_saturation,
                self.network.pn_core.throat.water_filled,
                self.network.pn_core.pore.radius,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                self.network.pn_core.pore.hinging_contact_angle,
                self.network.pn_core.pore.b,
                self.network.pn_core.pore.corner_half_angle,
                self.network.pn_core.pore.n_corners,
                self.network.pn_core.pore.water_saturation,
                self.network.pn_core.pore.water_filled,
            )
        else:
            return self.multiphase_oil_conductance(
                number_of_throats,
                throat_id_list,
                self.network.pn_core.throat.conns,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                self.network.pn_core.throat.hinging_contact_angle,
                self.network.pn_core.throat.b,
                self.network.pn_core.throat.corner_half_angle,
                self.network.pn_core.throat.n_corners,
                self.network.pn_core.throat.oil_saturation,
                self.network.pn_core.throat.oil_filled,
                self.network.pn_core.throat.advancing_contact_angle,
                self.network.pn_core.throat.bo,
                self.network.pn_core.throat.corner_oil_layer,
                self.network.pn_core.pore.radius,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                self.network.pn_core.pore.hinging_contact_angle,
                self.network.pn_core.pore.b,
                self.network.pn_core.pore.corner_half_angle,
                self.network.pn_core.pore.n_corners,
                self.network.pn_core.pore.oil_saturation,
                self.network.pn_core.pore.oil_filled,
                self.network.pn_core.pore.advancing_contact_angle,
                self.network.pn_core.pore.bo,
                self.network.pn_core.pore.corner_oil_layer,
            )

    @staticmethod
    @nb.njit
    def multiphase_oil_conductance(
        number_of_throats,
        throat_id_list,
        throat_conns,
        throat_l,
        throat_a,
        throat_g,
        throat_shape,
        throat_thetah,
        throat_b,
        throat_beta,
        throat_ncorners,
        throat_so,
        throat_oilfilled,
        throat_thetaa,
        throat_bo,
        throat_oillayer,
        pore_l,
        pore_a,
        pore_g,
        pore_shape,
        pore_thetah,
        pore_b,
        pore_beta,
        pore_ncorners,
        pore_so,
        pore_oilfilled,
        pore_thetaa,
        pore_bo,
        pore_oillayer,
    ):
        L = np.empty((number_of_throats, 3), np.float64)
        A = np.empty((number_of_throats, 3), np.float64)
        G = np.empty((number_of_throats, 3), np.float64)
        shape = np.empty((number_of_throats, 3), np.float64)
        n_corners = np.empty((number_of_throats, 3), np.int32)
        So = np.empty((number_of_throats, 3), np.float64)
        oil_filled = np.empty((number_of_throats, 3), np.bool_)
        theta_h = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.float64)
        b = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.float64)
        beta = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.float64)
        theta_a = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.float64)
        bo = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.float64)
        corner_oil_layer = np.empty((number_of_throats, 3, MAX_NUMBER_OF_CORNERS), np.bool_)

        for throat_id in throat_id_list:
            connections = throat_conns[throat_id]
            pore_0_id = connections[0]
            pore_1_id = connections[1]

            L[throat_id] = [pore_l[pore_0_id], throat_l[throat_id], pore_l[pore_1_id]]
            A[throat_id] = [pore_a[pore_0_id], throat_a[throat_id], pore_a[pore_1_id]]
            G[throat_id] = [pore_g[pore_0_id], throat_g[throat_id], pore_g[pore_1_id]]
            shape[throat_id] = [pore_shape[pore_0_id], throat_shape[throat_id], pore_shape[pore_1_id]]
            n_corners[throat_id] = [pore_ncorners[pore_0_id], throat_ncorners[throat_id], pore_ncorners[pore_1_id]]
            So[throat_id] = [pore_so[pore_0_id], throat_so[throat_id], pore_so[pore_1_id]]
            oil_filled[throat_id] = [pore_oilfilled[pore_0_id], throat_oilfilled[throat_id], pore_oilfilled[pore_1_id]]

            new_theta_a = [
                np.full(MAX_NUMBER_OF_CORNERS, pore_thetaa[pore_0_id]),
                np.full(MAX_NUMBER_OF_CORNERS, throat_thetaa[throat_id]),
                np.full(MAX_NUMBER_OF_CORNERS, pore_thetaa[pore_1_id]),
            ]
            new_theta_h = [pore_thetah[pore_0_id], throat_thetah[throat_id], pore_thetah[pore_1_id]]
            new_b = [pore_b[pore_0_id], throat_b[throat_id], pore_b[pore_1_id]]
            new_beta = [pore_beta[pore_0_id], throat_beta[throat_id], pore_beta[pore_1_id]]
            new_bo = [pore_bo[pore_0_id], throat_bo[throat_id], pore_bo[pore_1_id]]
            new_corner_oil_layer = [pore_oillayer[pore_0_id], throat_oillayer[throat_id], pore_oillayer[pore_1_id]]

            for j in range(3):
                theta_a[throat_id][j] = new_theta_a[j]
                theta_h[throat_id][j] = new_theta_h[j]
                b[throat_id][j] = new_b[j]
                beta[throat_id][j] = new_beta[j]
                bo[throat_id][j] = new_bo[j]
                corner_oil_layer[throat_id][j] = new_corner_oil_layer[j]

        return calculate_multiphase_network_oil_conductance(
            L,
            A,
            G,
            shape,
            So,
            b,
            bo,
            beta,
            theta_h,
            theta_a,
            corner_oil_layer,
            n_corners,
            oil_filled,
            number_of_throats,
            throat_id_list,
        )

    @staticmethod
    @nb.njit
    def multiphase_water_conductance(
        number_of_throats,
        throat_id_list,
        throat_conns,
        throat_l,
        throat_a,
        throat_g,
        throat_shape,
        throat_thetah,
        throat_b,
        throat_beta,
        throat_ncorners,
        throat_sw,
        throat_waterfilled,
        pore_l,
        pore_a,
        pore_g,
        pore_shape,
        pore_thetah,
        pore_b,
        pore_beta,
        pore_ncorners,
        pore_sw,
        pore_waterfilled,
    ):
        L = np.empty((number_of_throats, 3), np.float64)
        A = np.empty((number_of_throats, 3), np.float64)
        G = np.empty((number_of_throats, 3), np.float64)
        shape = np.empty((number_of_throats, 3), np.float64)
        theta_h = np.empty((number_of_throats, 3, 4), np.float64)
        b = np.empty((number_of_throats, 3, 4), np.float64)
        beta = np.empty((number_of_throats, 3, 4), np.float64)
        n_corners = np.empty((number_of_throats, 3), np.int32)
        Sw = np.empty((number_of_throats, 3), np.float64)
        water_filled = np.empty((number_of_throats, 3), np.bool_)

        for throat_id in throat_id_list:
            connections = throat_conns[throat_id]
            pore_0_id = connections[0]
            pore_1_id = connections[1]

            L[throat_id] = [pore_l[pore_0_id], throat_l[throat_id], pore_l[pore_1_id]]
            A[throat_id] = [pore_a[pore_0_id], throat_a[throat_id], pore_a[pore_1_id]]
            G[throat_id] = [pore_g[pore_0_id], throat_g[throat_id], pore_g[pore_1_id]]
            shape[throat_id] = [pore_shape[pore_0_id], throat_shape[throat_id], pore_shape[pore_1_id]]
            Sw[throat_id] = [pore_sw[pore_0_id], throat_sw[throat_id], pore_sw[pore_1_id]]
            n_corners[throat_id] = [pore_ncorners[pore_0_id], throat_ncorners[throat_id], pore_ncorners[pore_1_id]]
            water_filled[throat_id] = [
                pore_waterfilled[pore_0_id],
                throat_waterfilled[throat_id],
                pore_waterfilled[pore_1_id],
            ]

            new_theta_h = [pore_thetah[pore_0_id], throat_thetah[throat_id], pore_thetah[pore_1_id]]
            new_b = [pore_b[pore_0_id], throat_b[throat_id], pore_b[pore_1_id]]
            new_beta = [pore_beta[pore_0_id], throat_beta[throat_id], pore_beta[pore_1_id]]

            for j in range(3):
                theta_h[throat_id][j] = new_theta_h[j]
                b[throat_id][j] = new_b[j]
                beta[throat_id][j] = new_beta[j]

        return calculate_multiphase_network_water_conductance(
            L, A, G, shape, Sw, theta_h, b, beta, n_corners, water_filled, number_of_throats, throat_id_list
        )

    def __get_singlephase_parameters(self, throat_j):
        i = 0  # Pore 1
        j = 2  # Throat connecting pore 0 and 1
        k = 1  # Pore 2

        pore_list = self.network.get_neighbor_pores(throat_j.get_id())
        pore_i = pore_list[0]
        pore_k = pore_list[1]

        A = np.empty(3, dtype=np.float64)
        A[i] = pore_i.A
        A[j] = throat_j.A
        A[k] = pore_k.A
        G = np.empty(3, dtype=np.float64)
        G[i] = pore_i.G
        G[j] = throat_j.G
        G[k] = pore_k.G
        L = np.empty(3, dtype=np.float64)
        L[i] = pore_i.r
        L[j] = throat_j.L
        L[k] = pore_k.r
        shape = np.empty(3, dtype=np.float64)
        shape[i] = pore_i.shape
        shape[j] = throat_j.shape
        shape[k] = pore_k.shape

        return L, A, G, shape


class FlowRateCalculator:
    def __init__(self, network):
        self.network = network

    def calculate_singlephase_flow_rate(self, phase):
        if phase == WATER:
            phase_name = "water"
        else:
            phase_name = "oil"

        self.network.pn_core["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_singlephase_conductance"
        ]

        return self.__solve_flow_rate(phase)

    def calculate_total_flow_rate(self, phase, update_hydraulic_conductance=True):
        if update_hydraulic_conductance:
            self.update_hydraulic_conductance(phase)
        return self.__solve_flow_rate(phase)

    def update_hydraulic_conductance(self, phase):
        if phase == WATER:
            phase_name = "water"
        else:
            phase_name = "oil"

        # multiphase_throats = self.network.pn[f"throat.{phase_name}_saturation"] != 1
        multiphase_throats = self.network.pn_core.throat.all
        multiphase_throat_id_list = np.where(multiphase_throats)[0]

        conductance_calculator = ConductanceCalculator(self.network)
        multiphase_conductance = conductance_calculator.calculate_multiphase_conductance(
            phase, multiphase_throat_id_list
        )

        self.network.pn_core[f"throat.{phase_name}_multiphase_conductance"] = multiphase_conductance
        self.network.pn_core["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_multiphase_conductance"
        ]

    def __solve_flow_rate(self, phase):
        if phase == WATER:
            phase = openpnm.phase.Water(network=self.network.pn)
        else:
            phase = openpnm.phase.Air(network=self.network.pn)
        # water.add_model_collection(openpnm.models.collections.physics.standard)
        stokes_flow = openpnm.algorithms.StokesFlow(network=self.network.pn, phase=phase)

        stokes_flow.set_value_BC(
            pores=self.network.get_inlets_ids(), values=101325, mode="overwrite"
        )  # pressure in pa: 101325 pa = 1 atm
        stokes_flow.set_value_BC(pores=self.network.get_outlets_ids(), values=0, mode="overwrite")
        stokes_flow.run()
        inlet_flow = stokes_flow.rate(pores=self.network.get_inlets_ids())
        outlet_flow = stokes_flow.rate(pores=self.network.get_outlets_ids())
        flow_rate = (inlet_flow[0] - outlet_flow[0]) / 2
        return flow_rate


class KrelCalculator:
    def __init__(self, network, calculate_full_conductance=False):
        self.network = network
        self.calculate_full_conductance = calculate_full_conductance

        flow_rate_calculator = FlowRateCalculator(self.network)

        self.singlephase_flow_rate = [0.0] * 2
        self.singlephase_flow_rate[WATER] = flow_rate_calculator.calculate_singlephase_flow_rate(WATER)
        self.singlephase_flow_rate[OIL] = flow_rate_calculator.calculate_singlephase_flow_rate(OIL)
        self.absolute_permeability = self.__calculate_K()

    def calculate_kro(self):
        if self.calculate_full_conductance:
            update_hydraulic_conductance = False
            flow_rate_calculator = FlowRateCalculator(self.network)
            flow_rate_calculator.update_hydraulic_conductance(OIL)
        else:
            update_hydraulic_conductance = True

        network_clone = openpnm.network.Network()
        network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(network_clone["pore.oil_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(network_clone["throat.oil_saturation"] == 0)[0]
        try:
            valid_clusters_list = self.__trim(
                network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self.__calculate_network_flowrate(
            network_clone, update_hydraulic_conductance, valid_clusters_list, OIL
        )

        return flow_rate / self.singlephase_flow_rate[OIL]

    def calculate_krw(self):
        if self.calculate_full_conductance:
            update_hydraulic_conductance = False
            flow_rate_calculator = FlowRateCalculator(self.network)
            flow_rate_calculator.update_hydraulic_conductance(WATER)
        else:
            update_hydraulic_conductance = True

        network_clone = openpnm.network.Network()
        network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(network_clone["pore.water_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(network_clone["throat.water_saturation"] == 0)[0]
        try:
            valid_clusters_list = self.__trim(
                network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self.__calculate_network_flowrate(
            network_clone, update_hydraulic_conductance, valid_clusters_list, WATER
        )
        return flow_rate / self.singlephase_flow_rate[WATER]

    def get_absolute_permeability(self):
        return self.absolute_permeability

    def __calculate_network_flowrate(self, network_clone, update_hydraulic_conductance, valid_clusters_list, phase):
        if len(valid_clusters_list) > 0:
            flow_rate = 0.0
            for valid_cluster_id in valid_clusters_list:
                sub_network_clone = openpnm.network.Network()
                sub_network_clone.update(network_clone.copy())
                self.__trim_cluster(sub_network_clone, valid_cluster_id)
                flow_rate += self.__calculate_subnetwork_flowrate(
                    sub_network_clone, update_hydraulic_conductance, phase
                )
        else:
            flow_rate = self.__calculate_subnetwork_flowrate(network_clone, update_hydraulic_conductance, phase)
        return flow_rate

    def __calculate_subnetwork_flowrate(self, network_clone, update_hydraulic_conductance, phase):
        if len(network_clone.throats()) == 0 or len(network_clone.pores()) == 0:
            return 0
        new_network = Network(network_clone)
        if len(new_network.get_inlets_ids()) == 0 or len(new_network.get_outlets_ids()) == 0:
            return 0
        flow_rate_calculator = FlowRateCalculator(new_network)
        flow_rate = flow_rate_calculator.calculate_total_flow_rate(phase, update_hydraulic_conductance)
        return flow_rate

    def __calculate_K(self):
        A, L = self.__get_network_size(self.network.pn)
        qtsp = self.singlephase_flow_rate[WATER]
        return calculate_absolute_permeability(WATER, qtsp, A, L, 101325, 0)

    @staticmethod
    def __trim(pn, pores=[], throats=[]):
        if len(pores) == 0 and len(throats) == 0:
            return []
        if len(pores) == len(pn.pores()) or len(throats) == len(pn.throats()):
            raise ValueError()
        openpnm.topotools.trim(pn, pores=pores, throats=throats)
        try:
            pn.add_model(propname="pore.cluster_number", model=openpnm.models.network.cluster_number)
            pn.add_model(propname="pore.cluster_size", model=openpnm.models.network.cluster_size)
        except:
            return []

        clusters_with_inlets = pn["pore.cluster_number"][pn["pore.inlets"]]
        clusters_with_outlets = pn["pore.cluster_number"][pn["pore.outlets"]]
        valid_clusters = np.intersect1d(clusters_with_inlets, clusters_with_outlets)
        if len(valid_clusters) == 0:
            raise ValueError()
        return valid_clusters

    @staticmethod
    def __trim_cluster(pn, choosen_cluster):
        clustered_pores = pn[pn["pore.cluster_number"] != choosen_cluster]

        openpnm.topotools.trim(network=pn, pores=clustered_pores)

    @staticmethod
    def __get_network_size(pn):
        dimension_list = [0, 1, 2]
        element_max = [0.0] * len(dimension_list)
        element_min = [0.0] * len(dimension_list)

        for dimension in dimension_list:
            element_min[dimension] = pn["pore.coords"][:, dimension].min()
            element_max[dimension] = pn["pore.coords"][:, dimension].max()

        inlets = pn["pore.inlets"]
        outlets = pn["pore.outlets"]

        flow_dimension = None
        for dimension in dimension_list:
            inlet_min = pn["pore.coords"][inlets][:, dimension].min()
            inlet_max = pn["pore.coords"][inlets][:, dimension].max()
            if inlet_min == inlet_max:
                flow_dimension = dimension
                break

        dimension_list.remove(flow_dimension)
        cross_sectional_area = (
            element_max[dimension_list[0]]
            - element_min[dimension_list[0]] * element_max[dimension_list[1]]
            - element_min[dimension_list[1]]
        )
        length = element_max[flow_dimension] - element_min[flow_dimension]

        return cross_sectional_area, length


class SaturationCalculator:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase
        self.total_volume = self.network.pn_core.pore.volume.sum() + self.network.pn_core.throat.volume.sum()

    def update_saturations(self, pc):
        if self.phase == OIL:
            self.__update_drainage_saturations(pc)
        else:
            self.__update_imbibition_saturations()

    def update_element_saturations(self, type_name, id, pc):
        if self.phase == OIL:
            self.__update_drainage_element_saturations(type_name, id, pc)
        else:
            self.__update_imbibition_element_saturations(type_name, id)
        return (
            self.network.pn[f"{type_name}.water_saturation"][id]
            * self.network.pn[f"{type_name}.volume"][id]
            / self.total_volume
        )

    def __update_drainage_saturations(self, pc):
        (
            self.network.pn_core["pore.water_saturation"],
            self.network.pn_core["pore.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn_core.pore.shape,
            self.network.pn_core.pore.C_r,
            pc,
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
        )
        (
            self.network.pn_core["throat.water_saturation"],
            self.network.pn_core["throat.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn_core.throat.shape,
            self.network.pn_core.throat.C_r,
            pc,
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
        )

    def __update_imbibition_saturations(self):
        (
            self.network.pn_core["pore.water_saturation"],
            self.network.pn_core["pore.oil_saturation"],
        ) = calculate_imbibition_network_saturation(
            self.network.pn_core.pore.area,
            self.network.pn_core.pore.oil_filled,
            self.network.pn_core.pore.advancing_contact_angle,
            self.network.pn_core.pore.hinging_contact_angle,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.bo,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.corner_oil_layer,
        )
        (
            self.network.pn_core["throat.water_saturation"],
            self.network.pn_core["throat.oil_saturation"],
        ) = calculate_imbibition_network_saturation(
            self.network.pn_core.throat.cross_sectional_area,
            self.network.pn_core.throat.oil_filled,
            self.network.pn_core.throat.advancing_contact_angle,
            self.network.pn_core.throat.hinging_contact_angle,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.bo,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.corner_oil_layer,
        )

    def __update_drainage_element_saturations(self, type_name, id, pc):
        if type_name == "pore":
            area = self.network.pn_core.pore.area[id]
        else:
            area = self.network.pn_core.throat.cross_sectional_area[id]

        (
            self.network.pn[f"{type_name}.water_saturation"][id],
            self.network.pn[f"{type_name}.oil_saturation"][id],
        ) = calculate_drainage_element_saturation(
            self.network.pn[f"{type_name}.shape"][id],
            self.network.pn[f"{type_name}.C_r"][id],
            pc,
            area,
            self.network.pn[f"{type_name}.oil_filled"][id],
        )

    def __update_imbibition_element_saturations(self, type_name, id):
        if type_name == "pore":
            area = self.network.pn_core.pore.area[id]
        else:
            area = self.network.pn_core.throat.cross_sectional_area[id]

        (
            self.network.pn[f"{type_name}.water_saturation"][id],
            self.network.pn[f"{type_name}.oil_saturation"][id],
        ) = calculate_imbibition_element_saturation(
            area,
            self.network.pn[f"{type_name}.oil_filled"][id],
            self.network.pn[f"{type_name}.advancing_contact_angle"][id],
            self.network.pn[f"{type_name}.hinging_contact_angle"][id],
            self.network.pn[f"{type_name}.corner_half_angle"][id],
            self.network.pn[f"{type_name}.b"][id],
            self.network.pn[f"{type_name}.bo"][id],
            self.network.pn[f"{type_name}.n_corners"][id],
            self.network.pn[f"{type_name}.corner_oil_layer"][id],
        )

    def calculate_sw_total(self):
        return calculate_sw_total(
            np.concatenate(
                (
                    self.network.pn_core.throat.water_saturation,
                    self.network.pn_core.pore.water_saturation,
                )
            ),
            np.concatenate(
                (
                    self.network.pn_core.throat.volume,
                    self.network.pn_core.pore.volume,
                )
            ),
            np.concatenate(
                (
                    self.network.pn_core.throat.clay_volume,
                    self.network.pn_core.pore.clay_volume,
                )
            ),
            self.total_volume,
        )


spec = [
    ("throats_ids", nb.types.ListType(nb.types.int64)),
    ("pores_ids", nb.types.ListType(nb.types.int64)),
]


@jitclass(spec)
class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements
    """

    def __init__(self):
        self.throats_ids = nb.typed.List.empty_list(nb.types.int64)
        self.pores_ids = nb.typed.List.empty_list(nb.types.int64)

    def set_throats_ids(self, throats_ids):
        self.throats_ids = throats_ids

    def set_pores_ids(self, pores_ids):
        self.pores_ids = pores_ids

    def get_throats_ids(self):
        return np.asarray(self.throats_ids)

    def get_pores_ids(self):
        return np.asarray(self.pores_ids)

    def add_throat(self, throat_id):
        if throat_id not in self.throats_ids:
            self.throats_ids.append(throat_id)

    def add_pore(self, pore_id):
        if pore_id not in self.pores_ids:
            self.pores_ids.append(pore_id)

    def remove_throat(self, throat_id):
        if throat_id in self.throats_ids:
            self.throats_ids.remove(throat_id)

    def remove_pore(self, pore_id):
        if pore_id in self.pores_ids:
            self.pores_ids.remove(pore_id)

    def remove_throat_list(self, throat_id_list):
        for id in throat_id_list:
            self.remove_throat(id)

    def remove_pore_list(self, pore_id_list):
        for id in pore_id_list:
            self.remove_pore(id)


class CycleSimulation:
    def __init__(self, network, phase, calculate_full_conductance=False):
        self.network = network
        self.phase = phase

        self.pores_dict = self.network.get_pore_list()
        self.throats_dict = self.network.get_throat_list()

        self.krel_calculator = KrelCalculator(self.network, calculate_full_conductance)

        self.border_elements = BorderElements()
        self.border_elements.set_pores_ids(nb.typed.List(self.network.get_inlets_ids()))
        self.__find_initial_border_elements(self.border_elements, phase)

        self.trap_checker = TrapChecker(network, self.phase)

        self.saturation_calculator = SaturationCalculator(network, phase)

        self.current_pc = None

    def update_pc(self, pc, target_sw=None):
        invasion_pc = pc
        total_increased_sw = 0.0
        while True:
            if target_sw is not None:
                invasion_target_sw = target_sw - total_increased_sw
            else:
                invasion_target_sw = None
            invaded_pc = invasion_pc
            increased_sw, invasion_pc = self._update_invasions(invasion_pc, invasion_target_sw)
            self._update_network_geometry(invaded_pc)
            if target_sw is None:
                break
            elif increased_sw == 0.0:
                break
            else:
                total_increased_sw += increased_sw
                if total_increased_sw >= target_sw:
                    break
        return total_increased_sw, invaded_pc, invasion_pc

    def calculate_sw_total(self):
        return self.saturation_calculator.calculate_sw_total()

    def calculate_krel(self):
        return self.krel_calculator.calculate_krw(), self.krel_calculator.calculate_kro()

    def get_oil_filled_percentage(self):
        total_pores = self.network.get_number_of_pores()
        pores_filled = self.network.pn_core.pore.oil_filled.sum()
        return pores_filled, total_pores, int((pores_filled / total_pores) * 100)

    @abstractmethod
    def _check_invasion_condition(self, pc, element):
        pass

    def _update_element_filled_geometry(self, pc, element):
        self.__update_element_hinging_contact_angles(element.type_name, element.id, pc)
        return self.saturation_calculator.update_element_saturations(element.type_name, element.id, pc)

    def _update_network_geometry(self, pc):
        self.__update_hinging_contact_angles(pc)
        self.saturation_calculator.update_saturations(pc)

    def _update_element_entry_pc(self, element, pc):
        pass

    def _update_throat_entry_pc(self, throat_id, pc):
        pass

    def _update_pore_entry_pc(self, pore_id, pc):
        pass

    @abstractmethod
    def _get_element_to_fill(self, pc, border_elements):
        pass

    def _update_entry_pressures(self, pc):
        for throat_id in self.border_elements.get_throats_ids():
            self._update_throat_entry_pc(throat_id, pc)
        for pore_id in self.border_elements.get_pores_ids():
            self._update_pore_entry_pc(pore_id, pc)

    def _update_invasions(self, pc, target_sw=None):
        increased_sw = 0.0

        self._update_entry_pressures(pc)

        invasion_pc = pc
        while True:
            element_to_fill_list = self._get_element_to_fill(pc, self.border_elements)
            if len(element_to_fill_list) == 0:
                break
            else:
                element_to_fill = element_to_fill_list[0]

            invasion_pc = element_to_fill[2]

            if (target_sw is not None and increased_sw >= target_sw) or not self._can_be_filled(pc, invasion_pc):
                break

            if element_to_fill[0] == PORE:
                element = self.network.get_pore(element_to_fill[1])
            else:
                element = self.network.get_throat(element_to_fill[1])

            log.debug(
                f"Invasion {element.type_name:6} ID: {element.id:6} Pc: {element.pc:.5f} Radius: {element.r:.2e} G: {element.G:.4f} n_corners: {element.n_corners} area: {element.A}"
            )

            self.__fill_element(element, invasion_pc)
            increased_sw += self._update_element_filled_geometry(invasion_pc, element)
            self.trap_checker.update_trapped_pores()
            self.__remove_trapped_from_border()

        return increased_sw, invasion_pc

    def __remove_trapped_from_border(self):
        throat_id_list = self.border_elements.get_throats_ids()
        trapped_throat_list = np.where(self.network.pn_core.throat.trapped[throat_id_list] == True)[0]
        self.border_elements.remove_throat_list(throat_id_list[trapped_throat_list])

        pore_id_list = self.border_elements.get_pores_ids()
        trapped_pore_list = np.where(self.network.pn_core.pore.trapped[pore_id_list] == True)[0]
        self.border_elements.remove_pore_list(pore_id_list[trapped_pore_list])

    def __fill_element(self, element, pc):
        element.set_filled(self.phase, True)
        if element.is_pore():
            self.border_elements.remove_pore(element.get_id())
        else:
            self.border_elements.remove_throat(element.get_id())

        neighbor_list = self.network.get_neighbors(element)
        for neighbor in neighbor_list:
            if not neighbor.is_filled(self.phase):
                if neighbor.is_pore():
                    self.border_elements.add_pore(neighbor.get_id())
                else:
                    self.border_elements.add_throat(neighbor.get_id())
                self._update_element_entry_pc(neighbor, pc)

    def __update_hinging_contact_angles(self, pc):
        self.network.pn_core["pore.hinging_contact_angle"] = calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.pore.corner_half_angle,
            self.network.pn_core.pore.b,
            self.network.pn_core.pore.n_corners,
            self.network.pn_core.pore.receding_contact_angle,
            self.network.pn_core.pore.advancing_contact_angle,
        )
        self.network.pn_core["throat.hinging_contact_angle"] = calculate_network_hinging_contact_angle(
            pc,
            self.network.pn_core.throat.corner_half_angle,
            self.network.pn_core.throat.b,
            self.network.pn_core.throat.n_corners,
            self.network.pn_core.throat.receding_contact_angle,
            self.network.pn_core.throat.advancing_contact_angle,
        )

    def __update_element_hinging_contact_angles(self, type_name, id, pc):
        self.network.pn[f"{type_name}.hinging_contact_angle"][id] = calculate_hinging_contact_angle(
            pc,
            self.network.pn[f"{type_name}.corner_half_angle"][id],
            self.network.pn[f"{type_name}.b"][id],
            self.network.pn[f"{type_name}.n_corners"][id],
            self.network.pn[f"{type_name}.receding_contact_angle"][id],
            self.network.pn[f"{type_name}.advancing_contact_angle"][id],
        )

    def __find_initial_border_elements(self, border_elements, phase):
        def update_element_in_border(element):
            if element.is_filled(self.phase):
                if element.is_pore():
                    self.border_elements.remove_pore(element.get_id())
                else:
                    self.border_elements.remove_throat(element.get_id())
                neighbor_list = self.network.get_neighbors(element)
                for neighbor in neighbor_list:
                    if not neighbor.is_filled(self.phase):
                        if neighbor.is_pore():
                            self.border_elements.add_pore(neighbor.get_id())
                        else:
                            self.border_elements.add_throat(neighbor.get_id())
                return True
            return False

        border_changed = True
        while border_changed:
            border_changed = False
            for pore_id in border_elements.get_pores_ids():
                element = self.network.get_pore(pore_id)
                border_changed |= update_element_in_border(element)

            for throat_id in border_elements.get_throats_ids():
                element = self.network.get_throat(throat_id)
                border_changed |= update_element_in_border(element)


class TrapChecker:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase

    def update_trapped_pores(self):
        self.network.pn_core["pore.trapped"] = False
        self.network.pn_core["throat.trapped"] = False
        if self.phase == OIL:
            cluster_array = openpnm.topotools.find_clusters(
                self.network.pn, mask=self.network.pn_core.pore.water_saturation > 0
            )
        else:
            cluster_array = openpnm.topotools.find_clusters(
                self.network.pn, mask=self.network.pn_core.pore.oil_saturation > 0
            )
        outlets_pore_id_list = self.network.get_outlets_ids()
        clusters_with_outlets = np.unique(cluster_array.pore_labels[outlets_pore_id_list])
        for cluster_label in range(np.max(cluster_array.pore_labels) + 1):
            pore_cluster = np.where(cluster_array.pore_labels == cluster_label)[0]
            throat_cluster = np.where(cluster_array.throat_labels == cluster_label)[0]
            is_cluster_trapped = cluster_label not in clusters_with_outlets
            self.network.pn_core.pore.trapped[pore_cluster] = is_cluster_trapped
            self.network.pn_core.throat.trapped[throat_cluster] = is_cluster_trapped


spec = [
    ("enable_control", nb.types.float64),
    ("cycle", nb.types.int32),
    ("DRAINAGE_CYCLE", nb.types.int32),
    ("IMBIBITION_CYCLE", nb.types.int32),
    ("pc", nb.types.float64),
    ("pc_step_initial", nb.types.float64),
    ("pc_step", nb.types.float64),
    ("pc_step_min", nb.types.float64),
    ("pc_step_max", nb.types.float64),
    ("sw_step", nb.types.float64),
    ("pc_maximum", nb.types.float64),
    ("pc_minimum", nb.types.float64),
    ("sw_last", nb.types.float64),
    ("first_step", nb.types.boolean),
]


@jitclass(spec)
class PcIncrementer:
    def __init__(self, pc_step, pc_step_min, pc_step_max, sw_step, pc_maximum, pc_minimum, enable_control=True):
        self.DRAINAGE_CYCLE = 0
        self.IMBIBITION_CYCLE = 1

        self.enable_control = enable_control
        self.pc_step_initial = pc_step
        self.pc_step_min = pc_step_min
        self.pc_step_max = pc_step_max
        self.sw_step = sw_step
        self.pc_maximum = pc_maximum
        self.pc_minimum = pc_minimum

        self.cycle = self.DRAINAGE_CYCLE

        self.pc = 0
        self.pc_step = self.pc_step_initial

        self.first_step = True
        self.sw_last = 0

    def set_cycle(self, cycle):
        self.cycle = cycle

    def get_pc(self):
        return self.pc

    def get_pc_max(self):
        return self.pc_maximum

    def get_pc_min(self):
        return self.pc_minimum

    def set_pc(self, new_pc):
        self.pc = new_pc

    def set_pc_to_min(self):
        self.pc = self.pc_minimum

    def set_pc_to_max(self):
        self.pc = self.pc_maximum

    def increment(self, sw_current):
        self.__adjust_pc_step(sw_current)
        self.pc = min(self.pc + self.pc_step, self.pc_maximum)
        self.sw_last = sw_current
        self.first_step = False
        return self.pc

    def decrement(self, sw_current):
        self.__adjust_pc_step(sw_current)
        self.pc = max(self.pc - self.pc_step, self.pc_minimum)
        self.sw_last = sw_current
        self.first_step = False
        return self.pc

    def update_pc(self, sw_current=0.0):
        if self.cycle == self.DRAINAGE_CYCLE:
            return self.increment(sw_current)
        else:
            return self.decrement(sw_current)

    def cycle_should_end(self):
        if self.cycle == self.DRAINAGE_CYCLE:
            return self.pc >= self.get_pc_max()
        else:
            return self.pc <= self.get_pc_min()

    def __adjust_pc_step(self, sw_current):
        if self.enable_control and not self.first_step:
            sw_diff = abs(self.sw_last - sw_current)
            if sw_diff > 0:
                error = self.sw_step / sw_diff
                if error > 0:
                    self.pc_step *= error / 2
                else:
                    self.pc_step *= error
            else:
                self.pc_step = self.pc_step_initial
            self.pc_step = max(self.pc_step, self.pc_step_min)
            self.pc_step = min(self.pc_step, self.pc_step_max)
        else:
            self.pc_step = self.pc_step_initial
