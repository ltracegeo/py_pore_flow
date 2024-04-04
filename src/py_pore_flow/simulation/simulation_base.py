from abc import abstractmethod

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from py_pore_flow.equations import (
    calculate_absolute_permeability,
    calculate_C,
    calculate_drainage_network_saturation,
    calculate_hinging_contact_angle,
    calculate_imbibition_network_saturation,
    calculate_multiphase_network_oil_conductance,
    calculate_multiphase_network_water_conductance,
    calculate_singlephase_conductance,
    calculate_sw_total,
    WATER,
    OIL,
    MAX_NUMBER_OF_CORNERS,
)
from py_pore_flow.network import Network


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
        number_of_throats = len(self.network.pn[f"throat.all"])
        L = np.empty((number_of_throats, 3), np.float64)
        A = np.empty((number_of_throats, 3), np.float64)
        G = np.empty((number_of_throats, 3), np.float64)
        shape = np.empty((number_of_throats, 3), np.float64)
        theta_h = np.empty((number_of_throats, 3, 4), np.float64)
        b = np.empty((number_of_throats, 3, 4), np.float64)
        beta = np.empty((number_of_throats, 3, 4), np.float64)
        n_corners = np.empty((number_of_throats, 3), np.float64)

        if phase == WATER:
            Sw = np.empty((number_of_throats, 3), np.float64)
            water_filled = np.empty((number_of_throats, 3), np.float64)

            for j, throat in enumerate(self.network.get_throat_list()):
                i = throat.get_id()
                (
                    L[i],
                    A[i],
                    G[i],
                    shape[i],
                    Sw[i],
                    theta_h[i],
                    b[i],
                    beta[i],
                    n_corners[i],
                    water_filled[i],
                ) = self.__get_multiphase_water_parameters(throat)
            return calculate_multiphase_network_water_conductance(
                L, A, G, shape, Sw, theta_h, b, beta, n_corners, water_filled, number_of_throats, throat_id_list
            )
        else:
            So = np.empty((number_of_throats, 3), np.float64)
            oil_filled = np.empty((number_of_throats, 3), np.float64)
            theta_a = np.empty((number_of_throats, 3, 4), np.float64)
            bo = np.empty((number_of_throats, 3, 4), np.float64)

            for j, throat in enumerate(self.network.get_throat_list()):
                i = throat.get_id()
                (
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
                    oil_filled[i],
                ) = self.__get_multiphase_oil_parameters(throat)
            return calculate_multiphase_network_oil_conductance(
                L, A, G, shape, So, b, bo, beta, theta_h, theta_a, n_corners, oil_filled, number_of_throats
            )

    def __get_multiphase_water_parameters(self, throat_j):
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
        Sw = np.empty(3, dtype=np.float64)
        Sw[i] = pore_i.Sw
        Sw[j] = throat_j.Sw
        Sw[k] = pore_k.Sw
        theta_h = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        theta_h[i] = pore_i.theta_h
        theta_h[j] = throat_j.theta_h
        theta_h[k] = pore_k.theta_h
        b = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        b[i] = pore_i.b
        b[j] = throat_j.b
        b[k] = pore_k.b
        beta = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        beta[i] = pore_i.beta
        beta[j] = throat_j.beta
        beta[k] = pore_k.beta

        n_corners = np.empty(3, dtype=np.float64)
        n_corners[i] = pore_i.n_corners
        n_corners[j] = throat_j.n_corners
        n_corners[k] = pore_k.n_corners
        water_filled = np.empty(3, dtype=np.bool_)
        water_filled[i] = pore_i.water_filled
        water_filled[j] = throat_j.water_filled
        water_filled[k] = pore_k.water_filled

        return L, A, G, shape, Sw, theta_h, b, beta, n_corners, water_filled

    def __get_multiphase_oil_parameters(self, throat_j):
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
        So = np.empty(3, dtype=np.float64)
        So[i] = pore_i.So
        So[j] = throat_j.So
        So[k] = pore_k.So
        oil_filled = np.empty(3, dtype=np.bool_)
        oil_filled[i] = pore_i.oil_filled
        oil_filled[j] = throat_j.oil_filled
        oil_filled[k] = pore_k.oil_filled

        b = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        b[i] = pore_i.b
        b[j] = throat_j.b
        b[k] = pore_k.b
        bo = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        bo[i] = pore_i.bo
        bo[j] = throat_j.bo
        bo[k] = pore_k.bo
        beta = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        beta[i] = pore_i.beta
        beta[j] = throat_j.beta
        beta[k] = pore_k.beta
        theta_h = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        theta_h[i] = pore_i.theta_h
        theta_h[j] = throat_j.theta_h
        theta_h[k] = pore_k.theta_h
        theta_a = np.empty((3, MAX_NUMBER_OF_CORNERS), dtype=np.float64)
        theta_a[i] = np.full(MAX_NUMBER_OF_CORNERS, pore_i.advancing_contact_angle)
        theta_a[j] = np.full(MAX_NUMBER_OF_CORNERS, throat_j.advancing_contact_angle)
        theta_a[k] = np.full(MAX_NUMBER_OF_CORNERS, pore_k.advancing_contact_angle)
        n_corners = np.empty(3, dtype=np.float64)
        n_corners[i] = pore_i.n_corners
        n_corners[j] = throat_j.n_corners
        n_corners[k] = pore_k.n_corners

        return L, A, G, shape, So, b, bo, beta, theta_h, theta_a, n_corners, oil_filled

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

        self.network.pn["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_singlephase_conductance"
        ]

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

    def calculate_total_flow_rate(self, phase):
        if phase == WATER:
            phase_name = "water"
        else:
            phase_name = "oil"

        multiphase_throats = self.network.pn[f"throat.{phase_name}_saturation"] != 1
        multiphase_throat_id_list = np.where(multiphase_throats)[0]

        conductance_calculator = ConductanceCalculator(self.network)
        multiphase_conductance = conductance_calculator.calculate_multiphase_conductance(
            phase, multiphase_throat_id_list
        )

        self.network.pn["throat.hydraulic_conductance"] = np.where(
            multiphase_throats,
            multiphase_conductance,
            self.network.pn[f"throat.{phase_name}_singlephase_conductance"],
        )

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
    def __init__(self, network):
        self.network = network

        flow_rate_calculator = FlowRateCalculator(self.network)

        self.singlephase_flow_rate = [0.0] * 2
        self.singlephase_flow_rate[WATER] = flow_rate_calculator.calculate_singlephase_flow_rate(WATER)
        self.singlephase_flow_rate[OIL] = flow_rate_calculator.calculate_singlephase_flow_rate(OIL)
        self.absolute_permeability = self.__calculate_K()

    def calculate_kro(self):
        network_clone = openpnm.network.Network()
        network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(network_clone["pore.oil_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(network_clone["throat.oil_saturation"] == 0)[0]
        if len(pore_id_to_remove_list) > 0 or len(throat_id_to_remove_list) > 0:
            if not self.__trim(network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list):
                return 0
        if len(network_clone.throats()) == 0 or len(network_clone.pores()) == 0:
            return 0
        new_network = Network(network_clone)
        if len(new_network.get_inlets_ids()) == 0 or len(new_network.get_outlets_ids()) == 0:
            return 0
        flow_rate_calculator = FlowRateCalculator(new_network)
        flow_rate = flow_rate_calculator.calculate_total_flow_rate(OIL)
        return flow_rate / self.singlephase_flow_rate[OIL]

    def calculate_krw(self):
        network_clone = openpnm.network.Network()
        network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(network_clone["pore.water_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(network_clone["throat.water_saturation"] == 0)[0]
        if len(pore_id_to_remove_list) > 0 or len(throat_id_to_remove_list) > 0:
            if not self.__trim(network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list):
                return 0
        if len(network_clone.throats()) == 0 or len(network_clone.pores()) == 0:
            return 0
        new_network = Network(network_clone)
        if len(new_network.get_inlets_ids()) == 0 or len(new_network.get_outlets_ids()) == 0:
            return 0
        flow_rate_calculator = FlowRateCalculator(new_network)
        flow_rate = flow_rate_calculator.calculate_total_flow_rate(WATER)
        return flow_rate / self.singlephase_flow_rate[WATER]

    def get_absolute_permeability(self):
        return self.absolute_permeability

    def __calculate_K(self):
        A, L = self.__get_network_size(self.network.pn)
        qtsp = self.singlephase_flow_rate[WATER]
        return calculate_absolute_permeability(WATER, qtsp, A, L, 101325, 0)

    @staticmethod
    def __trim(pn, pores=[], throats=[]):
        if len(pores) == len(pn.pores()):
            return False
        if len(throats) == len(pn.throats()):
            return False
        openpnm.topotools.trim(pn, pores=pores, throats=throats)
        try:
            pn.add_model(propname="pore.cluster_number", model=openpnm.models.network.cluster_number)
            pn.add_model(propname="pore.cluster_size", model=openpnm.models.network.cluster_size)
        except:
            return True

        clusters_with_inlets = pn["pore.cluster_number"][pn["pore.inlets"]]
        clusters_with_outlets = pn["pore.cluster_number"][pn["pore.outlets"]]
        valid_clusters = np.intersect1d(clusters_with_inlets, clusters_with_outlets)
        if len(valid_clusters) == 0:
            return False
        if len(valid_clusters) == 1:
            choosen_cluster = valid_clusters[0]
        else:
            # Get bigger cluster
            cluster_sizes = []
            for cluster_number in valid_clusters:
                cluster_sizes.append(pn["pore.cluster_size"][pn["pore.cluster_number"] == cluster_number][0])
            choosen_cluster = valid_clusters[np.argmax(cluster_sizes)]
        clustered_pores = pn[pn["pore.cluster_number"] != choosen_cluster]

        openpnm.topotools.trim(network=pn, pores=clustered_pores)
        return True

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

    def update_saturations(self, pc):
        if self.phase == OIL:
            self.update_drainage_saturations(pc)
        else:
            self.update_imbibition_saturations(pc)

    def update_drainage_saturations(self, pc):
        (
            self.network.pn["pore.water_saturation"],
            self.network.pn["pore.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn["pore.shape"],
            self.network.pn["pore.radius"],
            self.network.pn["pore.C_r"],
            pc,
            self.network.pn["pore.area"],
            self.network.pn["pore.oil_filled"],
        )
        (
            self.network.pn["throat.water_saturation"],
            self.network.pn["throat.oil_saturation"],
        ) = calculate_drainage_network_saturation(
            self.network.pn["throat.shape"],
            self.network.pn["throat.radius"],
            self.network.pn["throat.C_r"],
            pc,
            self.network.pn["throat.cross_sectional_area"],
            self.network.pn["throat.oil_filled"],
        )

    def update_imbibition_saturations(self, pc):
        (
            self.network.pn["pore.water_saturation"],
            self.network.pn["pore.oil_saturation"],
        ) = calculate_imbibition_network_saturation(
            self.network.pn["pore.shape"],
            self.network.pn["pore.radius"],
            self.network.pn["pore.C_h"],
            pc,
            self.network.pn["pore.area"],
            self.network.pn["pore.oil_filled"],
            self.network.pn["pore.advancing_contact_angle"],
            self.network.pn["pore.hinging_contact_angle"],
            self.network.pn["pore.corner_half_angle"],
            self.network.pn["pore.b"],
            self.network.pn["pore.bo"],
            self.network.pn["pore.n_corners"],
            self.network.pn["pore.corner_oil_layer"],
        )
        (
            self.network.pn["throat.water_saturation"],
            self.network.pn["throat.oil_saturation"],
        ) = calculate_imbibition_network_saturation(
            self.network.pn["throat.shape"],
            self.network.pn["throat.radius"],
            self.network.pn["throat.C_h"],
            pc,
            self.network.pn["throat.cross_sectional_area"],
            self.network.pn["throat.oil_filled"],
            self.network.pn["throat.advancing_contact_angle"],
            self.network.pn["throat.hinging_contact_angle"],
            self.network.pn["throat.corner_half_angle"],
            self.network.pn["throat.b"],
            self.network.pn["throat.bo"],
            self.network.pn["throat.n_corners"],
            self.network.pn["throat.corner_oil_layer"],
        )

    def calculate_sw_total(self):
        return calculate_sw_total(
            np.concatenate(
                (
                    self.network.pn["throat.water_saturation"],
                    self.network.pn["pore.water_saturation"],
                )
            ),
            np.concatenate(
                (
                    self.network.pn["throat.volume"],
                    self.network.pn["pore.volume"],
                )
            ),
            np.concatenate(
                (
                    self.network.pn["throat.clay_volume"],
                    self.network.pn["pore.clay_volume"],
                )
            ),
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


class SimulationBase:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase

        self.pores_dict = self.network.get_pore_list()
        self.throats_dict = self.network.get_throat_list()

        self.krel_calculator = KrelCalculator(self.network)

        self.border_elements = BorderElements()
        self.border_elements.set_pores_ids(nb.typed.List(self.network.get_inlets_ids()))

        self.trap_checker = TrapChecker(network, self.phase)

        self.saturation_calculator = SaturationCalculator(network, phase)

    def update_pc(self, pc):
        self._update_invasions(pc)

        self.trap_checker.update_trapped_pores()

        self._update_b(pc)
        self.__update_hinging_contact_angles(pc)
        self.saturation_calculator.update_saturations(pc)

    def calculate_sw_total(self):
        return self.saturation_calculator.calculate_sw_total()

    def calculate_krel(self):
        return self.krel_calculator.calculate_krw(), self.krel_calculator.calculate_kro()

    def get_oil_filled_percentage(self):
        total_pores = self.network.get_number_of_pores()
        pores_filled = self.network.pn["pore.oil_filled"].sum()
        return pores_filled, total_pores, int((pores_filled / total_pores) * 100)

    @abstractmethod
    def _check_invasion_condition(self, pc, element):
        pass

    def _update_b(self, pc):
        pass

    def _on_element_invaded(self, pc, element):
        pass

    def _update_invasions(self, pc):
        current_border_throats_ids = self.border_elements.get_throats_ids()
        for throat_id in current_border_throats_ids:
            if not self.throats_dict[throat_id].is_filled(self.phase):
                self.__apply_oil_pressure_to_throat(self.throats_dict[throat_id], pc)

        current_border_pores_ids = self.border_elements.get_pores_ids()
        for pore_id in current_border_pores_ids:
            if not self.pores_dict[pore_id].is_filled(self.phase):
                self.__apply_oil_pressure_to_pore(self.pores_dict[pore_id], pc)

    def __apply_oil_pressure_to_throat(self, throat, pc):
        if self._check_invasion_condition(pc, throat):
            throat.set_filled(self.phase, True)
            self._on_element_invaded(pc, throat)
            self.border_elements.remove_throat(throat.get_id())

            neighbor_pore_list = self.network.get_neighbor_pores(throat.get_id())
            for neighbor_pore in neighbor_pore_list:
                if not neighbor_pore.is_filled(self.phase):
                    self.border_elements.add_pore(neighbor_pore.get_id())
                    self.__apply_oil_pressure_to_pore(neighbor_pore, pc)

    def __apply_oil_pressure_to_pore(self, pore, pc):
        if not pore.trapped and self._check_invasion_condition(pc, pore):
            pore.set_filled(self.phase, True)
            self._on_element_invaded(pc, pore)
            self.border_elements.remove_pore(pore.get_id())

            neighbor_throat_list = self.network.get_neighbor_throats(pore.get_id())
            for neighbor_throat in neighbor_throat_list:
                if not neighbor_throat.is_filled(self.phase):
                    self.border_elements.add_throat(neighbor_throat.get_id())
                    self.__apply_oil_pressure_to_throat(neighbor_throat, pc)

    def __update_hinging_contact_angles(self, pc):
        for pore in self.network.get_pore_list():
            hinging_contact_angle = calculate_hinging_contact_angle(pc, pore.r, pore.beta, pore.b, pore.n_corners)
            hinging_contact_angle = np.clip(
                hinging_contact_angle, pore.receding_contact_angle, pore.advancing_contact_angle
            )
            pore.theta_h = hinging_contact_angle
            if self.phase == OIL:
                pore.C_h = calculate_C(pore.theta_h, pore.beta, pore.n_corners)

        for throat in self.network.get_throat_list():
            hinging_contact_angle = calculate_hinging_contact_angle(
                pc, throat.r, throat.beta, throat.b, throat.n_corners
            )
            hinging_contact_angle = np.clip(
                hinging_contact_angle, throat.receding_contact_angle, throat.advancing_contact_angle
            )
            throat.theta_h = hinging_contact_angle
            if self.phase == OIL:
                throat.C_h = calculate_C(throat.theta_h, throat.beta, throat.n_corners)


class TrapChecker:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase

    def update_trapped_pores(self):
        self.network.pn["pore.trapped"] = False
        outlets_pore_id_list = self.network.get_outlets_ids()
        if self.phase == OIL:
            cluster_array = openpnm.topotools.find_clusters(
                self.network.pn, mask=self.network.pn["pore.water_saturation"] > 0
            )
        else:
            cluster_array = openpnm.topotools.find_clusters(
                self.network.pn, mask=self.network.pn["pore.oil_saturation"] > 0
            )
        for cluster_label in range(np.max(cluster_array.pore_labels) + 1):
            cluster = np.where(cluster_array.pore_labels == cluster_label)[0]
            self.network.pn["pore.trapped"][cluster] = not np.isin(outlets_pore_id_list, cluster).any()


spec = [
    ("enable_control", nb.types.float64),
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
        self.enable_control = enable_control
        self.pc_step_initial = pc_step
        self.pc_step_min = pc_step_min
        self.pc_step_max = pc_step_max
        self.sw_step = sw_step
        self.pc_maximum = pc_maximum
        self.pc_minimum = pc_minimum

        self.pc = 0
        self.pc_step = self.pc_step_initial

        self.first_step = True
        self.sw_last = 0

    def get_pc(self):
        return self.pc

    def get_pc_max(self):
        return self.pc_maximum

    def get_pc_min(self):
        return self.pc_minimum

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
