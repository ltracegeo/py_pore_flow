import numpy as np
import openpnm
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
)
from py_pore_flow.equations.conductance import (
    calculate_absolute_permeability,
)
from py_pore_flow.network.network import Network
from py_pore_flow.simulation.flow_rate_calculator import FlowRateCalculator


class KrelCalculator:
    def __init__(self, network, calculate_full_conductance=False):
        self.network = network
        self.calculate_full_conductance = calculate_full_conductance
        self.network_clone = openpnm.network.Network()
        self.sub_network_clone = openpnm.network.Network()

        flow_rate_calculator = FlowRateCalculator(self.network)

        self.singlephase_flow_rate = [0.0] * 2
        self.singlephase_flow_rate[WATER] = flow_rate_calculator.calculate_singlephase_flow_rate(WATER)
        self.singlephase_flow_rate[OIL] = flow_rate_calculator.calculate_singlephase_flow_rate(OIL)
        self.absolute_permeability = self._calculate_K()

    def calculate_kro(self):
        if self.calculate_full_conductance:
            update_hydraulic_conductance = False
            flow_rate_calculator = FlowRateCalculator(self.network)
            flow_rate_calculator.update_hydraulic_conductance(OIL)
        else:
            update_hydraulic_conductance = True

        self.network_clone.clear()
        self.network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(self.network_clone["pore.oil_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(self.network_clone["throat.oil_saturation"] == 0)[0]
        try:
            valid_clusters_list = self._trim(
                self.network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self._calculate_network_flowrate(
            self.network_clone, update_hydraulic_conductance, valid_clusters_list, OIL
        )

        return flow_rate / self.singlephase_flow_rate[OIL]

    def calculate_krw(self):
        if self.calculate_full_conductance:
            update_hydraulic_conductance = False
            flow_rate_calculator = FlowRateCalculator(self.network)
            flow_rate_calculator.update_hydraulic_conductance(WATER)
        else:
            update_hydraulic_conductance = True

        self.network_clone.clear()
        self.network_clone.update(self.network.pn.copy())
        pore_id_to_remove_list = np.where(self.network_clone["pore.water_saturation"] == 0)[0]
        throat_id_to_remove_list = np.where(self.network_clone["throat.water_saturation"] == 0)[0]
        try:
            valid_clusters_list = self._trim(
                self.network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self._calculate_network_flowrate(
            self.network_clone, update_hydraulic_conductance, valid_clusters_list, WATER
        )
        return flow_rate / self.singlephase_flow_rate[WATER]

    def get_absolute_permeability(self):
        return self.absolute_permeability

    def _calculate_network_flowrate(self, network_clone, update_hydraulic_conductance, valid_clusters_list, phase):
        if len(valid_clusters_list) > 0:
            flow_rate = 0.0
            for valid_cluster_id in valid_clusters_list:
                self.sub_network_clone.clear()
                self.sub_network_clone.update(network_clone.copy())
                self._trim_cluster(self.sub_network_clone, valid_cluster_id)
                flow_rate += self._calculate_subnetwork_flowrate(
                    self.sub_network_clone, update_hydraulic_conductance, phase
                )
        else:
            flow_rate = self._calculate_subnetwork_flowrate(network_clone, update_hydraulic_conductance, phase)
        return flow_rate

    def _calculate_subnetwork_flowrate(self, network_clone, update_hydraulic_conductance, phase):
        if len(network_clone.throats()) == 0 or len(network_clone.pores()) == 0:
            return 0
        new_network = Network(network_clone, log_conductivity_file=self.network.log_conductivity_file)
        if len(new_network.get_inlets_ids()) == 0 or len(new_network.get_outlets_ids()) == 0:
            return 0
        flow_rate_calculator = FlowRateCalculator(new_network)
        flow_rate = flow_rate_calculator.calculate_total_flow_rate(phase, update_hydraulic_conductance)
        return flow_rate

    def _calculate_K(self):
        A, L = self._get_network_size(self.network.pn)
        qtsp = self.singlephase_flow_rate[WATER]
        return calculate_absolute_permeability(WATER, qtsp, A, L, 101325, 0)

    @staticmethod
    def _trim(pn, pores=[], throats=[]):
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
    def _trim_cluster(pn, choosen_cluster):
        clustered_pores = pn[pn["pore.cluster_number"] != choosen_cluster]

        openpnm.topotools.trim(network=pn, pores=clustered_pores)

    @staticmethod
    def _get_network_size(pn):
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
