from abc import abstractmethod

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from equations import (
    calculate_absolute_permeability,
    calculate_drainage_displacement,
    calculate_hinging_contact_angle,
    calculate_multiphase_oil_conductance,
    calculate_multiphase_water_conductance,
    calculate_singlephase_conductance,
    calculate_sw_total,
    WATER,
    OIL,
)
from network import Network


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
        multiphase_conductance = [0.0] * len(self.network.throat_list)
        for throat_id in throat_id_list:
            if phase == WATER:
                L, A, G, shape, Sw, theta_h, b, beta = self.__get_multiphase_water_parameters(
                    self.network.throat_list[throat_id]
                )
                multiphase_conductance[throat_id] = calculate_multiphase_water_conductance(
                    L, A, G, shape, Sw, theta_h, b, beta
                )
            else:
                L, A, G, shape, So = self.__get_multiphase_oil_parameters(self.network.throat_list[throat_id])
                multiphase_conductance[throat_id] = calculate_multiphase_oil_conductance(L, A, G, shape, So)
        return multiphase_conductance

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
        theta_h = [None] * 3
        theta_h[i] = pore_i.theta_h
        theta_h[j] = throat_j.theta_h
        theta_h[k] = pore_k.theta_h
        b = [None] * 3
        b[i] = pore_i.b
        b[j] = throat_j.b
        b[k] = pore_k.b
        beta = [None] * 3
        beta[i] = pore_i.beta
        beta[j] = throat_j.beta
        beta[k] = pore_k.beta

        return L, A, G, shape, Sw, theta_h, b, beta

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

        return L, A, G, shape, So

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

        clustered_pores = pn["pore.cluster_size"] < pn["pore.cluster_size"].max()
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
    def __init__(self, pores_dict, throats_dict):
        self.pores_dict = pores_dict
        self.throats_dict = throats_dict

    def calculate_sw_total(self, pc):
        length = len(self.pores_dict) + len(self.throats_dict)

        sw_list = np.empty(length, dtype=np.float64)
        v_list = np.empty(length, dtype=np.float64)
        v_clay_list = np.empty(length, dtype=np.float64)

        i = 0

        for pore in self.pores_dict:
            self.__update_saturations(pore, pc)
            sw_list[i] = pore.Sw
            v_list[i] = pore.V
            v_clay_list[i] = pore.Vclay
            i += 1

        for throat in self.throats_dict:
            self.__update_saturations(throat, pc)
            sw_list[i] = throat.Sw
            v_list[i] = throat.V
            v_clay_list[i] = throat.Vclay
            i += 1

        return calculate_sw_total(sw_list, v_list, v_clay_list)

    def __update_saturations(self, element, current_pc):
        if element.is_oil_filled():
            Sw, So = calculate_drainage_displacement(element.shape, element.C, current_pc, element.A)
        else:
            Sw, So = 1, 0

        element.Sw = Sw
        element.So = So


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

        self.trap_checker = TrapChecker(network, self.pores_dict, self.phase)

        self.saturation_calculator = SaturationCalculator(self.pores_dict, self.throats_dict)

    def update_pc(self, pc):
        self._update_invasions(pc)

        self.trap_checker.update_trapped_pores()

        self.__update_hinging_contact_angles(pc)

    def calculate_sw_total(self, pc):
        return self.saturation_calculator.calculate_sw_total(pc)

    def calculate_krel(self):
        return self.krel_calculator.calculate_krw(), self.krel_calculator.calculate_kro()

    def get_oil_filled_percentage(self):
        total_pores = 0
        pores_filled = 0
        for pore in self.pores_dict:
            total_pores += 1
            if pore.is_oil_filled():
                pores_filled += 1

        return pores_filled, total_pores, int((pores_filled / total_pores) * 100)

    @abstractmethod
    def _check_invasion_condition(self, pc, element):
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
            self.border_elements.remove_throat(throat.get_id())

            neighbor_pore_list = self.network.get_neighbor_pores(throat.get_id())
            for neighbor_pore in neighbor_pore_list:
                if not neighbor_pore.is_filled(self.phase):
                    self.border_elements.add_pore(neighbor_pore.get_id())
                    self.__apply_oil_pressure_to_pore(neighbor_pore, pc)

    def __apply_oil_pressure_to_pore(self, pore, pc):
        if not pore.is_trapped() and self._check_invasion_condition(pc, pore):
            pore.set_filled(self.phase, True)
            self.border_elements.remove_pore(pore.get_id())

            neighbor_throat_list = self.network.get_neighbor_throats(pore.get_id())
            for neighbor_throat in neighbor_throat_list:
                if not neighbor_throat.is_filled(self.phase):
                    self.border_elements.add_throat(neighbor_throat.get_id())
                    self.__apply_oil_pressure_to_throat(neighbor_throat, pc)

    def __update_hinging_contact_angles(self, pc):
        for pore in self.network.get_pore_list():
            hinging_contact_angle = calculate_hinging_contact_angle(pc, pore.beta, pore.b)
            hinging_contact_angle = np.clip(
                hinging_contact_angle, pore.receding_contact_angle, pore.advancing_contact_angle
            )
            pore.theta_h = hinging_contact_angle

        for throat in self.network.get_throat_list():
            hinging_contact_angle = calculate_hinging_contact_angle(pc, throat.beta, throat.b)
            hinging_contact_angle = np.clip(
                hinging_contact_angle, throat.receding_contact_angle, throat.advancing_contact_angle
            )
            throat.theta_h = hinging_contact_angle


class TrapChecker:
    def __init__(self, network, pores_dict, phase):
        self.network = network
        self.pores_dict = pores_dict

        self.cluster_finder = ClusterFinder(network, pores_dict, phase)

    def update_trapped_pores(self):
        for pore_id, is_trapped in enumerate(self.get_pore_trap_status()):
            if is_trapped:
                self.pores_dict[pore_id].set_trapped()

    def get_pore_trap_status(self):
        pore_trap_status_list = [True] * self.network.get_number_of_pores()
        outlets_pore_id_list = self.network.get_outlets_ids()
        checked_outlets = []
        for outlet_pore_id in outlets_pore_id_list:
            if outlet_pore_id in checked_outlets:
                continue
            pores_in_cluster, outlets_in_cluster = self.cluster_finder.get_cluster_info(outlet_pore_id)
            for pore_id in pores_in_cluster:
                pore_trap_status_list[pore_id] = False
            checked_outlets.extend(outlets_in_cluster)

        return pore_trap_status_list


class ClusterFinder:
    def __init__(self, network, pores_dict, phase):
        self.network = network
        self.pores_dict = pores_dict
        self.phase = phase

        self.outlets_pore_id_list = self.network.get_outlets_ids()

        self.pores_in_cluster = []
        self.outlets_in_cluster = []

    def get_cluster_info(self, base_pore_id):
        self.pores_in_cluster = []
        self.outlets_in_cluster = []

        self.__check_cluster(base_pore_id)

        return self.pores_in_cluster, self.outlets_in_cluster

    def __check_cluster(self, pore_id):
        pore = self.pores_dict[pore_id]
        if not self.__check_condition(pore) and pore_id not in self.pores_in_cluster:
            if pore_id in self.outlets_pore_id_list:
                self.outlets_in_cluster.append(pore_id)
            self.pores_in_cluster.append(pore_id)
            neighbor_pore_id_list = self.get_neighbor_pores(pore_id)
            for neighbor_pore_id in neighbor_pore_id_list:
                self.__check_cluster(neighbor_pore_id)

    def __check_condition(self, pore):
        if self.phase == OIL:
            # return pore.is_oil_filled()
            return pore.Sw == 0
        else:
            # return pore.is_water_filled()
            return pore.So == 0

    def get_neighbor_pores(self, pore_id):
        neighbor_pore_id_list = []
        for throat in self.network.get_neighbor_throats(pore_id):
            pore_list = self.network.get_neighbor_pores(throat.get_id())
            try:
                if pore_list[0].get_id() == pore_id:
                    new_pore_id = pore_list[1].get_id()
                else:
                    new_pore_id = pore_list[0].get_id()
                if new_pore_id not in neighbor_pore_id_list:
                    neighbor_pore_id_list.append(new_pore_id)
            except IndexError:
                pass
        return neighbor_pore_id_list


class PcIncrementer:
    def __init__(self, parameter_manager, enable_control=True):
        self.enable_control = enable_control
        self.pc = parameter_manager.pc_minimum
        self.pc_step_initial = parameter_manager.pc_step
        self.pc_step = self.pc_step_initial
        self.pc_step_min = parameter_manager.pc_step_min
        self.pc_step_max = parameter_manager.pc_step_max
        self.sw_step = parameter_manager.sw_step
        self.pc_maximum = parameter_manager.pc_maximum
        self.pc_minimum = parameter_manager.pc_minimum

        self.sw_last = None

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
        self.pc += self.pc_step
        self.sw_last = sw_current
        return self.pc

    def decrement(self, sw_current):
        self.__adjust_pc_step(sw_current)
        self.pc -= self.pc_step
        self.sw_last = sw_current
        return self.pc

    def __adjust_pc_step(self, sw_current):
        if self.enable_control and self.sw_last is not None:
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
