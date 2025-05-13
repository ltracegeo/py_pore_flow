from abc import abstractmethod

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm
from pyflowsolver import NetworkManager
from pypardiso import spsolve
from scipy.sparse import csr_matrix

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    MAX_NUMBER_OF_CORNERS,
    calculate_hinging_contact_angle,
    calculate_sw_total,
)
from py_pore_flow.equations.conductance import (
    calculate_absolute_permeability,
    calculate_network_conductance,
    calculate_network_multiphase_conductivity,
    calculate_network_singlephase_conductivity,
)
from py_pore_flow.network import Network, PORE, THROAT
from py_pore_flow.simulation.clusters import PhaseConnectionChecker


# Flags
USE_OPENPNM_SOLVER = False
USE_OPENPNM_TRAPCHECKER = False

ENABLE_TRAPPING_FROM_INLET = True

TRAP_TOLERANCE = 0.0


class ConductanceCalculator:
    def __init__(self, network):
        self.network = network

    def calculate_singlephase_conductance(self, phase):
        self.__calculate_singlephase_conductivity(phase)
        self.__calculate_singlephase_conductance(phase)

    def calculate_multiphase_conductance(self, phase):
        self.__calculate_multiphase_conductivity(phase)
        self.__calculate_multiphase_conductance(phase)

    def __calculate_singlephase_conductance(self, phase):
        if phase == WATER:
            calculate_network_conductance(
                self.network.pn_core.throat.water_singlephase_conductance,
                self.network.pn_core.throat.water_singlephase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.water_singlephase_conductivity,
                self.network.pn_core.pore.radius,
            )
        else:
            calculate_network_conductance(
                self.network.pn_core.throat.oil_singlephase_conductance,
                self.network.pn_core.throat.oil_singlephase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.oil_singlephase_conductivity,
                self.network.pn_core.pore.radius,
            )

    def __calculate_multiphase_conductance(self, phase):
        if phase == WATER:
            calculate_network_conductance(
                self.network.pn_core.throat.water_multiphase_conductance,
                self.network.pn_core.throat.water_multiphase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.water_multiphase_conductivity,
                self.network.pn_core.pore.radius,
            )
        else:
            calculate_network_conductance(
                self.network.pn_core.throat.oil_multiphase_conductance,
                self.network.pn_core.throat.oil_multiphase_conductivity,
                self.network.pn_core.throat.length,
                self.network.pn_core.throat.conns,
                self.network.pn_core.pore.oil_multiphase_conductivity,
                self.network.pn_core.pore.radius,
            )

    def __calculate_singlephase_conductivity(self, phase):
        for element_type in [PORE, THROAT]:
            if element_type == PORE:
                pn_element = self.network.pn_core.pore
                area = self.network.pn_core.pore.area
            else:  # element_type == THROAT:
                pn_element = self.network.pn_core.throat
                area = self.network.pn_core.throat.cross_sectional_area

            if phase == WATER:
                singlephase_conductivity = pn_element.water_singlephase_conductivity
            else:  # phase == OIL:
                singlephase_conductivity = pn_element.oil_singlephase_conductivity

            calculate_network_singlephase_conductivity(
                singlephase_conductivity,
                area,
                pn_element.shape_factor,
                pn_element.shape,
                phase,
            )
            singlephase_conductivity *= pn_element.N

    def __calculate_multiphase_conductivity(self, phase):
        for element_type in [PORE, THROAT]:
            if element_type == PORE:
                pn_element = self.network.pn_core.pore
                area = self.network.pn_core.pore.area
            else:  # element_type == THROAT:
                pn_element = self.network.pn_core.throat
                area = self.network.pn_core.throat.cross_sectional_area

            if phase == WATER:
                multiphase_conductivity = pn_element.water_multiphase_conductivity
            else:  # phase == OIL:
                multiphase_conductivity = pn_element.oil_multiphase_conductivity

            calculate_network_multiphase_conductivity(
                multiphase_conductivity,
                area,
                pn_element.shape_factor,
                pn_element.shape,
                pn_element.water_saturation,
                pn_element.oil_saturation,
                pn_element.b,
                pn_element.bo,
                pn_element.corner_half_angle,
                pn_element.hinging_contact_angle,
                np.tile(pn_element.advancing_contact_angle[:, np.newaxis], MAX_NUMBER_OF_CORNERS),
                pn_element.corner_oil_layer,
                pn_element.n_corners,
                pn_element.water_filled,
                pn_element.oil_filled,
                phase,
            )
            multiphase_conductivity *= pn_element.N


class FlowRateCalculator:
    def __init__(self, network):
        self.network = network
        self.solver = Solver(self.network.pn)

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

        conductance_calculator = ConductanceCalculator(self.network)
        conductance_calculator.calculate_multiphase_conductance(phase)

        self.network.pn_core["throat.hydraulic_conductance"] = self.network.pn[
            f"throat.{phase_name}_multiphase_conductance"
        ]

    def __solve_flow_rate(self, phase):
        if USE_OPENPNM_SOLVER:
            return self.__openpnm_solver(phase)
        else:
            return self.__pyflowsolver()

    def __openpnm_solver(self, phase):
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

    def __pyflowsolver(self):
        sub_network = self.solver.get_sub_network()
        if sub_network is False:
            return 0

        conn = sub_network["throat.conns"].astype(np.int32)
        cond = sub_network["throat.hydraulic_conductance"].astype(np.float64)
        inlets = sub_network["pore.inlets"]
        outlets = sub_network["pore.outlets"]

        factor = 10 ** (
            (np.log10(cond.min()) + np.log10(cond.max())) / 2
        )  # Prevent pypardiso from breaking with values too close to 0
        cond = cond / factor

        network_manager = NetworkManager(
            conn=conn,
            cond=cond,
            inlets=inlets,
            outlets=outlets,
        )
        network_manager.generate_sparse_system()
        a_sparse_array, b_array = network_manager.get_sparse_system()

        if len(a_sparse_array["val"]) == 0:
            return 1

        A_csr = csr_matrix(
            (
                a_sparse_array["val"],
                a_sparse_array["col_idx"],
                np.append(a_sparse_array["row_ptr"], a_sparse_array["val"].size),
            )
        )
        x = spsolve(A_csr, b_array)

        pressure = network_manager.get_pressure_list(x)
        flow_rate = network_manager.get_flow_rate(pressure)

        return flow_rate * factor


class Solver:
    def __init__(self, spy_network):
        self.spy_network = spy_network
        self.connected_pores, self.connected_throats = self.__get_connected_spy_network(
            self.spy_network,
        )

    def get_sub_network(self):
        return self.__get_sub_spy(self.spy_network, self.connected_pores, self.connected_throats)

    @staticmethod
    def __get_sub_spy(spy_network, sub_pores, sub_throats):
        sub_pn = {}
        for prop in spy_network.keys():
            if prop.split(".")[0] == "pore":
                sub_pn[prop] = spy_network[prop][sub_pores]
            else:
                sub_pn[prop] = spy_network[prop][sub_throats]

        def _counter():
            i = -1
            while True:
                i += 1
                yield i

        counter = _counter()
        f_counter = lambda x: next(counter) if x else 0
        new_pore_index = np.fromiter(map(f_counter, sub_pores), dtype="int")

        if len(sub_pn["throat.conns"]) == 0:
            return False
        for i in np.nditer(sub_pn["throat.conns"], op_flags=["readwrite"]):
            i[...] = new_pore_index[i]
        return sub_pn

    @staticmethod
    def __get_connected_spy_network(network):
        _, cluster_labels = Solver.__get_clusters(network)
        in_labels = np.unique(cluster_labels[network[f"pore.inlets"]])
        out_labels = np.unique(cluster_labels[network[f"pore.outlets"]])
        common_labels = np.intersect1d(in_labels, out_labels, assume_unique=True)

        connected_pores = network.pores()[np.isin(cluster_labels, common_labels)]
        # connected_throats = network.throats()[
        #    np.isin(network["throat.conns"], connected_pores).all(axis=1)
        # ]

        return np.isin(cluster_labels, common_labels), np.isin(network["throat.conns"], connected_pores).all(axis=1)

    @staticmethod
    def __get_clusters(network):
        """
        clusters are numbered starting at 0
        """
        from scipy.sparse import csgraph as csg

        am = network.create_adjacency_matrix(fmt="coo", triu=True, drop_zeros=True)
        N, Cs = csg.connected_components(am, directed=False)
        return N, Cs


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
        self.absolute_permeability = self.__calculate_K()

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
            valid_clusters_list = self.__trim(
                self.network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self.__calculate_network_flowrate(
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
            valid_clusters_list = self.__trim(
                self.network_clone, pores=pore_id_to_remove_list, throats=throat_id_to_remove_list
            )
        except ValueError:
            return 0

        flow_rate = self.__calculate_network_flowrate(
            self.network_clone, update_hydraulic_conductance, valid_clusters_list, WATER
        )
        return flow_rate / self.singlephase_flow_rate[WATER]

    def get_absolute_permeability(self):
        return self.absolute_permeability

    def __calculate_network_flowrate(self, network_clone, update_hydraulic_conductance, valid_clusters_list, phase):
        if len(valid_clusters_list) > 0:
            flow_rate = 0.0
            for valid_cluster_id in valid_clusters_list:
                self.sub_network_clone.clear()
                self.sub_network_clone.update(network_clone.copy())
                self.__trim_cluster(self.sub_network_clone, valid_cluster_id)
                flow_rate += self.__calculate_subnetwork_flowrate(
                    self.sub_network_clone, update_hydraulic_conductance, phase
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

        self.total_volume = (self.network.pn_core.pore.volume * self.network.pn_core.pore.N).sum() + (
            self.network.pn_core.throat.volume * self.network.pn_core.throat.N
        ).sum()

    def update_saturations(self, pc):
        pore_sw, pore_so = self._get_pores_saturations(pc)
        if self.phase == WATER:
            update_mask = pore_sw > self.network.pn_core.pore.water_saturation
        else:
            update_mask = pore_so > self.network.pn_core.pore.oil_saturation
        self.network.pn_core.pore.water_saturation[update_mask] = pore_sw[update_mask]
        self.network.pn_core.pore.oil_saturation[update_mask] = pore_so[update_mask]

        throat_sw, throat_so = self._get_throats_saturations(pc)
        if self.phase == WATER:
            update_mask = throat_sw > self.network.pn_core.throat.water_saturation
        else:
            update_mask = throat_so > self.network.pn_core.throat.oil_saturation
        self.network.pn_core.throat.water_saturation[update_mask] = throat_sw[update_mask]
        self.network.pn_core.throat.oil_saturation[update_mask] = throat_so[update_mask]

    @abstractmethod
    def _get_pores_saturations(self, pc):
        pass

    @abstractmethod
    def _get_throats_saturations(self, pc):
        pass

    @abstractmethod
    def _get_element_saturations(self, element_type, id, pc):
        pass

    def update_element_saturations(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat

        sw, so = self._get_element_saturations(element_type, id, pc)
        if self.phase == WATER:
            should_update = sw > element_data.water_saturation[id]
        else:
            should_update = so > element_data.oil_saturation[id]

        if should_update:
            element_data.water_saturation[id] = sw
            element_data.oil_saturation[id] = so

        if self.phase == OIL:
            return element_data.oil_saturation[id] * element_data.volume[id] / self.total_volume
        else:
            return element_data.water_saturation[id] * element_data.volume[id] / self.total_volume

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
            np.concatenate(
                (
                    self.network.pn_core.throat.N,
                    self.network.pn_core.pore.N,
                )
            ),
            self.total_volume,
        )


class BorderElements:
    """
    Class to keep track of empty elements that are in contact with filled elements
    """

    def __init__(self):
        self.throats_ids = np.array([], np.int64)
        self.pores_ids = np.array([], np.int64)

    def set_throats_ids(self, throats_ids):
        self.throats_ids = throats_ids

    def set_pores_ids(self, pores_ids):
        self.pores_ids = pores_ids

    def get_throats_ids(self):
        return self.throats_ids

    def get_pores_ids(self):
        return self.pores_ids

    def add_throats_ids(self, throats_ids):
        self.throats_ids = np.concatenate((self.throats_ids, throats_ids))

    def add_pores_ids(self, pores_ids):
        self.pores_ids = np.concatenate((self.pores_ids, pores_ids))

    def remove_throats_ids(self, throat_ids):
        self.throats_ids = np.delete(self.throats_ids, np.isin(self.throats_ids, throat_ids))

    def remove_pores_ids(self, pore_ids):
        self.pores_ids = np.delete(self.pores_ids, np.isin(self.pores_ids, pore_ids))


class CycleSimulation:
    def __init__(self, network, phase, calculate_full_conductance=False):
        self.network = network
        self.phase = phase

        self.krel_calculator = KrelCalculator(self.network, calculate_full_conductance)

        self.phase_connection_checker = PhaseConnectionChecker(network, self.phase)
        self.phase_connection_checker.reset_phase_connections()

        if USE_OPENPNM_TRAPCHECKER:
            self.trap_checker = TrapCheckerOpenpnm(network, self.phase)
        else:
            self.trap_checker = TrapChecker(network, self.phase)
        self.trap_checker.update_trapped_pores()

        self.border_elements = BorderElements()
        self.__refresh_border_elements(True, None)

        self.saturation_calculator = None

        self.current_pc = None

    def update_pc(self, pc, target_sw=None):
        invasion_pc = pc
        total_sw_delta = 0.0

        self._update_network_geometry(pc)

        while True:
            invaded_pc = invasion_pc
            sw_delta, invasion_pc, no_invasion_left = self.__update_invasions(invasion_pc)
            self._update_network_geometry(invaded_pc)
            total_sw_delta += sw_delta

            if (target_sw is None) or (total_sw_delta >= target_sw) or (no_invasion_left):
                break
        return total_sw_delta, invaded_pc, invasion_pc, no_invasion_left

    def calculate_sw_total(self):
        return self.saturation_calculator.calculate_sw_total()

    def calculate_krel(self):
        return self.krel_calculator.calculate_krw(), self.krel_calculator.calculate_kro()

    def get_pores_left(self):
        if self.phase == WATER:
            return np.logical_not(self.network.pn_core.pore.water_filled | self.network.pn_core.pore.trapped).sum()
        else:
            return np.logical_not(self.network.pn_core.pore.oil_filled | self.network.pn_core.pore.trapped).sum()

    def get_oil_filled_percentage(self):
        total_pores = self.network.get_number_of_pores()
        pores_filled = self.network.pn_core.pore.oil_filled.sum()
        return pores_filled, total_pores, int((pores_filled / total_pores) * 100)

    @abstractmethod
    def _check_invasion_condition(self, pc, element):
        pass

    def _update_element_filled_geometry(self, pc, element):
        self._update_element_hinging_contact_angles(element.type, element.id, pc)
        return self.saturation_calculator.update_element_saturations(element.type, element.id, pc)

    def _update_network_geometry(self, pc):
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

    def __update_invasions(self, pc):
        sw_delta = 0.0

        self._update_entry_pressures(pc)

        no_invasion_left = False

        invasion_pc = pc
        while True:
            element_to_fill = self._get_element_to_fill(pc, self.border_elements)
            if element_to_fill is None:
                no_invasion_left = True
                break

            element_type, element_id, invasion_pc = element_to_fill

            if not self._can_be_filled(pc, invasion_pc):
                break

            if element_type == PORE:
                element = self.network.get_pore(element_id)
            else:
                element = self.network.get_throat(element_id)

            log.debug(
                f"Invasion {element.type_name:6} ID: {element.id:6} Pc: {element.pc:.5f} Radius: {element.r:.2e} G: {element.G:.4f} n_corners: {element.n_corners} area: {element.A}"
            )

            self.__fill_element(element, invasion_pc)
            sw_delta += self._update_element_filled_geometry(invasion_pc, element)
            reached_bubble = self.__check_for_bubble(element)
            if element.is_pore():
                self.phase_connection_checker.update_phase_connections(PORE, element.id)
            else:
                self.phase_connection_checker.update_phase_connections(THROAT, element.id)
            if (self.phase == OIL and element.So == 1.0) or (self.phase == WATER and element.Sw == 1.0):
                self.trap_checker.update_trapped_pores()

            new_pores, new_throats = self.__refresh_border_elements(reached_bubble, element)
            for pore_id in new_pores:
                self._update_element_entry_pc(self.network.get_pore(pore_id), pc)
            for throat_id in new_throats:
                self._update_element_entry_pc(self.network.get_throat(throat_id), pc)
        return sw_delta, invasion_pc, no_invasion_left

    def __check_for_bubble(self, element):
        id = element.id

        if element.type == PORE:
            element_data = self.network.pn_core.pore
            neighbor_data = self.network.pn_core.throat
        else:
            element_data = self.network.pn_core.throat
            neighbor_data = self.network.pn_core.pore

        if self.phase == WATER:
            neighbor_filled = neighbor_data.water_filled
            has_phase = neighbor_data.water_saturation > 0
            connected_to_inlet = neighbor_data.water_connected_to_inlet
        else:
            neighbor_filled = neighbor_data.oil_filled
            has_phase = neighbor_data.oil_saturation > 0
            connected_to_inlet = neighbor_data.oil_connected_to_inlet

        neighbor_ids = element_data.conns[id][: element_data.n_conns[id]]
        if (has_phase[neighbor_ids] & ~connected_to_inlet[neighbor_ids]).any():
            return True
        else:
            return False

    def __refresh_border_elements(self, reached_bubble, filled_element):
        if reached_bubble:
            new_pores, new_throats = self.__reset_border_elements(
                self.network.pn_core, self.border_elements, self.phase
            )
        else:
            new_pores, new_throats = self.__update_border_elements(
                self.network.pn_core, self.border_elements, self.phase, filled_element
            )

        return new_pores, new_throats

    def __fill_element(self, element, pc):
        element.set_filled(self.phase, True)
        if element.type == PORE:
            self.network.pn_core.pore.trapped[element.id] = False
            self.network.pn_core.pore.trapped_evacuation[element.id] = False
        else:
            self.network.pn_core.throat.trapped[element.id] = False
            self.network.pn_core.throat.trapped_evacuation[element.id] = False

    def _update_element_hinging_contact_angles(self, element_type, id, pc):
        if element_type == PORE:
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat
        calculate_hinging_contact_angle(
            pc,
            element_data.corner_half_angle[id],
            element_data.b[id],
            element_data.n_corners[id],
            element_data.receding_contact_angle[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
        )

    @staticmethod
    def __reset_border_elements(pn, border_elements, phase):
        border_pore_ids, border_throat_ids = CycleSimulation.__njit_reset_border_elements(
            phase,
            pn.pore.water_filled,
            pn.pore.oil_filled,
            pn.pore.water_connected_to_inlet,
            pn.pore.oil_connected_to_inlet,
            pn.pore.trapped,
            pn.pore.inlets,
            pn.pore.conns,
            pn.pore.n_conns,
            pn.throat.water_filled,
            pn.throat.oil_filled,
            pn.throat.water_connected_to_inlet,
            pn.throat.oil_connected_to_inlet,
            pn.throat.trapped,
            pn.throat.conns,
            pn.throat.n_conns,
        )

        if len(border_throat_ids) > 0:
            new_throats = np.setdiff1d(border_throat_ids, border_elements.get_throats_ids())
            border_elements.set_throats_ids(border_throat_ids)
        else:
            new_throats = np.array([], np.int64)
            border_elements.set_throats_ids(np.array([], np.int64))
        if len(border_pore_ids) > 0:
            new_pores = np.setdiff1d(border_pore_ids, border_elements.get_pores_ids())
            border_elements.set_pores_ids(border_pore_ids)
        else:
            new_pores = np.array([], np.int64)
            border_elements.set_pores_ids(np.array([], np.int64))

        pn.pore.in_border[:] = False
        pn.throat.in_border[:] = False
        pn.pore.in_border[border_elements.get_pores_ids()] = True
        pn.throat.in_border[border_elements.get_throats_ids()] = True

        return new_pores, new_throats

    @staticmethod
    def __update_border_elements(pn, border_elements, phase, filled_element):
        new_pores, new_throats, old_pores, old_throats = CycleSimulation.__njit_update_border_elements(
            phase,
            filled_element.type,
            filled_element.id,
            pn.pore.trapped,
            pn.pore.conns,
            pn.pore.n_conns,
            pn.pore.water_filled,
            pn.pore.oil_filled,
            pn.pore.in_border,
            pn.throat.trapped,
            pn.throat.conns,
            pn.throat.n_conns,
            pn.throat.water_filled,
            pn.throat.oil_filled,
            pn.throat.in_border,
        )

        border_elements.add_pores_ids(new_pores)
        border_elements.add_throats_ids(new_throats)
        border_elements.remove_pores_ids(old_pores)
        border_elements.remove_throats_ids(old_throats)
        border_elements.get_pores_ids().sort()
        border_elements.get_throats_ids().sort()

        pn.pore.in_border[old_pores] = False
        pn.throat.in_border[old_throats] = False
        pn.pore.in_border[new_pores] = True
        pn.throat.in_border[new_throats] = True

        return new_pores, new_throats

    @staticmethod
    @nb.njit(cache=True)
    def __njit_update_border_elements(
        phase,
        filled_element_type,
        filled_element_id,
        pore_trapped,
        pore_conns,
        pore_n_conns,
        pore_water_filled,
        pore_oil_filled,
        pore_in_border,
        throat_trapped,
        throat_conns,
        throat_n_conns,
        throat_water_filled,
        throat_oil_filled,
        throat_in_border,
    ):
        # setup
        if filled_element_type == PORE:
            element_conns = pore_conns
            element_n_conns = pore_n_conns
            neighbor_trapped = throat_trapped
            neighbor_water_filled = throat_water_filled
            neighbor_oil_filled = throat_oil_filled
            neighbor_in_border = throat_in_border
        else:
            element_conns = throat_conns
            element_n_conns = throat_n_conns
            neighbor_trapped = pore_trapped
            neighbor_water_filled = pore_water_filled
            neighbor_oil_filled = pore_oil_filled
            neighbor_in_border = pore_in_border

        if phase == WATER:
            neighbor_filled = neighbor_water_filled
        else:
            neighbor_filled = neighbor_oil_filled

        # pores to add
        neighbor_ids = element_conns[filled_element_id][: element_n_conns[filled_element_id]].astype(np.int64)
        new_border_neighbors = list()
        for id in neighbor_ids:
            if not neighbor_filled[id] and not neighbor_trapped[id] and not neighbor_in_border[id]:
                new_border_neighbors.append(id)

        if filled_element_type == PORE:
            new_pores = np.empty(0, np.int64)
            new_throats = np.array(new_border_neighbors, np.int64)
        else:
            new_pores = np.array(new_border_neighbors, np.int64)
            new_throats = np.empty(0, np.int64)

        # pores to remove
        old_pores = list()
        for pore_id in range(len(pore_trapped)):
            if pore_trapped[pore_id] and pore_in_border[pore_id]:
                old_pores.append(pore_id)
        old_throats = list()
        for throat_id in range(len(throat_trapped)):
            if throat_trapped[throat_id] and throat_in_border[throat_id]:
                old_throats.append(throat_id)
        if filled_element_type == PORE:
            if pore_in_border[filled_element_id]:
                old_pores.append(filled_element_id)
        else:
            if throat_in_border[filled_element_id]:
                old_throats.append(filled_element_id)

        return new_pores, new_throats, np.array(old_pores, np.int64), np.array(old_throats, np.int64)

    @staticmethod
    @nb.njit(cache=True)
    def __njit_reset_border_elements(
        phase,
        pore_water_filled,
        pore_oil_filled,
        pore_water_connected_to_inlet,
        pore_oil_connected_to_inlet,
        pore_trapped,
        pore_inlets,
        pore_conns,
        pore_n_conns,
        throat_water_filled,
        throat_oil_filled,
        throat_water_connected_to_inlet,
        throat_oil_connected_to_inlet,
        throat_trapped,
        throat_conns,
        throat_n_conns,
    ):
        if phase == WATER:
            pore_phase_filled = pore_water_filled
            throat_phase_filled = throat_water_filled
            pore_phase_connected_to_inlet = pore_water_connected_to_inlet
            throat_phase_connected_to_inlet = throat_water_connected_to_inlet
        else:
            pore_phase_filled = pore_oil_filled
            throat_phase_filled = throat_oil_filled
            pore_phase_connected_to_inlet = pore_oil_connected_to_inlet
            throat_phase_connected_to_inlet = throat_oil_connected_to_inlet

        border_pore_ids = list()
        for pore_id in range(len(pore_inlets)):
            if pore_inlets[pore_id]:
                if not pore_phase_filled[pore_id] and not pore_trapped[pore_id]:
                    border_pore_ids.append(pore_id)
        for pore_id in range(len(pore_phase_filled)):
            if (not pore_phase_filled[pore_id]) and not pore_trapped[pore_id]:
                for throat_id in pore_conns[pore_id][: pore_n_conns[pore_id]]:
                    if throat_phase_connected_to_inlet[throat_id] and throat_phase_filled[throat_id]:
                        border_pore_ids.append(pore_id)
                        break
        border_throat_ids = list()
        for throat_id in range(len(throat_phase_filled)):
            if (not throat_phase_filled[throat_id]) and not throat_trapped[throat_id]:
                for pore_id in throat_conns[throat_id][: throat_n_conns[throat_id]]:
                    if pore_phase_connected_to_inlet[pore_id] and pore_phase_filled[pore_id]:
                        border_throat_ids.append(throat_id)
                        break

        border_pore_ids = np.unique(np.array(border_pore_ids, dtype=np.int64))
        border_throat_ids = np.unique(np.array(border_throat_ids, dtype=np.int64))

        return border_pore_ids, border_throat_ids


class TrapCheckerOpenpnm:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase
        self.pn_clone = openpnm.network.Network()

    def update_trapped_pores(self):
        self.pn_clone.clear()
        self.pn_clone.update(self.network.pn.copy())

        if self.phase == OIL:
            trapping_pores = self.network.pn_core.pore.oil_saturation == 1.0
            trapping_throats = self.network.pn_core.throat.oil_saturation == 1.0
        else:
            trapping_pores = self.network.pn_core.pore.water_saturation == 1.0
            trapping_throats = self.network.pn_core.throat.water_saturation == 1.0
        openpnm.topotools.trim(
            self.pn_clone,
            pores=trapping_pores | self.network.pn_core.pore.trapped,
            throats=trapping_throats | self.network.pn_core.throat.trapped,
        )

        self.network.pn_core["pore.trapped"] = False
        self.network.pn_core["throat.trapped"] = False

        number_of_pores = len(self.pn_clone["pore.all"])
        number_of_throats = len(self.pn_clone["throat.all"])
        if number_of_pores == 0 or number_of_throats == 0:
            return
        cluster_array = openpnm.topotools.find_clusters(self.pn_clone, mask=np.full(number_of_pores, True))

        clusters_with_outlets = np.unique(cluster_array.pore_labels[self.pn_clone["pore.outlets"]])
        for cluster_label in range(max(cluster_array.pore_labels.max(), cluster_array.throat_labels.max()) + 1):
            if cluster_label not in clusters_with_outlets:
                pore_cluster = cluster_array.pore_labels == cluster_label
                throat_cluster = cluster_array.throat_labels == cluster_label
                pore_id_list = self.pn_clone["pore.id"][pore_cluster]
                throat_id_list = self.pn_clone["throat.id"][throat_cluster]
                self.network.pn_core.pore.trapped[pore_id_list] = True
                self.network.pn_core.throat.trapped[throat_id_list] = True


class TrapChecker:
    def __init__(self, network, phase):
        self.network = network
        self.phase = phase
        self.network.pn_core.pore.trapped[:] = False
        self.network.pn_core.throat.trapped[:] = False
        self.network.pn_core.pore.trapped_evacuation[:] = False
        self.network.pn_core.throat.trapped_evacuation[:] = False

    def update_trapped_pores(self):
        if self.phase == WATER:
            self.network.pn_core["pore.trapped"] = (
                self.network.pn_core.pore.oil_filled & ~self.network.pn_core.pore.oil_connected_to_outlet
            )
            self.network.pn_core["throat.trapped"] = (
                self.network.pn_core.throat.oil_filled & ~self.network.pn_core.throat.oil_connected_to_outlet
            )
            self.network.pn_core["pore.trapped_evacuation"] = (
                self.network.pn_core.pore.trapped | ~self.network.pn_core.pore.water_connected_to_inlet
            )
            self.network.pn_core["throat.trapped_evacuation"] = (
                self.network.pn_core.throat.trapped | ~self.network.pn_core.throat.water_connected_to_inlet
            )
        else:
            self.network.pn_core["pore.trapped"] = (
                self.network.pn_core.pore.water_filled & ~self.network.pn_core.pore.water_connected_to_outlet
            )
            self.network.pn_core["throat.trapped"] = (
                self.network.pn_core.throat.water_filled & ~self.network.pn_core.throat.water_connected_to_outlet
            )
            self.network.pn_core["pore.trapped_evacuation"] = (
                self.network.pn_core.pore.trapped | ~self.network.pn_core.pore.oil_connected_to_inlet
            )
            self.network.pn_core["throat.trapped_evacuation"] = (
                self.network.pn_core.throat.trapped | ~self.network.pn_core.throat.oil_connected_to_inlet
            )


spec = [
    ("enable_control", nb.types.float64),
    ("cycle", nb.types.int32),
    ("DRAINAGE_CYCLE", nb.types.int32),
    ("IMBIBITION_CYCLE", nb.types.int32),
    ("SECOND_DRAINAGE_CYCLE", nb.types.int32),
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
    def __init__(
        self,
        pc_step,
        pc_step_min,
        pc_step_max,
        sw_step,
        pc_maximum,
        pc_minimum,
        pc_initial=0,
        enable_control=True,
    ):
        self.DRAINAGE_CYCLE = 0
        self.IMBIBITION_CYCLE = 1
        self.SECOND_DRAINAGE_CYCLE = 2

        self.enable_control = enable_control
        self.pc_step_initial = pc_step
        self.pc_step_min = pc_step_min
        self.pc_step_max = pc_step_max
        self.sw_step = sw_step
        self.pc_maximum = pc_maximum
        self.pc_minimum = pc_minimum

        self.cycle = self.DRAINAGE_CYCLE

        self.pc = pc_initial
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
        if self.cycle == self.DRAINAGE_CYCLE or self.cycle == self.SECOND_DRAINAGE_CYCLE:
            return self.increment(sw_current)
        else:
            return self.decrement(sw_current)

    def cycle_should_end(self):
        if self.cycle == self.DRAINAGE_CYCLE or self.cycle == self.SECOND_DRAINAGE_CYCLE:
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
