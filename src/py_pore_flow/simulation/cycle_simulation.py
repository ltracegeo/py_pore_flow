from abc import abstractmethod

import numba as nb
from numba.experimental import jitclass
import numpy as np
import openpnm

from pyflowsolver import NetworkManager

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    MAX_NUMBER_OF_CORNERS,
    calculate_drainage_element_saturation,
    calculate_drainage_network_saturation,
    calculate_hinging_contact_angle,
    calculate_imbibition_element_saturation,
    calculate_imbibition_network_saturation,
    calculate_sw_total,
)
from py_pore_flow.equations.conductance import (
    calculate_absolute_permeability,
    calculate_network_conductance,
    calculate_network_multiphase_oil_conductivity,
    calculate_network_multiphase_water_conductivity,
    calculate_network_singlephase_conductivity,
)
from py_pore_flow.network import Network
from pypardiso import spsolve
from scipy.sparse import csr_matrix

# Flags
USE_OPENPNM_SOLVER = False
USE_OPENPNM_TRAPCHECKER = False

TRAP_TOLERANCE = 0.0


# Constants
PORE = 0
THROAT = 1


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
        if phase == WATER:
            calculate_network_singlephase_conductivity(
                self.network.pn_core.pore.water_singlephase_conductivity,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                WATER,
            )
            calculate_network_singlephase_conductivity(
                self.network.pn_core.throat.water_singlephase_conductivity,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                WATER,
            )
        else:
            calculate_network_singlephase_conductivity(
                self.network.pn_core.pore.oil_singlephase_conductivity,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                OIL,
            )
            calculate_network_singlephase_conductivity(
                self.network.pn_core.throat.oil_singlephase_conductivity,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                OIL,
            )

    def __calculate_multiphase_conductivity(self, phase):
        if phase == WATER:
            calculate_network_multiphase_water_conductivity(
                self.network.pn_core.pore.water_multiphase_conductivity,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                self.network.pn_core.pore.water_saturation,
                self.network.pn_core.pore.hinging_contact_angle,
                self.network.pn_core.pore.b,
                self.network.pn_core.pore.corner_half_angle,
                self.network.pn_core.pore.n_corners,
                self.network.pn_core.pore.water_filled,
            )
            calculate_network_multiphase_water_conductivity(
                self.network.pn_core.throat.water_multiphase_conductivity,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                self.network.pn_core.throat.water_saturation,
                self.network.pn_core.throat.hinging_contact_angle,
                self.network.pn_core.throat.b,
                self.network.pn_core.throat.corner_half_angle,
                self.network.pn_core.throat.n_corners,
                self.network.pn_core.throat.water_filled,
            )
        else:
            calculate_network_multiphase_oil_conductivity(
                self.network.pn_core.pore.oil_multiphase_conductivity,
                self.network.pn_core.pore.area,
                self.network.pn_core.pore.shape_factor,
                self.network.pn_core.pore.shape,
                self.network.pn_core.pore.oil_saturation,
                self.network.pn_core.pore.b,
                self.network.pn_core.pore.bo,
                self.network.pn_core.pore.corner_half_angle,
                self.network.pn_core.pore.hinging_contact_angle,
                np.tile(self.network.pn_core.pore.advancing_contact_angle[:, np.newaxis], MAX_NUMBER_OF_CORNERS),
                self.network.pn_core.pore.corner_oil_layer,
                self.network.pn_core.pore.n_corners,
                self.network.pn_core.pore.oil_filled,
            )
            calculate_network_multiphase_oil_conductivity(
                self.network.pn_core.throat.oil_multiphase_conductivity,
                self.network.pn_core.throat.cross_sectional_area,
                self.network.pn_core.throat.shape_factor,
                self.network.pn_core.throat.shape,
                self.network.pn_core.throat.oil_saturation,
                self.network.pn_core.throat.b,
                self.network.pn_core.throat.bo,
                self.network.pn_core.throat.corner_half_angle,
                self.network.pn_core.throat.hinging_contact_angle,
                np.tile(self.network.pn_core.throat.advancing_contact_angle[:, np.newaxis], MAX_NUMBER_OF_CORNERS),
                self.network.pn_core.throat.corner_oil_layer,
                self.network.pn_core.throat.n_corners,
                self.network.pn_core.throat.oil_filled,
            )


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
        connected_pores, connected_throats = get_connected_spy_network(
            self.network.pn,
        )
        sub_network = get_sub_spy(self.network.pn, connected_pores, connected_throats)
        if sub_network is False:
            return 0

        conn = sub_network["throat.conns"].astype(np.int32)
        cond = sub_network["throat.hydraulic_conductance"].astype(np.float64)
        inlets = sub_network["pore.inlets"]
        outlets = sub_network["pore.outlets"]
        network_manager = NetworkManager(
            conn=conn,
            cond=cond,
            inlets=inlets,
            outlets=outlets,
        )
        network_manager.generate_sparse_system()
        a_sparse_array, b_array = network_manager.get_sparse_system()

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

        return flow_rate


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
        self.total_volume = self.network.pn_core.pore.volume.sum() + self.network.pn_core.throat.volume.sum()

    def update_saturations(self, pc):
        if self.phase == OIL:
            self.__update_drainage_saturations(pc)
        else:
            self.__update_imbibition_saturations()

    def update_element_saturations(self, type_name, id, pc):
        if type_name == "pore":
            element_data = self.network.pn_core.pore
        else:
            element_data = self.network.pn_core.throat

        if self.phase == OIL:
            self.__update_drainage_element_saturations(type_name, id, pc)
            return element_data.oil_saturation[id] * element_data.volume[id] / self.total_volume
        else:
            self.__update_imbibition_element_saturations(type_name, id)
            return element_data.water_saturation[id] * element_data.volume[id] / self.total_volume

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
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        (element_data.water_saturation[id], element_data.oil_saturation[id],) = calculate_drainage_element_saturation(
            element_data.shape[id],
            element_data.C_r[id],
            pc,
            area,
            element_data.oil_filled[id],
        )

    def __update_imbibition_element_saturations(self, type_name, id):
        if type_name == "pore":
            element_data = self.network.pn_core.pore
            area = self.network.pn_core.pore.area[id]
        else:
            element_data = self.network.pn_core.throat
            area = self.network.pn_core.throat.cross_sectional_area[id]

        (element_data.water_saturation[id], element_data.oil_saturation[id],) = calculate_imbibition_element_saturation(
            area,
            element_data.oil_filled[id],
            element_data.advancing_contact_angle[id],
            element_data.hinging_contact_angle[id],
            element_data.corner_half_angle[id],
            element_data.b[id],
            element_data.bo[id],
            element_data.n_corners[id],
            element_data.corner_oil_layer[id],
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

        if USE_OPENPNM_TRAPCHECKER:
            self.trap_checker = TrapCheckerOpenpnm(network, self.phase)
        else:
            self.trap_checker = TrapChecker(network, self.phase)
        self.trap_checker.update_trapped_pores()

        self.saturation_calculator = SaturationCalculator(network, phase)

        self.current_pc = None

    def update_pc(self, pc, target_sw=None):
        invasion_pc = pc
        total_sw_delta = 0.0
        while True:
            if target_sw is not None:
                invasion_target_sw = target_sw - total_sw_delta
            else:
                invasion_target_sw = None
            invaded_pc = invasion_pc
            sw_delta, invasion_pc = self._update_invasions(invasion_pc, invasion_target_sw)
            self._update_network_geometry(invaded_pc)
            if target_sw is None:
                break
            elif sw_delta == 0.0:
                break
            else:
                total_sw_delta += sw_delta
                if total_sw_delta >= target_sw:
                    break
        return total_sw_delta, invaded_pc, invasion_pc

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
        sw_delta = 0.0

        self._update_entry_pressures(pc)

        invasion_pc = pc
        while True:
            element_to_fill = self._get_element_to_fill(pc, self.border_elements)
            if element_to_fill is None:
                break

            element_type, element_id, invasion_pc = element_to_fill

            if (target_sw is not None and sw_delta >= target_sw) or not self._can_be_filled(pc, invasion_pc):
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
            if (self.phase == OIL and element.So == 1.0) or (self.phase == WATER and element.Sw == 1.0):
                self.trap_checker.update_trapped_pores()
                self.__remove_trapped_from_border()

        return sw_delta, invasion_pc

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
            if not neighbor.is_filled(self.phase) and not neighbor.trapped:
                if neighbor.is_pore():
                    self.border_elements.add_pore(neighbor.get_id())
                else:
                    self.border_elements.add_throat(neighbor.get_id())
                self._update_element_entry_pc(neighbor, pc)

    def __update_element_hinging_contact_angles(self, type_name, id, pc):
        if type_name == "pore":
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
        self.nP = len(self.network.pn_core.pore.all)
        self.nT = len(self.network.pn_core.throat.all)
        self.parent_array = np.arange(self.nP + self.nT, dtype=np.uint32)
        self.size_array = np.ones((self.nP + self.nT))
        self.network.pn_core.pore.trapped[:] = False
        self.network.pn_core.throat.trapped[:] = False

    def reset_status(self):
        # self.network.pn_core.pore.trapped[:] = False
        # self.network.pn_core.throat.trapped[:] = False
        self._jit_reset_status(
            self.parent_array,
            self.size_array,
            self.nP,
            self.nT,
        )

    def update_trapped_pores(self):
        self.reset_status()
        self._jit_update_trapped_pores(
            self.parent_array,
            self.size_array,
            self.nP,
            self.nT,
            self.network.pn_core.pore.trapped,
            self.network.pn_core.throat.trapped,
            self.network.pn_core.pore.oil_saturation,
            self.network.pn_core.throat.oil_saturation,
            self.network.pn_core.throat.conns,
            self.network.pn_core.pore.outlets,
            self.phase,
        )

    @staticmethod
    @nb.njit
    def _jit_update_trapped_pores(
        parent_array,
        size_array,
        nP,
        nT,
        pore_trapped,
        throat_trapped,
        pore_oil_saturation,
        throat_oil_saturation,
        throat_conns,
        pore_outlets,
        phase,
    ):
        union = lambda p, q: wqupc_union(p, q, parent_array, size_array)
        find = lambda p: wqupc_find(p, parent_array)

        if phase == OIL:
            is_trapped = lambda x: x >= (1.0 - TRAP_TOLERANCE)
        elif phase == WATER:
            is_trapped = lambda x: x <= (TRAP_TOLERANCE)
        for throat in range(nT):
            t_saturation = throat_oil_saturation[throat]
            if is_trapped(t_saturation):
                continue
            pore_0, pore_1 = throat_conns[throat]
            p0_saturation = pore_oil_saturation[pore_0]
            p1_saturation = pore_oil_saturation[pore_1]
            if not is_trapped(p0_saturation):
                union(throat, (pore_0 + nT))
            if not is_trapped(p1_saturation):
                union(throat, (pore_1 + nT))

        outlets_clusters = set()

        for pore in range(nP):
            if pore_outlets[pore] == True:
                cluster = find(pore + nT)
                outlets_clusters.add(cluster)

        for throat in range(nT):
            cluster = find(throat)
            if cluster not in outlets_clusters:
                throat_trapped[throat] = True

        for pore in range(nP):
            cluster = find(pore + nT)
            if cluster not in outlets_clusters:
                pore_trapped[pore] = True

    @staticmethod
    @nb.njit
    def _jit_reset_status(parent_array, size_array, nP, nT):
        for i in range(nP + nT):
            parent_array[i] = i
            size_array[i] = 1


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


def get_sub_spy(spy_network, sub_pores, sub_throats):
    sub_pn = {}
    for prop in spy_network.keys():
        if prop.split(".")[0] == "pore":
            sub_pn[prop] = spy_network[prop][sub_pores]
        else:
            sub_pn[prop] = spy_network[prop][sub_throats]

    counter = _counter()
    f_counter = lambda x: next(counter) if x else 0
    new_pore_index = np.fromiter(map(f_counter, sub_pores), dtype="int")

    if len(sub_pn["throat.conns"]) == 0:
        return False
    for i in np.nditer(sub_pn["throat.conns"], op_flags=["readwrite"]):
        i[...] = new_pore_index[i]
    return sub_pn


def get_connected_spy_network(network):
    _, cluster_labels = get_clusters(network)
    in_labels = np.unique(cluster_labels[network[f"pore.inlets"]])
    out_labels = np.unique(cluster_labels[network[f"pore.outlets"]])
    common_labels = np.intersect1d(in_labels, out_labels, assume_unique=True)

    connected_pores = network.pores()[np.isin(cluster_labels, common_labels)]
    # connected_throats = network.throats()[
    #    np.isin(network["throat.conns"], connected_pores).all(axis=1)
    # ]

    return np.isin(cluster_labels, common_labels), np.isin(network["throat.conns"], connected_pores).all(axis=1)


def _counter():
    i = -1
    while True:
        i += 1
        yield i


def get_clusters(network):
    """
    clusters are numbered starting at 0
    """
    from scipy.sparse import csgraph as csg

    am = network.create_adjacency_matrix(fmt="coo", triu=True, drop_zeros=True)
    N, Cs = csg.connected_components(am, directed=False)
    return N, Cs


@nb.njit
def wqupc_find(p, parent_array):
    root = p
    while root != parent_array[root]:
        # parent_array[root] = parent_array[parent_array[root]]
        root = parent_array[root]

    while p != root:
        next_p = parent_array[p]
        parent_array[p] = root
        p = next_p

    return root


@nb.njit
def wqupc_union(p, q, parent_array, size_array):
    # Find the roots of the elements p and q
    rootP = wqupc_find(p, parent_array)
    rootQ = wqupc_find(q, parent_array)

    if rootP == rootQ:
        return

    if size_array[rootP] < size_array[rootQ]:
        parent_array[rootP] = rootQ
        size_array[rootQ] += size_array[rootP]
    else:
        parent_array[rootQ] = rootP
        size_array[rootP] += size_array[rootQ]
