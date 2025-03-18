import pickle
import sys

import numpy as np
import openpnm

from py_pore_flow import log
from py_pore_flow.equations.geometry import (
    WATER,
    OIL,
    calculate_area,
    calculate_beta,
    calculate_C,
    calculate_drainage_b,
    calculate_max_b,
    calculate_r,
    calculate_shape,
    calculate_sides,
)
from py_pore_flow.equations.entry_pressure import (
    calculate_drainage_pc,
    calculate_snap_off_viability,
    calculate_spontaneous_imbibition_viability,
    calculate_snap_off_pc_network,
    calculate_imbibition_piston_like_displacement_pc_network,
)
from py_pore_flow.network import initialize_pn, Network
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.parameters.parameter_manager import ParameterManager
from py_pore_flow.simulation.drainage import DrainageSimulation
from py_pore_flow.simulation.imbibition import ImbibitionSimulation
from py_pore_flow.simulation.cycle_simulation import ConductanceCalculator, PcIncrementer


def run_two_phase_simulation(
    pn,
    parameters_dict={},
    generate_vtu=False,
    generate_cas=True,
    setup_network=True,
    drainage_snapshot_pc=None,
    imbibition_snapshot_pc=None,
    config=None,
):
    previous_limit = sys.getrecursionlimit()

    number_of_elements = len(pn["pore.all"]) + len(pn["throat.all"])
    sys.setrecursionlimit(previous_limit + number_of_elements)

    cycle = [[], []]
    Pc = [[], []]
    Sw = [[], []]
    Krw = [[], []]
    Kro = [[], []]
    log.info(f"Parameters: {parameters_dict}")
    log.info(f"Config: {config}")
    simulation = Simulation(
        pn,
        parameters_dict,
        generate_vtu,
        generate_cas,
        setup_network,
        drainage_snapshot_pc,
        imbibition_snapshot_pc,
        config,
    )

    if not config or not config["skip_drainage"]:
        Pc[0], Sw[0], Krw[0], Kro[0] = simulation.run_drainage()
        cycle[0] = [1] * len(Pc[0])
    if not simulation.parameter_manager.skip_imbibition:
        Pc[1], Sw[1], Krw[1], Kro[1] = simulation.run_imbibition()
        cycle[1] = [2] * len(Pc[1])

    sys.setrecursionlimit(previous_limit)

    return (
        np.array(cycle[0] + cycle[1]),
        np.array(Pc[0] + Pc[1]),
        np.array(Sw[0] + Sw[1]),
        np.array(Krw[0] + Krw[1]),
        np.array(Kro[0] + Kro[1]),
    )


def load_snapshot(snapshot_file):
    return StepSimulation.load_snapshot(snapshot_file)


class Simulation:
    def __init__(
        self,
        pn,
        parameters_dict={},
        generate_vtu=False,
        generate_cas=True,
        setup_network=True,
        drainage_snapshot_pc=None,
        imbibition_snapshot_pc=None,
        config=None,
    ):
        self.pn = pn
        self.drainage_snapshot_pc = drainage_snapshot_pc
        self.imbibition_snapshot_pc = imbibition_snapshot_pc
        generate_cas = generate_cas
        set_initial_values = setup_network

        self.step_simulation = MultiStepSimulation(
            self.pn,
            parameters_dict,
            config,
            generate_cas=generate_cas,
            calculate_full_conductance=generate_vtu,
            set_initial_values=set_initial_values,
        )
        self.parameter_manager = self.step_simulation.get_parameter_manager()

        self.generate_vtu = generate_vtu

        log.info(f"Max Pc: {self.parameter_manager.pc_maximum}")
        log.info(f"Min Pc: {self.parameter_manager.pc_minimum}")
        self.pc_incrementer = PcIncrementer(
            self.parameter_manager.pc_step,
            self.parameter_manager.pc_step_min,
            self.parameter_manager.pc_step_max,
            self.parameter_manager.drainage_sw_step,
            self.parameter_manager.pc_maximum,
            self.parameter_manager.pc_minimum,
            config["pc"] if config else self.parameter_manager.pc_initial,
            enable_control=self.parameter_manager.control_by_pc,
        )
        self.step_simulation.set_pc_incrementer(self.pc_incrementer)

        self.snapshot_not_taken = True

        if generate_vtu:
            pn_to_vtu(self.pn, self.pc_incrementer.get_pc(), 1, 0)

    def run_drainage(self):
        self.step_simulation.set_max_pc(self.parameter_manager.pc_maximum)
        return self.__run_cycle("drainage")

    def run_imbibition(self):
        return self.__run_cycle("imbibition")

    def __run_cycle(self, cycle_name):
        log.info(f"Start {cycle_name}")

        if cycle_name == "drainage":
            self.pc_incrementer.set_cycle(self.pc_incrementer.DRAINAGE_CYCLE)
        else:
            self.pc_incrementer.set_cycle(self.pc_incrementer.IMBIBITION_CYCLE)
        self.step_simulation.set_cycle(cycle_name)

        Pc_array = []
        Sw_array = []
        Krw_array = []
        Kro_array = []

        simulation_end = False

        while not simulation_end:
            last_pc = None
            current_pc = self.pc_incrementer.get_pc()

            while current_pc != last_pc:
                last_pc = current_pc
                self.__check_snapshot(cycle_name, current_pc)

                (
                    current_pc,
                    Sw,
                    Krw,
                    Kro,
                    pores_filled,
                    total_pores,
                    percentage,
                ) = self.step_simulation.update(current_pc)

                Pc_array.append(current_pc)
                Sw_array.append(Sw)
                Krw_array.append(Krw)
                Kro_array.append(Kro)

                log.info(
                    f"Pc: {current_pc:8.2f}, Sw: {Sw:.6f}, Krw: {Krw:.6f}, Kro: {Kro:.6f}, Filled: {pores_filled:4}/{total_pores} ({percentage:2}%)."
                )

                if self.generate_vtu:
                    pn_to_vtu(self.pn, current_pc, Sw, 1 if cycle_name == "drainage" else 2)

                self.pc_incrementer.set_pc(current_pc)

                log.info(
                    f"Step {cycle_name}, {simulation_end}, {Sw}, {self.parameter_manager.enforced_swi_1}, {self.parameter_manager.enforced_swi_2}"
                )
                if self.parameter_manager.end_at_sor and cycle_name == "imbibition" and Kro <= 0:
                    simulation_end = True
                elif self.step_simulation.cycle_should_end():
                    simulation_end = True
                elif Sw <= self.parameter_manager.enforced_swi_1 and cycle_name == "drainage":
                    simulation_end = True
                    break
                elif Sw >= self.parameter_manager.enforced_swi_2 and cycle_name == "imbibition":
                    simulation_end = True
                    break

            if self.parameter_manager.control_by_pc:
                self.pc_incrementer.update_pc(Sw)

        self.__check_snapshot_final(cycle_name, current_pc)

        return Pc_array, Sw_array, Krw_array, Kro_array

    def __check_snapshot(self, cycle_name, current_pc):
        if (
            isinstance(self.drainage_snapshot_pc, int)
            or isinstance(self.drainage_snapshot_pc, float)
            and self.snapshot_not_taken
            and cycle_name == "drainage"
            and current_pc >= self.drainage_snapshot_pc
        ) or (
            self.imbibition_snapshot_pc is not None
            and self.snapshot_not_taken
            and cycle_name == "imbibition"
            and current_pc <= self.imbibition_snapshot_pc
        ):
            snapshot_dict = self.step_simulation.get_snapshot(current_pc)
            self.__save_snapshot(snapshot_dict)

    def __check_snapshot_final(self, cycle_name, current_pc):
        if (
            isinstance(self.drainage_snapshot_pc, str)
            and self.drainage_snapshot_pc == "final"
            and self.snapshot_not_taken
            and cycle_name == "drainage"
        ):
            snapshot_dict = self.step_simulation.get_snapshot(current_pc)
            snapshot_dict["cycle"] = "imbibition"
            self.__save_snapshot(snapshot_dict)

    def __save_snapshot(self, snapshot_dict):
        with open("snapshot.bin", "wb") as fp:
            pickle.dump(snapshot_dict, fp)
        self.snapshot_not_taken = False
        log.info("Snapshot created")


def calculate_initial_values(network):
    for pore in network.pore_list:
        calculate_element_initial_values(pore)

    for throat in network.throat_list:
        calculate_element_initial_values(throat)

    return network


def calculate_drainage_values(network, max_pc):
    for pore in network.pore_list:
        calculate_element_drainage_values(pore, max_pc)

    for throat in network.throat_list:
        calculate_element_drainage_values(throat, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    conductance_calculator.calculate_singlephase_conductance(OIL)
    conductance_calculator.calculate_singlephase_conductance(WATER)

    network.pn_core["pore.trapped"] = False
    network.pn_core["throat.trapped"] = False
    network.pn_core["pore.trapped_evacuation"] = False
    network.pn_core["throat.trapped_evacuation"] = False

    network.pn_core.pore.hinging_contact_angle[:, :] = network.pn_core.pore.receding_contact_angle[:, np.newaxis]
    network.pn_core.throat.hinging_contact_angle[:, :] = network.pn_core.throat.receding_contact_angle[:, np.newaxis]

    return network


def calculate_imbibition_values(network, max_pc):
    for pore in network.pore_list:
        calculate_element_imbibition_values(pore, max_pc)

    for throat in network.throat_list:
        calculate_element_imbibition_values(throat, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    conductance_calculator.calculate_singlephase_conductance(OIL)
    conductance_calculator.calculate_singlephase_conductance(WATER)

    network.pn_core["pore.trapped"] = False
    network.pn_core["throat.trapped"] = False
    network.pn_core["pore.trapped_evacuation"] = False
    network.pn_core["throat.trapped_evacuation"] = False

    calculate_snap_off_values(network, max_pc)
    calculate_piston_like_values(network, max_pc)

    return network


def calculate_piston_like_values(network, max_pc):
    Rmin = calculate_r(max_pc, 100000)

    network.pn_core["throat.piston_like_pressure"] = calculate_imbibition_piston_like_displacement_pc_network(
        network.pn_core.throat.shape_factor,
        network.pn_core.throat.radius,
        network.pn_core.throat.b,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.shape,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.n_corners,
        Rmin,
    )

    network.pn_core["pore.piston_like_pressure"] = calculate_imbibition_piston_like_displacement_pc_network(
        network.pn_core.pore.shape_factor,
        network.pn_core.pore.radius,
        network.pn_core.pore.b,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.shape,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.n_corners,
        Rmin,
    )


def calculate_snap_off_values(network, max_pc):
    Rmin = calculate_r(max_pc, 100000)

    network.pn_core["throat.snap_off_pressure"] = calculate_snap_off_pc_network(
        network.pn_core.throat.radius,
        network.pn_core.throat.corner_half_angle,
        network.pn_core.throat.advancing_contact_angle,
        network.pn_core.throat.receding_contact_angle,
        network.pn_core.throat.n_corners,
        Rmin,
        np.zeros(len(network.pn_core.throat.radius)),
    )

    network.pn_core["pore.snap_off_pressure"] = calculate_snap_off_pc_network(
        network.pn_core.pore.radius,
        network.pn_core.pore.corner_half_angle,
        network.pn_core.pore.advancing_contact_angle,
        network.pn_core.pore.receding_contact_angle,
        network.pn_core.pore.n_corners,
        Rmin,
        np.zeros(len(network.pn_core.pore.radius)),
    )


def calculate_element_initial_values(
    element,
):
    element.shape, element.G = calculate_shape(element.G)
    element.A = calculate_area(element.r, element.G)
    element.beta = calculate_beta(element.G, element.shape)
    element.sides = calculate_sides(element.beta, element.r, element.n_corners)
    element.max_b = calculate_max_b(element.sides, element.n_corners)


def calculate_element_drainage_values(element, max_pc):
    element.b = calculate_drainage_b(
        element.beta, np.full(element.n_corners, element.receding_contact_angle), element.n_corners, max_pc
    )
    element.C_r = calculate_C(
        np.full(element.n_corners, element.receding_contact_angle), element.beta, element.n_corners
    )
    element.C_a = calculate_C(
        np.full(element.n_corners, element.advancing_contact_angle), element.beta, element.n_corners
    )
    element.set_pc(
        calculate_drainage_pc(
            element.G,
            element.r,
            element.receding_contact_angle,
            element.shape,
            element.beta,
            element.n_corners,
        )
    )


def calculate_element_imbibition_values(element, max_pc):
    Rmin = calculate_r(max_pc, 100000)

    element.b = calculate_drainage_b(
        element.beta, np.full(element.n_corners, element.receding_contact_angle), element.n_corners, max_pc
    )
    element.C_r = calculate_C(
        np.full(element.n_corners, element.receding_contact_angle), element.beta, element.n_corners
    )
    element.C_a = calculate_C(
        np.full(element.n_corners, element.advancing_contact_angle), element.beta, element.n_corners
    )
    element.spontaneous_imbibition = calculate_spontaneous_imbibition_viability(
        element.advancing_contact_angle,
        element.receding_contact_angle,
        element.beta,
        element.n_corners,
        element.r,
        element.G,
        Rmin,
    )

    element.snap_off_viability = calculate_snap_off_viability(
        element.receding_contact_angle,
        element.beta,
        element.n_corners,
    )


class StepSimulation:
    def __init__(
        self,
        pn,
        parameters_dict={},
        config={},
        generate_cas=True,
        calculate_full_conductance=False,
        set_initial_values=True,
    ):
        self.simulation = None
        self.max_pc = config["max_pc"] if config else None
        self.calculate_full_conductance = calculate_full_conductance
        self.is_first_step = True
        self.current_pc = 0
        self.cycle = "drainage"

        self.parameter_manager = ParameterManager(parameters_dict)

        if set_initial_values:
            initialize_pn(pn)

        self.network = Network(pn)

        if generate_cas:
            self.parameter_manager.generate_contact_angles(self.network)

        if set_initial_values:
            calculate_initial_values(self.network)

    def set_max_pc(self, max_pc):
        self.max_pc = max_pc

    def set_cycle(self, cycle="imbibition"):
        self.cycle = cycle
        if cycle == "drainage":
            self.parameter_manager.update_initial_cas(self.network)
            calculate_drainage_values(self.network, self.max_pc)
            self.simulation = DrainageSimulation(self.network, self.calculate_full_conductance)
            self.max_pc = 0
        else:
            self.parameter_manager.update_equilibrium_cas(self.network)
            calculate_imbibition_values(self.network, self.max_pc)
            self.simulation = ImbibitionSimulation(self.network, self.max_pc, self.calculate_full_conductance)

    def update(self, pc):
        self.current_pc = self._perform_step(pc)

        Sw = self.simulation.calculate_sw_total()
        Krw, Kro = self.simulation.calculate_krel()

        pores_filled, total_pores, percentage = self.simulation.get_oil_filled_percentage()

        return self.current_pc, Sw, Krw, Kro, pores_filled, total_pores, percentage

    def get_pores_left(self):
        return self.simulation.get_pores_left()

    def get_parameter_manager(self):
        return self.parameter_manager

    def get_snapshot(self, current_pc):
        snapshot_dict = {}
        snapshot_dict["parameters"] = self.parameter_manager.parameters_dict
        snapshot_dict["pc"] = current_pc
        snapshot_dict["max_pc"] = self.max_pc
        snapshot_dict["cycle"] = self.cycle
        snapshot_dict["network"] = self.network.pn.copy()
        return snapshot_dict

    @staticmethod
    def load_snapshot(snapshot_file):
        with open(snapshot_file, "rb") as fp:
            snapshot_dict = pickle.load(fp)

        pn = openpnm.network.Network()
        pn.update(snapshot_dict["network"])
        parameters = snapshot_dict["parameters"]
        parameters["pc_initial"] = snapshot_dict["pc"]

        config = {}
        if snapshot_dict["cycle"] == "imbibition":
            config["skip_drainage"] = True
        else:
            config["skip_drainage"] = False
        config["pc"] = snapshot_dict["pc"]
        config["max_pc"] = snapshot_dict["max_pc"]

        return pn, parameters, config

    def _perform_step(self, pc):
        return self.__step_by_pc(pc)

    def __step_by_pc(self, pc):
        self.simulation.update_pc(pc)
        self.max_pc = max(self.max_pc, pc)
        return pc


class MultiStepSimulation(StepSimulation):
    def __init__(
        self,
        pn,
        parameters_dict={},
        config={},
        generate_cas=True,
        calculate_full_conductance=False,
        set_initial_values=True,
    ):
        super().__init__(pn, parameters_dict, config, generate_cas, calculate_full_conductance, set_initial_values)
        self.pc_incrementer = None
        self.sw_step = None

    def set_pc_incrementer(self, pc_incrementer):
        self.pc_incrementer = pc_incrementer

    def set_cycle(self, cycle="imbibition"):
        super().set_cycle(cycle)
        self.cycle = cycle
        if cycle == "drainage":
            self.sw_step = None if self.parameter_manager.control_by_pc else self.parameter_manager.drainage_sw_step
        else:
            self.sw_step = None if self.parameter_manager.control_by_pc else self.parameter_manager.imbibition_sw_step
        self.is_first_step = True

    def cycle_should_end(self):
        return self.pc_incrementer.cycle_should_end() or self.get_pores_left() == 0

    def _perform_step(self, pc):
        if self.sw_step is None or self.is_first_step:
            current_pc = super()._perform_step(pc)
        else:
            current_pc = self.__step_by_sw(pc)
        self.is_first_step = False
        return current_pc

    def __step_by_sw(self, pc):
        total_sw_delta = 0.0
        next_pc = pc
        while True:
            sw_delta, current_pc, next_pc = self.simulation.update_pc(next_pc, self.sw_step)
            self.max_pc = max(self.max_pc, current_pc)
            total_sw_delta += sw_delta
            if total_sw_delta >= self.sw_step or self.cycle_should_end():
                break
            if current_pc == next_pc:
                self.pc_incrementer.set_pc(current_pc)
                next_pc = self.pc_incrementer.update_pc()
        return current_pc
