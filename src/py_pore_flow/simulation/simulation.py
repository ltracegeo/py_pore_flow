import sys

import numpy as np

from py_pore_flow.parameters.parameter_manager import DEFAULT_PARAMETERS, ParameterManager
from py_pore_flow.equations import (
    calculate_beta,
    calculate_C,
    calculate_drainage_b,
    calculate_drainage_pc,
    calculate_max_b,
    calculate_r,
    calculate_shape,
    calculate_sides,
    calculate_snap_off_viability,
    calculate_spontaneous_imbibition_viability,
    WATER,
    OIL,
)
from py_pore_flow.network import initialize_pn, Network
from py_pore_flow.output import pn_to_vtu
from py_pore_flow.simulation.drainage import run_drainage
from py_pore_flow.simulation.imbibition import run_imbibition
from py_pore_flow.simulation.simulation_base import ConductanceCalculator, PcIncrementer
from py_pore_flow.graphic.network_plot import WATER, OIL, NetworkPlot


def run_two_phase_simulation(
    pn, parameters_dict=DEFAULT_PARAMETERS, generate_vtu=False, plot_elements=False, debug=False
):
    previous_limit = sys.getrecursionlimit()

    number_of_elements = len(pn["pore.all"]) + len(pn["throat.all"])
    sys.setrecursionlimit(previous_limit + number_of_elements)

    parameters = DEFAULT_PARAMETERS
    parameters.update(parameters_dict)

    cycle = [None] * 2
    Pc = [None] * 2
    Sw = [None] * 2
    Krw = [None] * 2
    Kro = [None] * 2

    simulation = Simulation(pn, parameters, generate_vtu, plot_elements, debug)
    Pc[0], Sw[0], Krw[0], Kro[0] = simulation.run_drainage()
    Pc[1], Sw[1], Krw[1], Kro[1] = simulation.run_imbibition()

    cycle[0] = [1] * len(Pc[0])
    cycle[1] = [2] * len(Pc[1])

    sys.setrecursionlimit(previous_limit)

    return (
        np.array(cycle[0] + cycle[1]),
        np.array(Pc[0] + Pc[1]),
        np.array(Sw[0] + Sw[1]),
        np.array(Krw[0] + Krw[1]),
        np.array(Kro[0] + Kro[1]),
    )


class Simulation:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS, generate_vtu=False, plot_elements=False, debug=False):
        self.debug = debug
        self.plot_elements = plot_elements
        self.generate_vtu = generate_vtu

        self.parameter_manager = ParameterManager(parameters_dict)
        self.pc_incrementer = PcIncrementer(
            self.parameter_manager.pc_step,
            self.parameter_manager.pc_step_min,
            self.parameter_manager.pc_step_max,
            self.parameter_manager.sw_step,
            self.parameter_manager.pc_maximum,
            self.parameter_manager.pc_minimum,
        )

        initialize_pn(pn)
        self.network = Network(pn)

        self.parameter_manager.generate_contact_angles(self.network)

        calculate_initial_values(self.network)

        if generate_vtu:
            pn_to_vtu(pn, self.pc_incrementer.get_pc(), 1, 0)

        if self.plot_elements:
            self.network_plot = NetworkPlot(pn)
            for id in self.network.pn.throats():
                self.network_plot.create_throat_plot(id)
            for id in self.network.pn.pores():
                self.network_plot.create_pore_plot(id)
        else:
            self.network_plot = None

    def run_drainage(self):
        self.parameter_manager.update_initial_cas(self.network)
        calculate_drainage_values(self.network, self.pc_incrementer.get_pc_max())
        return run_drainage(self.network, self.pc_incrementer, self.network_plot, self.generate_vtu, debug=self.debug)

    def run_imbibition(self):
        self.parameter_manager.update_equilibrium_cas(self.network)
        calculate_imbibition_values(self.network, self.pc_incrementer.get_pc_max())
        return run_imbibition(
            self.network,
            self.pc_incrementer,
            self.network_plot,
            self.parameter_manager.end_at_sor,
            self.generate_vtu,
            debug=self.debug,
        )


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
    oil_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(OIL)
    water_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(WATER)
    network.pn["throat.oil_singlephase_conductance"] = oil_singlephase_conductance
    network.pn["throat.water_singlephase_conductance"] = water_singlephase_conductance

    return network


def calculate_imbibition_values(network, max_pc):
    for pore in network.pore_list:
        calculate_element_imbibition_values(pore, max_pc)

    for throat in network.throat_list:
        calculate_element_imbibition_values(throat, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    oil_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(OIL)
    water_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(WATER)
    network.pn["throat.oil_singlephase_conductance"] = oil_singlephase_conductance
    network.pn["throat.water_singlephase_conductance"] = water_singlephase_conductance

    return network


def calculate_element_initial_values(
    element,
):
    element.shape = calculate_shape(element.G)
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
            element.C_r,
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
