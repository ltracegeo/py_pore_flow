from parameters.parameter_manager import DEFAULT_PARAMETERS, ParameterManager

from equations import (
    calculate_beta,
    calculate_C,
    calculate_drainage_b,
    calculate_drainage_pc,
    calculate_shape,
    WATER,
    OIL,
)
from network import initialize_pn, Network
from simulation.drainage import run_drainage
from simulation.imbibition import run_imbibition
from simulation.simulation_base import ConductanceCalculator, PcIncrementer


class Simulation:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS, debug=False):
        self.debug = debug

        self.parameter_manager = ParameterManager(parameters_dict)
        self.pc_incrementer = PcIncrementer(self.parameter_manager)

        initialize_pn(pn)
        self.network = Network(pn)

    def run_drainage(self):
        self.parameter_manager.update_initial_cas(self.network)
        calculate_initial_values(self.network, self.pc_incrementer.get_pc_max())
        self.pc_incrementer.set_pc_to_min()
        return run_drainage(self.network, self.pc_incrementer, debug=self.debug)

    def run_imbibition(self):
        self.parameter_manager.update_equilibrium_cas(self.network)
        calculate_initial_values(self.network, self.pc_incrementer.get_pc_max())
        self.pc_incrementer.set_pc_to_max()
        return run_imbibition(self.network, self.pc_incrementer, debug=self.debug)


def calculate_initial_values(network, max_pc):
    for pore in network.pore_list:
        calculate_element_initial_values(pore, max_pc)

    for throat in network.throat_list:
        calculate_element_initial_values(throat, max_pc)

    conductance_calculator = ConductanceCalculator(network)
    oil_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(OIL)
    water_singlephase_conductance = conductance_calculator.calculate_singlephase_conductance(WATER)
    network.pn["throat.oil_singlephase_conductance"] = oil_singlephase_conductance
    network.pn["throat.water_singlephase_conductance"] = water_singlephase_conductance

    return network


def calculate_element_initial_values(element, max_pc):
    element.shape = calculate_shape(element.G)
    element.beta = calculate_beta(element.G, element.shape)
    element.b = calculate_drainage_b(element.beta, element.receding_contact_angle, max_pc)
    element.C_r = calculate_C(element.receding_contact_angle, element.beta)
    element.C_a = calculate_C(element.advancing_contact_angle, element.beta)
    element.set_pc(
        calculate_drainage_pc(element.G, element.r, element.receding_contact_angle, element.shape, element.C_r, 0)
    )
