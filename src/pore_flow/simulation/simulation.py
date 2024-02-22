import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pore_flow.parameters.parameter_manager import DEFAULT_PARAMETERS, ParameterManager
from pore_flow.equations import (
    calculate_beta,
    calculate_C,
    calculate_drainage_b,
    calculate_drainage_pc,
    calculate_shape,
    WATER,
    OIL,
)
from pore_flow.network import initialize_pn, Network
from pore_flow.simulation.drainage import run_drainage
from pore_flow.simulation.imbibition import run_imbibition
from pore_flow.simulation.simulation_base import ConductanceCalculator, PcIncrementer


def run_two_phase_simulation(pn, parameters_dict=DEFAULT_PARAMETERS, generate_vtu=False, debug=False):
    parameters = DEFAULT_PARAMETERS
    parameters.update(parameters_dict)

    cycle = [None] * 2
    Pc = [None] * 2
    Sw = [None] * 2
    Krw = [None] * 2
    Kro = [None] * 2

    simulation = Simulation(pn, parameters, generate_vtu, debug)
    Pc[0], Sw[0], Krw[0], Kro[0] = simulation.run_drainage()
    Pc[1], Sw[1], Krw[1], Kro[1] = simulation.run_imbibition()

    cycle[0] = [1] * len(Pc[0])
    cycle[1] = [2] * len(Pc[1])

    pn_to_vtk(pn)

    return (
        cycle[0] + cycle[1],
        Pc[0] + Pc[1],
        Sw[0] + Sw[1],
        Krw[0] + Krw[1],
        Kro[0] + Kro[1],
    )


class Simulation:
    def __init__(self, pn, parameters_dict=DEFAULT_PARAMETERS, generate_vtu=False, debug=False):
        self.debug = debug
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

    def run_drainage(self):
        self.parameter_manager.update_initial_cas(self.network)
        calculate_initial_values(self.network, self.pc_incrementer.get_pc_max())
        return run_drainage(self.network, self.pc_incrementer, self.generate_vtu, debug=self.debug)

    def run_imbibition(self):
        self.parameter_manager.update_equilibrium_cas(self.network)
        calculate_initial_values(self.network, self.pc_incrementer.get_pc_max())
        return run_imbibition(self.network, self.pc_incrementer, self.generate_vtu, debug=self.debug)


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


def pn_to_vtu(pn):
    unstructured_grid = pn_to_vtk(pn)
    file_name = "output.vtu"

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(unstructured_grid)
    writer.Write()


def pn_to_vtk(pn):
    n_throats = len(pn.throats())
    n_pores = len(pn.pores())

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(n_pores)
    vtk_points.SetData(numpy_to_vtk(pn["pore.coords"]))

    np_cells = np.empty([n_throats, 3], dtype=np.int32)
    for throat_id in pn.throats():
        neighbors_id_list = pn.find_connected_pores(throat_id)[0]
        np_cells[throat_id][0] = n_throats + neighbors_id_list[0]
        np_cells[throat_id][1] = n_throats + neighbors_id_list[1]
        np_cells[throat_id][2] = throat_id

    cell_array_data = np.column_stack((np.ones(n_throats) * 3, np_cells)).ravel()

    vtk_cell_array = vtk.vtkCellArray()
    vtk_cell_array.SetNumberOfCells(n_throats)
    vtk_cell_array.SetCells(n_throats, numpy_to_vtk(cell_array_data, deep=True, array_type=vtk.VTK_ID_TYPE))

    radius_data = numpy_to_vtk(np.concatenate((pn["throat.radius"], pn["pore.radius"])))
    radius_data.SetName("radius")

    sw_data = numpy_to_vtk(np.concatenate((pn["throat.water_saturation"], pn["pore.water_saturation"])))
    sw_data.SetName("Sw")

    throat_radius_data = numpy_to_vtk(pn["throat.radius"])
    throat_radius_data.SetName("RRR")

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(vtk_points)
    unstructured_grid.SetCells(vtk.VTK_TRIANGLE, vtk_cell_array)
    unstructured_grid.GetPointData().AddArray(radius_data)
    unstructured_grid.GetPointData().AddArray(sw_data)

    return unstructured_grid


def read_vtk(unstructured_grid):
    n_cells = unstructured_grid.GetNumberOfCells()
    print(f"Number of cells: {n_cells}")

    for i in range(n_cells):
        # print(f"AHHH 0")
        left_pore_id = unstructured_grid.GetCell(i).GetPointIds().GetId(0)
        # position = unstructured_grid.GetPoint(left_pore_id)
        # print(f"AHHH 1", position)
        right_pore_id = unstructured_grid.GetCell(i).GetPointIds().GetId(1)
        throat_id = unstructured_grid.GetCell(i).GetPointIds().GetId(2)
        left_pos = unstructured_grid.GetPoint(left_pore_id)
        right_pos = unstructured_grid.GetPoint(right_pore_id)
        # print(f"left_pore_id: {left_pore_id}")

        pore_id = 0
        for pore_position in (left_pore_id, right_pore_id):
            radius = unstructured_grid.GetPointData().GetArray("radius").GetComponent(pore_position, 0)
            # print(f"ID: {pore_id} - Radius: {radius}")
            sw = unstructured_grid.GetPointData().GetArray("Sw").GetComponent(pore_position, 0)
            print(f"ID: {pore_id} - Sw: {sw}, Radius: {radius}")
            pore_id += 1
