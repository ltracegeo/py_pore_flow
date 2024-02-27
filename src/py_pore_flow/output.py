import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


id_count = 0


def pn_to_vtu(pn, pc, sw, cycle):
    global id_count

    unstructured_grid = pn_to_vtk(pn, pc)
    ouput_dir_name = "Output_res"
    if not os.path.exists(ouput_dir_name):
        os.makedirs(ouput_dir_name)

    if cycle == 0:
        phase = "Init"
    elif cycle == 1:
        phase = "OInj"
    else:
        phase = "WInj"
    file_name = f"{ouput_dir_name}/{cycle}_{phase}{id_count:04}.vtu"

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(unstructured_grid)
    writer.SetCompressorTypeToNone()
    writer.SetDataModeToAscii()
    writer.Write()

    line = f"<!-- Animation data used by GeoSlicer. Deleting this comment will break compatibility. # Sw: {sw} Cycle: {cycle} -->"
    add_string_to_file(file_name, 4, line)

    id_count += 1


def pn_to_vtk(pn, pc):
    n_throats = len(pn.throats())
    n_pores = len(pn.pores())

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(n_pores)
    vtk_points.SetData(numpy_to_vtk(pn["pore.coords"]))

    np_cells = np.empty([n_throats, 3], dtype=np.int32)
    for throat_id in pn.throats():
        neighbors_id_list = pn.find_connected_pores(throat_id)[0]
        np_cells[throat_id][0] = neighbors_id_list[0]
        np_cells[throat_id][1] = neighbors_id_list[1]
        np_cells[throat_id][2] = throat_id

    cell_array_data = np.column_stack((np.full(n_throats, 3), np_cells)).ravel()

    vtk_cell_array = vtk.vtkCellArray()
    vtk_cell_array.SetNumberOfCells(n_throats)
    vtk_cell_array.SetCells(n_throats, numpy_to_vtk(cell_array_data, deep=True, array_type=vtk.VTK_ID_TYPE))

    pore_radius_data = numpy_to_vtk(pn["pore.radius"])
    pore_radius_data.SetName("radius")

    pore_sw_data = numpy_to_vtk(pn["pore.water_saturation"])
    pore_sw_data.SetName("Sw")

    throat_radius_data = numpy_to_vtk(pn["throat.radius"])
    throat_radius_data.SetName("RRR")

    throat_sw_data = numpy_to_vtk(pn["throat.water_saturation"])
    throat_sw_data.SetName("Sw")

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(vtk_points)
    unstructured_grid.GetPointData().AddArray(pore_radius_data)
    unstructured_grid.GetPointData().AddArray(pore_sw_data)
    unstructured_grid.SetCells(vtk.VTK_TRIANGLE, vtk_cell_array)
    unstructured_grid.GetCellData().AddArray(throat_radius_data)
    unstructured_grid.GetCellData().AddArray(throat_sw_data)

    pc_data = numpy_to_vtk(np.full(n_throats, pc))
    pc_data.SetName("Pc")
    unstructured_grid.GetCellData().AddArray(pc_data)

    return unstructured_grid


def add_string_to_file(file_path, line_number, new_content):
    lines = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    lines.insert(line_number - 1, new_content + "\n")

    with open(file_path, "w") as file:
        file.writelines(lines)
