import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


id_count = 0


def pn_to_vtu(pn):
    global id_count

    unstructured_grid = pn_to_vtk(pn)
    if not os.path.exists("Output_res"):
        os.makedirs("Output_res")
    file_name = f"Output_res/output_{id_count}.vtu"

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(file_name)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    id_count += 1


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
