import os

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from py_pore_flow.graphic.element_graphic import ElementGraphic
from py_pore_flow.network import PORE, THROAT


id_count = 0


def pn_to_vtu(pn, pc, sw, cycle):
    global id_count

    unstructured_grid = pn_to_vtk(pn, pc, sw, cycle)
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


def pn_to_vtk(pn, pc, sw, cycle):
    throat_id_list = pn.throats()
    n_throats = len(throat_id_list)
    n_pores = len(pn["pore.all"])

    cycle_data = vtk.vtkIntArray()
    cycle_data.SetName("Cycle")
    cycle_data.InsertNextValue(cycle)
    sw_data = vtk.vtkDoubleArray()
    sw_data.SetName("Sw")
    sw_data.InsertNextValue(sw)
    field_data = vtk.vtkFieldData()
    field_data.AddArray(cycle_data)
    field_data.AddArray(sw_data)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(n_pores)
    vtk_points.SetData(numpy_to_vtk(pn["pore.coords"]))

    np_cells = np.empty([n_throats, 2], dtype=np.int32)
    for i, throat_id in enumerate(throat_id_list):
        neighbors_id_list = pn.find_connected_pores(throat_id)[0]
        np_cells[i][0] = neighbors_id_list[0]
        np_cells[i][1] = neighbors_id_list[1]

    vtk_cell_array = vtk.vtkCellArray()
    for cell in np_cells:
        vtk_cell_array.InsertNextCell(2)
        vtk_cell_array.InsertCellPoint(cell[0])
        vtk_cell_array.InsertCellPoint(cell[1])

    pore_radius_data = numpy_to_vtk(pn["pore.radius"])
    pore_radius_data.SetName("radius")

    pore_sw_data = numpy_to_vtk(pn["pore.water_saturation"])
    pore_sw_data.SetName("Sw")

    pore_inlets_data = numpy_to_vtk(pn["pore.inlets"].astype(int))
    pore_inlets_data.SetName("inlets")

    pore_outlets_data = numpy_to_vtk(pn["pore.outlets"].astype(int))
    pore_outlets_data.SetName("outlets")

    throat_radius_data = numpy_to_vtk(pn["throat.radius"][throat_id_list])
    throat_radius_data.SetName("RRR")

    throat_sw_data = numpy_to_vtk(pn["throat.water_saturation"][throat_id_list])
    throat_sw_data.SetName("Sw")

    throat_condW_data = numpy_to_vtk(1e18 * pn["throat.water_multiphase_conductance"][throat_id_list])
    throat_condW_data.SetName("condW")

    throat_condO_data = numpy_to_vtk(1e18 * pn["throat.oil_multiphase_conductance"][throat_id_list])
    throat_condO_data.SetName("condO")

    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetFieldData(field_data)
    unstructured_grid.SetPoints(vtk_points)
    unstructured_grid.GetPointData().AddArray(pore_radius_data)
    unstructured_grid.GetPointData().AddArray(pore_sw_data)
    unstructured_grid.GetPointData().AddArray(pore_inlets_data)
    unstructured_grid.GetPointData().AddArray(pore_outlets_data)
    unstructured_grid.SetCells(vtk.VTK_ID_TYPE, vtk_cell_array)
    unstructured_grid.GetCellData().AddArray(throat_radius_data)
    unstructured_grid.GetCellData().AddArray(throat_sw_data)
    unstructured_grid.GetCellData().AddArray(throat_condW_data)
    unstructured_grid.GetCellData().AddArray(throat_condO_data)

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


def element_to_openfoam(pn, element_type, element_id):
    if element_type == PORE:
        element_data = pn.pore
        return
    else:
        element_data = pn.throat

    if element_data.n_corners[element_id] != 3:
        print("not triangle")
        return

    def get_element_vertices(element_data, element_id):
        angles = element_data.corner_half_angle[element_id][: element_data.n_corners[element_id]] * 2
        radius = element_data.radius[element_id]
        vertices = ElementGraphic.triangle_vertices(angles, radius)
        vertices = np.pad(vertices, ((0,0), (0,1)), "constant", constant_values=0)
        return vertices[0], vertices[1], vertices[2]

    A_coord, B_coord, C_coord = get_element_vertices(element_data, element_id)

    # Triangle in XY plane
    A = np.array(A_coord)
    B = np.array(B_coord)
    C = np.array(C_coord)

    if element_type == THROAT:
        length = element_data.length[element_id]
    else:
        length = 0

    # Extruded points
    A1 = A + [0, 0, length]
    B1 = B + [0, 0, length]
    C1 = C + [0, 0, length]

    # Create a block using 8 points
    # We'll add a centroid to make a degenerate hex block (OpenFOAM allows that with care)
    centroid = (A + B + C) / 3
    centroid1 = centroid + [0, 0, length]

    points = [A, B, C, centroid, A1, B1, C1, centroid1]

    def foam_vertex_str(i, p):
        return f"    ({p[0]:.6f} {p[1]:.6f} {p[2]:.6f}) // v{i}"

    with open("blockMeshDict", "w") as f:
        f.write(
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object blockMeshDict;\n}\n\n"
        )
        f.write("convertToMeters 1.0;\n\n")

        f.write("vertices\n(\n")
        for i, p in enumerate(points):
            f.write(foam_vertex_str(i, p) + "\n")
        f.write(");\n\n")

        f.write("blocks\n(\n")
        f.write("    hex (0 1 2 3 4 5 6 7) (1 1 1) simpleGrading (1 1 1)\n")
        f.write(");\n\n")

        f.write("edges()\n;\n\n")

        f.write("boundary\n(\n")
        f.write("    walls\n    {\n        type wall;\n        faces\n        (\n")
        f.write("            (0 1 5 4)\n")
        f.write("            (1 2 6 5)\n")
        f.write("            (2 0 4 6)\n")
        f.write("        );\n    }\n")
        f.write(");\n\n")

        f.write("mergePatchPairs()\n(\n);\n")
